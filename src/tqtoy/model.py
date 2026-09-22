"""Two-temperature, two-population 0D model of the thermal quench.

State vector (normalised by ``FLUID_SCALES`` and ``NIJ_SCALE``)::

    [Te_hot, Te_cold, ne_hot, ne_cold, Ti_hot, Ti_cold, ni_hot, ni_cold,
     n_{0,0} ... n_{Z0,0},  n_{0,1} ... n_{Z1,1},  ...]

where n_{i,j} is the density of charge state i of impurity element j.
Temperatures are in eV and densities in m^-3.

With ``model.populations = 1`` the single thermal population is stored in the
"cold" slots and the "hot" slots are frozen dummies (T = 1 eV, n = 1 m^-3).

Equations (per population, electrons e and main ions i)::

    d(ne)/dt = S_D + sum_ij i dn_ij/dt
    d(ni)/dt = S_D
    d(ne Te)/dt = 2/(3e) (P_ohm - P_rad - P_stoch - P_exch - c Te)
    d(ni Ti)/dt = 2/(3e) (P_exch - c Ti)
    dn_ij/dt : ionisation/recombination balance + impurity source

With two populations, the injected deuterons and the electrons released by impurity
ionisation are added to the COLD population, and the two populations exchange energy
collisionally and nothing else. No particle is transferred between them. The exchange
therefore relaxes the temperature difference symmetrically::

    d(Te_hot - Te_cold)/dt |_exchange = -2/(3e) P_ee (1/ne_hot + 1/ne_cold)

with P_ee proportional to (Te_hot - Te_cold). The full derivation is in
docs/MODEL_EQUATIONS.pdf.

``solver.state_variables`` chooses what the four thermal slots of the state vector
hold. With ``energy`` (default) they hold the energy densities n_e T_e and n_i T_i,
which are the two equations above as written. With ``temperature`` they hold T, which
divides them by n and adds the dilution term -T (dn/dt) / n. The two are the same
equations, one multiplied by n. Outputs are temperatures either way.

``TQModel.rhs`` is passed to the ODE solver. ``TQModel.evaluate`` returns every
intermediate term (powers, currents, sources) and is used by the diagnostics,
so the post-processing uses exactly the same physics as the solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .atomic import AtomicData, get_atomic_data
from .config import Config
from .constants import E_CHARGE, M_P
from .physics.ablation import D2_fraction, ablation_rate, sample_shard_radii
from .physics.collisions import exchange_power
from .physics.impurities import charge_state_derivative, radiated_power
from .physics.power import ohmic_power, stochastic_power
from .physics.resistivity import (eta_multi_species, eta_spitzer, linear_current_ramp,
                                  share_current, zeff)

FLUID_NAMES = ("Te_hot", "Te_cold", "ne_hot", "ne_cold", "Ti_hot", "Ti_cold", "ni_hot", "ni_cold")
FLUID_SCALES = (1e3, 1e3, 1e20, 1e20, 1e3, 1e3, 1e20, 1e20)
NIJ_SCALE = 1e19
N_FLUID = len(FLUID_NAMES)

# solver.state_variables = energy: the four temperature slots carry the energy density n T
# [eV m^-3] instead of T [eV]. THERMAL_SLOTS gives, for each of them, the slot of its own
# density, which turns one into the other.
THERMAL_SLOTS = ((0, 2), (1, 3), (4, 6), (5, 7))
ENERGY_SCALES = list(FLUID_SCALES)
for _k, _n in THERMAL_SLOTS:
    ENERGY_SCALES[_k] = FLUID_SCALES[_k] * FLUID_SCALES[_n]
ENERGY_SCALES = tuple(ENERGY_SCALES)                       # (1e23, 1e23, 1e20, 1e20, 1e23, ...)
_FD_STEP = float(np.sqrt(np.finfo(float).eps))      # forward difference of the fluid columns
_FD_REJECT = float(np.finfo(float).eps)**0.875      # a smaller difference is round-off (scipy)
_FD_RETRIES = 4                                     # larger steps tried when it is

# Default floor (solver.floor): temperatures and densities are floored at this value inside
# the right-hand side. At the floor, only the part of a derivative that would push a quantity
# further below its own floor is removed. See TQModel.evaluate.
FLOOR = 0.01


@dataclass(frozen=True)
class StateLayout:
    """Position of each quantity in the state vector."""

    atomic_numbers: tuple

    @property
    def offsets(self) -> List[int]:
        """Index of the neutral state of each element."""
        out, pos = [], N_FLUID
        for Z in self.atomic_numbers:
            out.append(pos)
            pos += Z + 1
        return out

    @property
    def size(self) -> int:
        return N_FLUID + sum(Z + 1 for Z in self.atomic_numbers)

    @property
    def z_max(self) -> int:
        return max(self.atomic_numbers)

    def scales(self, fluid_scales=FLUID_SCALES) -> np.ndarray:
        return np.array(list(fluid_scales) + [NIJ_SCALE] * (self.size - N_FLUID))

    def nij_from_vector(self, y) -> np.ndarray:
        """Charge-state densities (Z_max+1, n_elements) [m^-3] from a normalised state vector."""
        nij = np.zeros([self.z_max + 1, len(self.atomic_numbers)])
        for j, (Z, off) in enumerate(zip(self.atomic_numbers, self.offsets)):
            nij[:Z + 1, j] = np.asarray(y[off:off + Z + 1]) * NIJ_SCALE
        return nij


class TQModel:
    """Right-hand side of the thermal-quench ODE system for one configuration."""

    def __init__(self, config: Config, atomic: Optional[AtomicData] = None, provider=None):
        self.config = config
        self.atomic = atomic if atomic is not None else get_atomic_data(config.element_names, provider)
        if len(self.atomic.elements) != len(config.impurities):
            raise ValueError("The atomic data do not match the impurity elements of the configuration.")
        self.rates = self.atomic.rates
        self.Z = tuple(self.atomic.atomic_numbers)
        self.layout = StateLayout(self.Z)

        m, inj, p = config.model, config.injection, config.plasma
        self.two_populations = m.populations == 2
        self.block_ei = m.block_electron_ion_exchange
        self.multi_species = m.resistivity == "multi_species"
        # The multi-species conductivity already carries its own ne_hot factor.
        self.weighted_sharing = m.current_sharing == "density_weighted" and not self.multi_species
        self.rate_multiplier = m.atomic_rate_multiplier
        self.loss_coef = m.linear_loss_coefficient
        self.stochastic = m.stochastic
        self.m_ion = m.main_ion_mass * M_P
        rl = m.radiation_limit
        self.rad_cap = (rl.Te_below if rl.enabled else None, rl.time_scale)
        self.floor = config.solver.floor
        self.energy_state = config.solver.state_variables == "energy"
        self.fluid_scales = ENERGY_SCALES if self.energy_state else FLUID_SCALES
        self.atol = config.solver.atol          # threshold of the Jacobian difference step
        self.J0 = p.J
        self.J_ramp = (p.J_final, p.J_ramp_time) if p.J_final is not None else None
        self.B = p.B

        # Injection window and prescribed sources
        self.charge_state = inj.charge_state
        for Z, name in zip(self.Z, config.element_names):
            if self.charge_state > Z:
                raise ValueError(f"injection.charge_state = {self.charge_state} exceeds Z = {Z} of {name}.")
        self.t_start = inj.t_start
        self.t_stop = inj.t_start + inj.duration
        dt_inj = self.t_stop - self.t_start
        imp = list(config.impurities.values())
        self.use_ablation = inj.source == "ablation"
        self.S_D = inj.n_D / dt_inj
        self.S_imp = [c.injected_atom_density / dt_inj for c in imp]

        # Ablation source
        self.r_p_samples = None
        if self.use_ablation:
            ab = inj.ablation
            self.X = ab.D2_fraction if ab.D2_fraction is not None else D2_fraction(inj.n_D, imp[0].injected_atom_density)
            self.ab = ab
            if ab.parks is not None:
                pk = ab.parks
                self.r_p_samples = sample_shard_radii(ab.n_shards, pk.n_atoms, pk.solid_density,
                                                      pk.r_min, pk.r_max, seed=ab.seed)

        # Initial state
        nij0 = np.zeros([self.layout.z_max + 1, len(self.Z)])
        ne_from_imp = 0
        for j, c in enumerate(imp):
            nij0[self.charge_state, j] = c.background_density
            ne_from_imp += self.charge_state * c.background_density
        Ti0 = p.Ti0 if p.Ti0 is not None else p.Te0
        ni0 = p.ni0 if p.ni0 is not None else p.ne0
        if self.two_populations:
            fluid0 = [p.Te0, p.cold_T0, p.ne0, p.cold_n0 + ne_from_imp, Ti0, p.cold_T0, ni0, p.cold_n0]
        else:
            fluid0 = [1.0, p.Te0, 1.0, p.ne0 + ne_from_imp, 1.0, Ti0, 1.0, ni0]
        state0 = list(fluid0)
        if self.energy_state:                          # the thermal slots carry n T, not T
            for k, n in THERMAL_SLOTS:
                state0[k] = fluid0[k] * fluid0[n]
        y0 = [v / s for v, s in zip(state0, self.fluid_scales)]
        for j, Z in enumerate(self.Z):
            for i in range(0, Z + 1):
                y0.append(nij0[i, j] / NIJ_SCALE)
        self.y0 = np.array(y0)

    # ------------------------------------------------------------------ helpers
    def _clip(self, values: Sequence[float]):
        """Apply the floor to the 8 fluid quantities. Returns (values, at_floor flags).

        A quantity at or below the floor raises its flag. What the flag does to the derivatives
        is described in :meth:`evaluate`.
        """
        floor = self.floor
        vals = [v if v > floor else floor for v in values]
        flags = [v <= floor for v in values]
        return vals, flags

    def fluid_from_state(self, y) -> List[float]:
        """The 8 fluid quantities in physical units, before the floor is applied.

        With ``solver.state_variables = energy`` the thermal slots hold n T [eV m^-3] and the
        temperature is recovered as T = (n T) / n. The density is floored first, so that a
        population reduced to nothing gives a finite temperature instead of 0 / 0.
        """
        values = [y[k] * self.fluid_scales[k] for k in range(N_FLUID)]
        if self.energy_state:
            floor = self.floor
            for k, n in THERMAL_SLOTS:
                density = values[n]
                values[k] = values[k] / (density if density > floor else floor)
        return values

    def state_from_fluid(self, values: Sequence[float]) -> List[float]:
        """The 8 normalised thermal and density slots from physical temperatures and densities.

        Inverse of :meth:`fluid_from_state`, up to the floor applied there.
        """
        out = list(values)
        if self.energy_state:
            for k, n in THERMAL_SLOTS:
                out[k] = values[k] * values[n]
        return [v / s for v, s in zip(out, self.fluid_scales)]

    def _unpack(self, y):
        vals, flags = self._clip(self.fluid_from_state(y))
        return vals, flags, self.layout.nij_from_vector(y)

    def current_density(self, t: float) -> float:
        if self.J_ramp is None:
            return self.J0
        return linear_current_ramp(self.J0, t, self.J_ramp[0], self.J_ramp[1])

    def _stochastic(self, ne, Te):
        s = self.stochastic
        if not s.enabled:
            return 0.0
        return stochastic_power(ne, Te, s.deltaB_over_B, s.r, s.a, s.coulomb_log)

    def _dTe(self, ne, Te, Dne, P_ohm, P_rad, P_in, P_st):
        """Electron thermal derivative: d(n_e T_e)/dt in energy mode, dT_e/dt otherwise.

        The two forms are the same equation, one multiplied by n_e. The energy form carries no
        1 / n_e and no dilution term -T_e (dn_e/dt) / n_e, so it stays regular for a population
        born almost empty (see ``solver.state_variables``).
        """
        power = (2 / (3 * E_CHARGE)) * (P_ohm - P_rad + P_in - P_st - self.loss_coef * Te)
        if self.energy_state:
            return power
        return (power - Dne * Te) / ne

    def _dTi(self, ni, Ti, Dni, P_in):
        """Ion thermal derivative: d(n_i T_i)/dt in energy mode, dT_i/dt otherwise."""
        power = (2 / (3 * E_CHARGE)) * (P_in - self.loss_coef * Ti)
        if self.energy_state:
            return power
        return (power - Dni * Ti) / ni

    # ------------------------------------------------------------------ core
    def evaluate(self, t: float, vals, flags, nij, full: bool = False):
        """Time derivatives (physical units) and, if ``full``, all intermediate terms.

        Returns (d_fluid, d_nij, terms) with d_fluid the 8 fluid derivatives in the
        order of FLUID_NAMES and terms a dict (empty unless ``full``).
        """
        Te_hot, Te_cold, ne_hot, ne_cold, Ti_hot, Ti_cold, ni_hot, ni_cold = vals
        fl_Te_hot, fl_Te_cold, fl_ne_hot, fl_ne_cold, fl_Ti_hot, fl_Ti_cold, fl_ni_hot, fl_ni_cold = flags

        def limit_n(value, at_floor_n, at_floor_T):
            """Density derivative at the floor: only the part that would push it lower is removed.

            A quantity is limited by its own floor alone. A population whose temperature has
            collapsed therefore keeps collecting the particles injected into it, and a density
            cannot run away into negative values.
            """
            return max(value, 0.0) if at_floor_n else value

        def limit_T(value, at_floor_n, at_floor_T):
            """Thermal derivative at the floor.

            The density floor switches the equation off. In temperature mode because it carries
            1 / n, and in energy mode because the state holds n T while the temperature is read
            back as (n T) / n, a ratio that is noise once the density is pinned at the floor.
            Holding both keeps that ratio at its last finite value.
            """
            if at_floor_n:
                return 0.0
            return max(value, 0.0) if at_floor_T else value

        # A species whose temperature is at the floor still has a meaningful particle balance,
        # so the density equations below are only clamped, never frozen.
        need_e_cold = fl_ne_cold
        need_e_hot = fl_ne_hot
        two = self.two_populations
        rates = self.rates

        # --- Sources
        g_hot = g_cold = 0.0
        if self.use_ablation:
            ab = self.ab
            if two:
                g_hot = ablation_rate(ne_hot, Te_hot, self.X, ab.volume, ab.shard_radius, ab.n_shards, t,
                                      r_p_samples=self.r_p_samples, B=self.B, deuterium=ab.deuterium_coefficients,
                                      switch_on_time=ab.switch_on_time)
            g_cold = ablation_rate(ne_cold, Te_cold, self.X, ab.volume, ab.shard_radius, ab.n_shards, t,
                                   r_p_samples=self.r_p_samples, B=self.B, deuterium=ab.deuterium_coefficients,
                                   switch_on_time=ab.switch_on_time)
            g = g_cold + g_hot
            S_imp = [g * (1 - self.X)]
            S_D = g * self.X * 2
        else:
            S_imp, S_D = self.S_imp, self.S_D
        injecting = (t >= self.t_start) & (t <= self.t_stop)
        src = S_imp if injecting else None

        # --- Impurity charge states
        q, mult = self.charge_state, self.rate_multiplier
        if two:
            Dnij_cold = charge_state_derivative(rates, nij, ne_cold, Te_cold, None, q, mult)
            Dnij_hot = charge_state_derivative(rates, nij, ne_hot, Te_hot, src, q, mult)
            Dnij = Dnij_cold + Dnij_hot
        else:
            Dnij = charge_state_derivative(rates, nij, ne_cold, Te_cold, src, q, mult)

        # --- Thermal exchange (P_a_b > 0: power flowing from a to b)
        mi = self.m_ion
        P_eh_ec = P_ih_ic = P_eh_ih = P_eh_ic = P_ec_ic = P_ec_ih = 0.0
        if two:
            P_eh_ec = exchange_power(ne_hot, Te_hot, ne_cold, Te_cold, "ee", Te_hot, ne_hot, mi)
            P_ih_ic = exchange_power(ni_hot, Ti_hot, ni_cold, Ti_cold, "ii", Ti_hot, ni_hot, mi)
            if not self.block_ei:
                P_eh_ih = exchange_power(ne_hot, Te_hot, ni_hot, Ti_hot, "ei", Te_hot, ne_hot, mi)
                P_eh_ic = exchange_power(ne_hot, Te_hot, ni_cold, Ti_cold, "ei", Te_hot, ne_hot, mi)
                P_ec_ic = exchange_power(ne_cold, Te_cold, ni_cold, Ti_cold, "ei", Te_cold, ne_cold, mi)
                P_ec_ih = exchange_power(ne_cold, Te_cold, ni_hot, Ti_hot, "ei", Te_cold, ne_cold, mi)
        elif not self.block_ei:
            P_ec_ic = exchange_power(ne_cold, Te_cold, ni_cold, Ti_cold, "ei", Te_cold, ne_cold, mi)

        # --- Densities
        def free_electron_source(S0):
            Dne = S0
            for j, Z in enumerate(self.Z):
                for i in range(0, Z + 1):
                    Dne += i * Dnij[i, j]
            return Dne

        # The two populations exchange energy collisionally and nothing else. No particle is
        # transferred between them, so their particle content changes only through the
        # injection and the impurity ionisation balance. See docs/MODEL_EQUATIONS.pdf.
        Dne_hot = Dni_hot = 0.0
        if injecting:
            Dne_cold = limit_n(free_electron_source(S_D), fl_ne_cold, fl_Te_cold)
            Dni_cold = limit_n(S_D, fl_ni_cold, fl_Ti_cold)
        elif two and t < self.t_start:
            Dne_cold = 0.0                      # the cold population does not exist yet
            Dni_cold = 0.0
        else:
            Dne_cold = limit_n(free_electron_source(0.0), fl_ne_cold, fl_Te_cold)
            Dni_cold = 0.0

        J = self.current_density(t)

        # --- Temperatures
        if not two:
            need = full or not need_e_cold
            Z_eff = zeff(ni_cold, nij, self.Z)
            eta_hot, eta_cold = np.inf, eta_spitzer(Te_cold, Z_eff, ne_cold)
            J_hot, J_cold = 0.0, J
            P_ohm_hot = P_rad_hot = P_st_hot = 0.0
            if need:
                P_ohm_cold = ohmic_power(J, eta_cold)
                P_rad_cold = radiated_power(rates, nij, ne_cold, Te_cold, *self.rad_cap)
                P_st_cold = self._stochastic(ne_cold, Te_cold)
            DTe_cold = limit_T(self._dTe(ne_cold, Te_cold, Dne_cold, P_ohm_cold, P_rad_cold,
                                         -P_ec_ic, P_st_cold) if need else 0.0, fl_ne_cold, fl_Te_cold)
            DTi_cold = limit_T(self._dTi(ni_cold, Ti_cold, Dni_cold, P_ec_ic), fl_ni_cold, fl_Ti_cold)
            d_fluid = [0.0, DTe_cold, 0.0, Dne_cold, 0.0, DTi_cold, 0.0, Dni_cold]
        else:
            Z_eff = zeff(ni_hot + ni_cold, nij, self.Z)
            if self.multi_species:
                eta_hot, eta_cold = eta_multi_species(Te_hot, Te_cold, Ti_hot, Ti_cold, ne_hot, ne_cold,
                                                      ni_hot, ni_cold, Z_eff, self.m_ion)
            else:
                eta_hot = eta_spitzer(Te_hot, Z_eff, ne_hot)
                eta_cold = eta_spitzer(Te_cold, Z_eff, ne_cold)
                if self.weighted_sharing:
                    # Each population carries the current in proportion to its own electron
                    # density. The Spitzer resistivity does not depend on the electron density,
                    # because the n_e of sigma = n_e e^2 tau / m_e cancels against tau
                    # proportional to 1 / n_e. That cancellation holds for a complete plasma.
                    # For a sub-population the collision frequency is set by the total ion
                    # density, so its conductivity is sigma_p = (n_e,p / n_e,tot) / eta(T_p).
                    # The two conductivities add up to 1 / eta when the temperatures are equal.
                    # Without this weight, a population emptied of its electrons still carries a
                    # finite share of the current (model.current_sharing).
                    ne_tot = ne_hot + ne_cold
                    eta_hot = eta_hot * ne_tot / ne_hot
                    eta_cold = eta_cold * ne_tot / ne_cold
            J_hot, J_cold = share_current(J, eta_hot, eta_cold)
            cold_active = t >= self.t_start
            need_hot = full or not need_e_hot
            need_cold = full or (cold_active and not need_e_cold)
            if need_hot:
                P_ohm_hot = ohmic_power(J_hot, eta_hot)
                P_rad_hot = radiated_power(rates, nij, ne_hot, Te_hot, *self.rad_cap)
                P_st_hot = self._stochastic(ne_hot, Te_hot)
            if need_cold:
                P_ohm_cold = ohmic_power(J_cold, eta_cold)
                P_rad_cold = radiated_power(rates, nij, ne_cold, Te_cold, *self.rad_cap)
                P_st_cold = self._stochastic(ne_cold, Te_cold)
            DTe_hot = limit_T(self._dTe(ne_hot, Te_hot, Dne_hot, P_ohm_hot, P_rad_hot,
                                        -P_eh_ih - P_eh_ic - P_eh_ec, P_st_hot) if need_hot else 0.0,
                              fl_ne_hot, fl_Te_hot)
            DTi_hot = limit_T(self._dTi(ni_hot, Ti_hot, Dni_hot, P_eh_ih + P_ec_ih - P_ih_ic),
                              fl_ni_hot, fl_Ti_hot)
            if cold_active:
                DTe_cold = limit_T(self._dTe(ne_cold, Te_cold, Dne_cold, P_ohm_cold, P_rad_cold,
                                             -P_ec_ic - P_ec_ih + P_eh_ec, P_st_cold) if need_cold else 0.0,
                                   fl_ne_cold, fl_Te_cold)
                DTi_cold = limit_T(self._dTi(ni_cold, Ti_cold, Dni_cold, P_ec_ic + P_eh_ic + P_ih_ic),
                                   fl_ni_cold, fl_Ti_cold)
            else:
                DTe_cold = 0
                DTi_cold = 0
            d_fluid = [DTe_hot, DTe_cold, Dne_hot, Dne_cold, DTi_hot, DTi_cold, Dni_hot, Dni_cold]

        terms: Dict[str, float] = {}
        if full:
            terms = dict(
                J=J, J_hot=J_hot, J_cold=J_cold, Zeff=Z_eff, eta_hot=eta_hot, eta_cold=eta_cold,
                E_hot=eta_hot * J_hot if two else np.nan, E_cold=eta_cold * J_cold,
                P_ohm_hot=P_ohm_hot, P_ohm_cold=P_ohm_cold,
                P_rad_hot=P_rad_hot, P_rad_cold=P_rad_cold,
                P_stoch_hot=P_st_hot, P_stoch_cold=P_st_cold,
                P_loss_lin_e=self.loss_coef * (Te_cold + (Te_hot if two else 0.0)),
                P_loss_lin_i=self.loss_coef * (Ti_cold + (Ti_hot if two else 0.0)),
                P_ehot_ecold=P_eh_ec, P_ihot_icold=P_ih_ic, P_ehot_ihot=P_eh_ih,
                P_ehot_icold=P_eh_ic, P_ecold_icold=P_ec_ic, P_ecold_ihot=P_ec_ih,
                S_D=S_D if injecting else 0.0,
                S_imp=float(np.sum(S_imp)) if injecting else 0.0,
                g_hot=g_hot, g_cold=g_cold,
            )
        return d_fluid, Dnij, terms

    # ------------------------------------------------------------------ Jacobian
    def _balance_operator(self, ne, Te):
        """Charge-state balance as a linear operator, at fixed electron density and temperature.

        At fixed (ne, Te) the balance is ``dn/dt = M n`` for each element, with M tridiagonal:
        M[i+1, i] = S_i ne (ionisation), M[i, i+1] = a_{i+1} ne (recombination) and the
        diagonal the sum of the two losses. Returns, per element, (M, w) with
        ``w[k] = sum_i i M[i, k] = (S_k - a_k) ne``, the number of free electrons released
        per unit density of the state k.
        """
        blocks = []
        for el in self.rates:
            Z = el.Z
            S, a, _, _ = el.coefficients(ne, Te)
            M = np.zeros((Z + 1, Z + 1))
            k = np.arange(Z)
            M[k + 1, k] = S * ne
            M[k, k + 1] = a * ne
            M[k, k] -= S * ne
            M[k + 1, k + 1] -= a * ne
            w = np.zeros(Z + 1)
            w[:Z] += S * ne
            w[1:] -= a * ne
            blocks.append((M, w))
        return blocks

    def _radiation_gradient(self, ne, Te, nij):
        """d(P_rad)/d(n_ij) [W m^-3 per m^-3] per element, and whether the limiter is active.

        The radiated power is linear in the charge-state densities at fixed (ne, Te):
        P = sum_ij ne n_ij (PLT_i + PRB_i). The low-temperature limiter replaces it by a
        constant when it applies, and the gradient is then zero.
        """
        gradients, power = [], 0.0
        for j, el in enumerate(self.rates):
            Z = el.Z
            _, _, line, continuum = el.coefficients(ne, Te)
            g = np.zeros(Z + 1)
            g[:Z] += line
            g[1:] += continuum
            g *= ne
            gradients.append(g)
            power += float(g @ nij[:Z + 1, j])
        cap_Te, cap_time = self.rad_cap
        limited = cap_Te is not None and Te < cap_Te and power > 3/2*ne*Te*E_CHARGE/cap_time
        return gradients, limited

    def _eta_slopes(self, Z_eff, vals):
        """(eta_hot, eta_cold) and their derivatives with respect to Z_eff.

        The derivative is a central difference in Z_eff alone: the two resistivity models
        would otherwise each need their own closed form, for a quantity that is a single
        scalar function of Z_eff at fixed temperatures and densities.
        """
        Te_hot, Te_cold, ne_hot, ne_cold, Ti_hot, Ti_cold, ni_hot, ni_cold = vals

        def etas(Z):
            if not self.two_populations:
                return np.inf, eta_spitzer(Te_cold, Z, ne_cold)
            if self.multi_species:
                return eta_multi_species(Te_hot, Te_cold, Ti_hot, Ti_cold, ne_hot, ne_cold,
                                         ni_hot, ni_cold, Z, self.m_ion)
            hot, cold = eta_spitzer(Te_hot, Z, ne_hot), eta_spitzer(Te_cold, Z, ne_cold)
            if self.weighted_sharing:      # the weights do not depend on Z_eff
                ne_tot = ne_hot + ne_cold
                hot, cold = hot * ne_tot / ne_hot, cold * ne_tot / ne_cold
            return hot, cold

        eta_hot, eta_cold = etas(Z_eff)
        dZ = 1e-6 * Z_eff
        up, down = etas(Z_eff + dZ), etas(Z_eff - dZ)
        d_hot = 0.0 if not self.two_populations else (up[0] - down[0]) / (2 * dZ)
        return eta_hot, eta_cold, d_hot, (up[1] - down[1]) / (2 * dZ)

    def jacobian(self, t, y):
        """Jacobian of :meth:`rhs` (normalised variables), used when solver.jacobian = analytic.

        The columns of the impurity charge states are analytic. At fixed electron density and
        temperature the charge-state balance is linear in those densities, and the fluid
        equations depend on them only through the free-electron source, Z_eff (hence the
        resistivities, the current sharing and the ohmic power) and the radiated power.
        The eight fluid columns are forward differences: the collision times, the ablation
        law and the resistivities would each need a derivative of their own, for eight
        columns out of ``8 + sum(Z + 1)``.

        The Jacobian only drives the Newton iteration and the step-size control of the
        implicit solver. An inaccurate one costs steps. It does not move the solution of
        the collocation equations, which is set by ``rtol`` and ``atol``.
        """
        y = np.array(y, dtype=float)
        n = self.layout.size
        jac = np.zeros((n, n))
        vals, flags, nij = self._unpack(y)
        Te_hot, Te_cold, ne_hot, ne_cold, Ti_hot, Ti_cold, ni_hot, ni_cold = vals
        fl_Te_hot, fl_Te_cold, fl_ne_hot, fl_ne_cold = flags[0], flags[1], flags[2], flags[3]
        two = self.two_populations
        mult = self.rate_multiplier
        offsets, Zs = self.layout.offsets, self.Z

        # Rows held at a constant by the floor. A temperature is frozen by the density floor,
        # and a quantity sitting on its own floor is frozen only while its derivative points
        # below it, which the limited derivative reports as an exact zero.
        f0 = self._rhs_at(t, y)
        dead_Te_hot = fl_ne_hot or (fl_Te_hot and f0[0] == 0.0)
        dead_Te_cold = fl_ne_cold or (fl_Te_cold and f0[1] == 0.0)
        dead_ne_cold = fl_ne_cold and f0[3] == 0.0

        # --- charge-state balance and radiated power, per population
        blocks = self._balance_operator(ne_cold, Te_cold)
        grad_cold, limited_cold = self._radiation_gradient(ne_cold, Te_cold, nij)
        if two:
            hot_blocks = self._balance_operator(ne_hot, Te_hot)
            blocks = [(Mc + Mh, wc + wh) for (Mc, wc), (Mh, wh) in zip(blocks, hot_blocks)]
            grad_hot, limited_hot = self._radiation_gradient(ne_hot, Te_hot, nij)

        # --- Z_eff and the resistive chain
        ni_main = ni_hot + ni_cold if two else ni_cold
        numerator = denominator = ni_main
        for j, Z in enumerate(Zs):                # Z_eff and its denominator in one pass
            states = np.arange(Z + 1, dtype=float)
            column = nij[:Z + 1, j]
            numerator += float((states * states) @ column)
            denominator += float(states @ column)
        Z_eff = numerator / denominator
        eta_hot, eta_cold, d_eta_hot, d_eta_cold = self._eta_slopes(Z_eff, vals)
        J_current = self.current_density(t)
        if two:
            total = eta_hot + eta_cold
            J_hot, J_cold = share_current(J_current, eta_hot, eta_cold)
            # The sharing term is kept for exactness but is negligible: both resistivities carry
            # the same Z_eff factor, so the numerator of d_J_hot cancels to about 1e-11 of the
            # first term. An error there cannot be seen in the Jacobian.
            d_J_hot = J_current * (d_eta_cold * eta_hot - eta_cold * d_eta_hot) / total**2
            d_P_ohm_hot = d_eta_hot * J_hot**2 + 2 * eta_hot * J_hot * d_J_hot
            d_P_ohm_cold = d_eta_cold * J_cold**2 - 2 * eta_cold * J_cold * d_J_hot
        else:
            d_P_ohm_cold = d_eta_cold * J_current**2

        # --- rows of the fluid equations that depend on the charge states
        i_Te_hot, i_Te_cold, i_ne_cold = 0, 1, 3
        cold_active = not two or t >= self.t_start          # the cold population has been created
        ne_cold_source = cold_active and not dead_ne_cold
        Te_cold_source = cold_active and not dead_Te_cold
        two_thirds_e = 2 / (3 * E_CHARGE)
        for j, Z in enumerate(Zs):
            off = offsets[j]
            M, w = blocks[j]
            states = np.arange(Z + 1, dtype=float)
            jac[off:off + Z + 1, off:off + Z + 1] = mult * M
            d_Zeff = states * (states - Z_eff) / denominator
            if ne_cold_source:
                jac[i_ne_cold, off:off + Z + 1] = mult * w
            if Te_cold_source:
                # Energy mode differentiates d(n_e T_e)/dt = (2/3e) P: no 1 / n_e and no
                # dilution term. Temperature mode differentiates that same row divided by n_e.
                d_rad = np.zeros(Z + 1) if limited_cold else grad_cold[j]
                row = two_thirds_e * (d_P_ohm_cold * d_Zeff - d_rad)
                jac[i_Te_cold, off:off + Z + 1] = (
                    row if self.energy_state else (row - Te_cold * mult * w) / ne_cold)
            if two and not dead_Te_hot:
                d_rad = np.zeros(Z + 1) if limited_hot else grad_hot[j]
                row = two_thirds_e * (d_P_ohm_hot * d_Zeff - d_rad)
                jac[i_Te_hot, off:off + Z + 1] = row if self.energy_state else row / ne_hot
        # --- normalisation of the analytic columns, then the fluid columns by differences
        scales = self.layout.scales(self.fluid_scales)
        jac[:, N_FLUID:] *= scales[N_FLUID:] / scales[:, None]
        # With one population the hot slots are inactive dummies: their rows and columns are zero.
        columns = range(N_FLUID) if two else (i_Te_cold, 3, 5, 7)
        for c in columns:
            # Step of scipy.integrate._ivp.common.num_jac: relative to the component, floored at
            # the absolute tolerance, and directed like the derivative it belongs to. A column
            # whose difference is lost to round-off is retried with a larger step, as scipy does.
            scale = max(self.atol, abs(y[c])) * (1.0 if f0[c] >= 0 else -1.0)
            shifted = y.copy()
            factor = _FD_STEP
            column = None
            for _ in range(_FD_RETRIES):
                shifted[c] = y[c] + factor * scale
                step = shifted[c] - y[c]
                if step != 0.0:
                    f_new = self._rhs_at(t, shifted)
                    difference = f_new - f0
                    k = int(np.argmax(np.abs(difference)))
                    if abs(difference[k]) >= _FD_REJECT * max(abs(f0[k]), abs(f_new[k])):
                        column = difference / step
                        break
                    column = difference / step
                factor *= 10.0
            jac[:, c] = column if column is not None else 0.0
        return jac

    def rhs(self, t, y):
        """Normalised time derivative of the state vector (for scipy.integrate.solve_ivp)."""
        vals, flags, nij = self._unpack(y)
        d_fluid, Dnij, _ = self.evaluate(t, vals, flags, nij)
        ds_dt = [d / s for d, s in zip(d_fluid, self.fluid_scales)]
        for j, Z in enumerate(self.Z):
            for i in range(0, Z + 1):
                ds_dt.append(Dnij[i, j] / NIJ_SCALE)
        # Radau and LSODA ask for the Jacobian at the point they have just evaluated; BDF asks
        # at its predicted state, where the right-hand side was never evaluated.
        self._last_rhs = (t, np.array(y, dtype=float), np.array(ds_dt))
        return ds_dt

    def _rhs_at(self, t, y):
        """Right-hand side at (t, y), reusing the value of the last call when it is the same point."""
        last = getattr(self, "_last_rhs", None)
        if last is not None and last[0] == t and np.array_equal(last[1], y):
            return last[2].copy()               # never hand out the cached array itself
        return np.asarray(self.rhs(t, y), dtype=float)

    def terms_along(self, t, fluid: Dict[str, np.ndarray], nij: np.ndarray) -> Dict[str, np.ndarray]:
        """Evaluate all intermediate terms along a stored trajectory.

        fluid : dict of 1D arrays with keys FLUID_NAMES [physical units]
        nij : array (n_t, Z_max+1, n_elements) [m^-3]
        """
        n_t = len(t)
        out: Dict[str, np.ndarray] = {}
        for k in range(n_t):
            vals, flags = self._clip([fluid[name][k] for name in FLUID_NAMES])
            _, _, terms = self.evaluate(t[k], vals, flags, nij[k], full=True)
            for key, v in terms.items():
                if key not in out:
                    out[key] = np.full(n_t, np.nan)
                out[key][k] = v
        return out
