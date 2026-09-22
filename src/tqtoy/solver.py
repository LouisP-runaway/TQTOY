"""Time integration of one configuration."""

from __future__ import annotations

import time as _time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.integrate import solve_ivp

from .atomic import AtomicData
from .config import Config
from .model import FLUID_NAMES, N_FLUID, NIJ_SCALE, THERMAL_SLOTS, TQModel

STATUS_COMPLETED = 0   # reached the last output time
STATUS_STOPPED = 1     # stopped by the event (a fluid quantity became negative or NaN)
STATUS_FAILED = -1     # solver failure
STATUS_NOT_RUN = -2    # scan point not computed
STATUS_LABELS = {STATUS_COMPLETED: "completed", STATUS_STOPPED: "stopped (negative value)",
                 STATUS_FAILED: "failed", STATUS_NOT_RUN: "not run"}


@dataclass
class RunResult:
    """Time traces of one simulation (physical units: s, eV, m^-3)."""

    t: np.ndarray
    fluid: Dict[str, np.ndarray]          # keys: FLUID_NAMES
    nij: np.ndarray                       # (n_t, Z_max+1, n_elements)
    elements: list
    atomic_numbers: list
    config: Config
    status: int = STATUS_COMPLETED
    message: str = ""
    elapsed: float = 0.0                  # wall-clock time [s]
    r_p_samples: Optional[np.ndarray] = None
    energies: Optional[Dict[str, np.ndarray]] = None   # cumulated energies at the output times [J m^-3]
    fhte: Optional[object] = None                      # tqtoy.fhte.FHTEResult (hot-tail estimate), if computed

    def __getattr__(self, name):
        fluid = self.__dict__.get("fluid", {})
        if name in fluid:
            return fluid[name]
        raise AttributeError(name)

    @property
    def two_populations(self) -> bool:
        return self.config.model.populations == 2

    @property
    def ne_total(self) -> np.ndarray:
        """Free-electron density of the active populations [m^-3]."""
        return self.ne_cold + self.ne_hot if self.two_populations else self.ne_cold

    @property
    def Te_avg(self) -> np.ndarray:
        """Density-weighted electron temperature of the active populations [eV]."""
        if not self.two_populations:
            return self.Te_cold.copy()
        return (self.Te_hot * self.ne_hot + self.Te_cold * self.ne_cold) / (self.ne_hot + self.ne_cold)

    @property
    def n_imp(self) -> np.ndarray:
        """Total impurity density of each element (n_t, n_elements) [m^-3]."""
        return np.sum(self.nij, axis=1)

    @property
    def ok(self) -> bool:
        return self.status in (STATUS_COMPLETED, STATUS_STOPPED)


def _event_factory(model: TQModel):
    n_check = N_FLUID if model.two_populations else 2
    # The derivative of a quantity sitting on its floor cannot push it below, so the exact
    # solution stays non-negative. The floor itself, 0.01 m^-3, is 1e-22 of the density scale,
    # which double precision cannot distinguish from zero at the scale of the state vector. An
    # excursion below zero there is dense-output noise, not a solution, so the event is offset
    # by the absolute tolerance, the smallest value the solver resolves.
    margin = model.atol

    def event_negative(t, y):
        """Crosses zero when a fluid quantity becomes significantly negative. Returns -1 on NaN."""
        head = y[:n_check]
        if np.any(np.isnan(head)):
            return -1.0
        return min(head) + margin

    event_negative.terminal = True
    event_negative.direction = -1
    return event_negative


ENERGY_KEYS = ("E_ohm", "E_rad", "E_stoch", "E_lin")


def _energy_integrals(model: TQModel, sol, t_out: np.ndarray, refine: int) -> Dict[str, np.ndarray]:
    """Cumulated ohmic, radiated, stochastic and linear-loss energies at the output times [J m^-3].

    The powers are integrated on the internal solver steps, each subdivided into
    ``refine`` intervals, using the dense output of the solver. This is much more
    accurate than integrating the powers sampled at the output times only.
    """
    ts = np.asarray(sol.sol.ts)
    frac = np.linspace(0.0, 1.0, refine + 1)[:-1]
    fine = (ts[:-1, None] + np.diff(ts)[:, None] * frac[None, :]).ravel()
    fine = np.unique(np.concatenate([fine, t_out]))
    fine = fine[(fine >= t_out[0]) & (fine <= t_out[-1])]
    y = sol.sol(fine)
    P = np.zeros((4, fine.size))
    for k in range(fine.size):
        vals, flags, nij = model._unpack(y[:, k])
        _, _, tm = model.evaluate(fine[k], vals, flags, nij, full=True)
        P[0, k] = tm["P_ohm_hot"] + tm["P_ohm_cold"]
        P[1, k] = tm["P_rad_hot"] + tm["P_rad_cold"]
        P[2, k] = tm["P_stoch_hot"] + tm["P_stoch_cold"]
        P[3, k] = tm["P_loss_lin_e"] + tm["P_loss_lin_i"]
    cum = np.concatenate([np.zeros((4, 1)), np.cumsum(0.5 * (P[:, 1:] + P[:, :-1]) * np.diff(fine), axis=1)], axis=1)
    return {key: np.interp(t_out, fine, cum[i]) for i, key in enumerate(ENERGY_KEYS)}


def simulate(config: Config, atomic: Optional[AtomicData] = None, provider=None,
             t_eval: Optional[np.ndarray] = None) -> RunResult:
    """Integrate the model for one configuration.

    Parameters
    ----------
    config : Config (its ``scan`` section is ignored)
    atomic : pre-loaded AtomicData (optional, avoids reloading ADAS)
    provider : custom atomic data provider (optional)
    t_eval : output times [s]; default: ``config.output_times()``
    """
    tic = _time.time()
    model = TQModel(config, atomic=atomic, provider=provider)
    t_eval = config.output_times() if t_eval is None else np.asarray(t_eval, dtype=float)
    t_span = (t_eval[0], t_eval[-1])
    s = config.solver
    track = config.output.energy_integrals
    # dense_output does not change the integration steps, only stores the interpolants.
    kwargs = dict(method=s.method, events=_event_factory(model), t_eval=t_eval, rtol=s.rtol, atol=s.atol)
    if s.jacobian == "analytic":
        kwargs["jac"] = model.jacobian
    try:
        sol = solve_ivp(model.rhs, t_span, model.y0, dense_output=track, **kwargs)
    except ValueError as exc:
        # A collapsing step size can produce two equal step times, which the dense
        # output rejects. Integrate again without it (no energy integrals).
        if not (track and "ts" in str(exc)):
            raise
        track = False
        sol = solve_ivp(model.rhs, t_span, model.y0, dense_output=False, **kwargs)

    t = np.asarray(sol.t)
    y = np.asarray(sol.y).reshape(model.layout.size, -1)
    if sol.status == -1:
        status, message = STATUS_FAILED, str(sol.message)
    elif sol.status == 1:
        status = STATUS_STOPPED
        message = f"stopped at t = {sol.t_events[0][0]:.4g} s: a fluid quantity became negative"
    else:
        status, message = STATUS_COMPLETED, ""

    # Outputs are always temperatures and densities, whatever solver.state_variables holds.
    fluid = {name: y[k] * model.fluid_scales[k] for k, name in enumerate(FLUID_NAMES)}
    if model.energy_state:
        for k, n in THERMAL_SLOTS:
            density = fluid[FLUID_NAMES[n]]
            fluid[FLUID_NAMES[k]] = fluid[FLUID_NAMES[k]] / np.maximum(density, model.floor)
    nij = np.zeros([len(t), model.layout.z_max + 1, len(model.Z)])
    for j, (Z, off) in enumerate(zip(model.Z, model.layout.offsets)):
        for i in range(0, Z + 1):
            nij[:, i, j] = y[off + i] * NIJ_SCALE
    energies = None
    if track and len(t) > 1 and sol.sol is not None:
        energies = _energy_integrals(model, sol, t, config.output.energy_refinement)
    return RunResult(t=t, fluid=fluid, nij=nij, elements=list(config.element_names), energies=energies,
                     atomic_numbers=list(model.Z), config=config,
                     status=status, message=message, elapsed=_time.time() - tic,
                     r_p_samples=model.r_p_samples)
