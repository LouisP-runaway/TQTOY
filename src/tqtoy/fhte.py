"""Fast Hot Tail Estimator (FHTE): runaway electrons produced by the hot-tail mechanism.

Principle
---------
A test electron of normalised momentum p = gamma v / c follows the collisional
slowing-down and the acceleration by the parallel field (no pitch angle)::

    dgamma/dt = -s A(gamma) gamma / p + (e E / me c) p / gamma
    A = n lnL(gamma) e^4 / (4 pi eps0^2 me^2 c^3)

with s = +1 above the bulk thermal energy and s = -1 below it. At each evaluation
time t, the critical momentum p_c(t) = (E / E_c - 1)^(-1/2), with
E_c = me c A(gamma_c) / e (Connor-Hastie with the relativistic Coulomb logarithm,
gamma_c found by fixed-point iterations), separates runaways from thermalised
electrons. The trajectory ending at p_c(t) is integrated backward to the initial
time: its initial momentum p_limit(t) is the smallest initial momentum of a
runaway at t. The runaway fraction is the fraction of the initial Maxwell-Juttner
distribution (temperature T(t0)) above p_limit::

    n_RE(t) / n0 = int_{gamma_limit}^{gamma_max} f_MJ dgamma / int_1^{gamma_max} f_MJ dgamma

The estimator needs T(t), n(t) and either the current density J(t) (the field is
then E = eta_Spitzer J, with Z_eff = ``zeff`` and a factor ``neoclassical_factor``)
or a prescribed field E(t). It can be run on any prescribed evolution
(:func:`hot_tail_estimate`, :func:`run_prescribed`) or on a TQ toy-model run
(:func:`hot_tail_from_run`).

T, n and a prescribed E are interpolated linearly in time inside the momentum
equation.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from functools import lru_cache
from math import pi, sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from scipy.integrate import solve_ivp, trapezoid
from scipy.special import erf

from .config import ConfigError, FHTEConfig, _build, read_yaml

# Constants of the estimator. Do not reformat these expressions: a literal and the
# product that produces it can differ in the last bit (see tqtoy.constants).
EPS_0 = 8.85418782*1E-12
CHARGE = 1.602176565*1E-19
E0_E = 510998.95000                 # electron rest energy [eV]
C = 299792458
HBAR = 1.054571817*1E-34
ME = E0_E*CHARGE/C**2
MU_E = ME/2                         # reduced mass of an electron-electron pair
A_PREFACTOR = 2.991535170013209e-20  # e^4 / (4 pi eps0^2 me^2 c^3) [m^3 s^-1]
P_NO_RUNAWAY = 100.0                # critical momentum used when E <= E_c


# --------------------------------------------------------------------------- physics

def coulomb_log(gamma, Te, n):
    """Coulomb logarithm of an electron of Lorentz factor gamma on a thermal bulk (Te [eV], n [m^-3]).

    ln(lambda_D / b_min), with b_min the largest of the classical and quantum impact
    parameters, evaluated with the relative velocity u^2 = 3 Te e / me + v^2.
    """
    debye_length = np.sqrt((EPS_0*Te)/(n*CHARGE))
    u = np.sqrt(3*Te*CHARGE/ME + (1-1/(gamma**2))*C**2)
    b_classical = (CHARGE**2) / (4*pi*EPS_0*MU_E*u**2)
    b_quantum = (HBAR)/(2*MU_E*u)
    return np.log(debye_length/np.maximum(b_quantum, b_classical))


def chandrasekhar_coefficient(Te, gamma):
    """Chandrasekhar function erf(x) - x erf'(x), x = v / v_th with v_th = (3 Te e / me)^1/2."""
    beta = np.sqrt(1-1/gamma**2)
    v_th = sqrt(3*Te*CHARGE/ME)
    x = beta*C/v_th
    dx = 1e-10
    erf_prime = (erf(x + dx) - erf(x - dx)) / (2 * dx)
    return erf(x) - x*erf_prime


def drag_coefficient(gamma, Te, n, chandrasekhar=False):
    """Collisional coefficient A = n lnL e^4 / (4 pi eps0^2 me^2 c^3) [s^-1].

    Floors: Te >= 1 eV, n >= 1e15 m^-3.
    """
    if Te < 1:
        Te = 1
    if n < 1e15:
        n = 1e15
    A = n * coulomb_log(gamma, Te, n) * A_PREFACTOR
    if chandrasekhar:
        A = A*chandrasekhar_coefficient(Te, gamma)
    return A


def eta_spitzer_fhte(Te, Z_eff, ne, factor=2.0):
    """Spitzer resistivity of the FHTE [Ohm m], multiplied by ``factor`` (neoclassical correction)."""
    if Te < 1:
        Te = 1
    if ne < 1e15:
        ne = 1e15
    if Te < 10:
        coulomb_log_e = max(23.0000 - np.log((ne*1e-6)**0.5*Te**(-1.5)), 1)
    else:
        coulomb_log_e = max(24.1513 - np.log((ne*1e-6)**0.5*Te**(-1.0)), 1)
    coef_zeff = Z_eff*(1.+1.198*Z_eff+0.222*Z_eff**2)/(1.+2.966*Z_eff+0.753*Z_eff**2) / \
        ((1.+1.198+0.222)/(1.+2.966+0.753))
    return 1.65e-9 * coulomb_log_e * (Te*0.001)**(-1.5) * coef_zeff * factor


def critical_field(ne, Te, gamma_c, chandrasekhar=False):
    """Critical field E_c = me c A(gamma_c) / e [V m^-1]."""
    return ME*C/CHARGE * drag_coefficient(gamma_c, Te, ne, chandrasekhar)


def critical_momentum(E, ne, Te, gamma_c, chandrasekhar=False):
    """Critical momentum p_c = (E/E_c - 1)^-1/2 (normalised to me c), P_NO_RUNAWAY if E <= E_c."""
    E_c = critical_field(ne, Te, gamma_c, chandrasekhar)
    if E > E_c:
        return 1/np.sqrt(E/E_c - 1)
    return P_NO_RUNAWAY


@lru_cache(maxsize=8)
def _maxwell_juttner_tail(T0, gamma_max, n_points):
    """Grid of Lorentz factors and the normalised tail integral above each grid point.

    ``tail[k]`` is the trapezoidal integral of the Maxwell-Juttner distribution from
    ``gamma[k]`` to ``gamma_max``, divided by the integral over the whole grid. The
    integrand is built once per temperature: the estimator evaluates the tail at every
    output time, and the grid holds 2e5 points by default.
    """
    gamma = np.linspace(1, gamma_max, n_points)
    theta = T0 * CHARGE / (ME * C**2)
    beta = np.sqrt(1.0 - 1.0/gamma**2)
    with np.errstate(divide="ignore"):
        ln_f = 2*np.log(gamma) + np.log(beta) - gamma/theta
    f_mj = np.exp(ln_f - np.max(ln_f))
    terms = 0.5 * (f_mj[1:] + f_mj[:-1]) * np.diff(gamma)        # trapezoid terms
    tail = np.concatenate([np.cumsum(terms[::-1])[::-1], [0.0]])  # tail[k] = sum(terms[k:])
    return gamma, tail / trapezoid(f_mj, gamma)


def runaway_fraction(p_limit, T0, gamma_max=4.0, n_points=200000):
    """Fraction of a Maxwell-Juttner distribution of temperature T0 [eV] with p > p_limit."""
    gamma, tail = _maxwell_juttner_tail(float(T0), float(gamma_max), int(n_points))
    k = int(np.searchsorted(gamma, np.sqrt(1+p_limit**2), side="left"))
    return min(float(tail[k]), 1.0) if k < len(gamma) else 0.0


# --------------------------------------------------------------------------- estimator

@dataclass
class FHTEResult:
    """Output of the estimator. Arrays of length n_output unless stated otherwise."""

    t_output: np.ndarray        # evaluation times [s]
    fraction: np.ndarray        # runaway fraction n_RE / n0 (0 at the first time)
    p_c: np.ndarray             # critical momentum [me c]
    p_limit: np.ndarray         # initial momentum of the trajectory ending at p_c [me c]
    E: np.ndarray               # parallel field [V m^-1]
    E_c: np.ndarray             # critical field [V m^-1]
    T: np.ndarray               # bulk temperature [eV] (after the floor)
    n: np.ndarray               # density [m^-3] (after the floor)
    p_trajectory: np.ndarray    # (n_output, n_trajectory) backward trajectories [me c]
    t_trajectory: np.ndarray    # (n_output, n_trajectory) their times, from t_output[i] back to t0 [s]
    T0: float                   # initial temperature of the Maxwell-Juttner distribution [eV]
    n0: float                   # initial density [m^-3]
    status: int = 0             # 0: all trajectories integrated, -1: at least one failure
    message: str = ""

    @property
    def final_fraction(self) -> float:
        return float(self.fraction[-1])

    @property
    def density(self) -> np.ndarray:
        """Runaway density n_RE = fraction x n0 [m^-3]."""
        return self.fraction * self.n0

    @property
    def final_density(self) -> float:
        return float(self.density[-1])


def _output_times(t0, t1, n, spacing):
    if spacing == "log" and t0 > 0:
        return 10**np.linspace(np.log10(t0), np.log10(t1), n)
    return np.linspace(t0, t1, n)


def _linear_interpolator(x, y):
    """Linear interpolation with linear extrapolation, on scalars.

    Inside the range of ``x``, ``numpy.interp`` gives the same values as
    ``scipy.interpolate.interp1d(kind='linear')`` (checked bit for bit on the reference traces)
    and is an order of magnitude faster on scalars, which matters because the right-hand side of
    the momentum equation calls it at every step. Outside that range, the two extrapolations are
    mathematically equal but anchored on different nodes, so they can differ in the last bits.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    def evaluate(value):
        if not isinstance(value, (float, int, np.floating)):        # array
            v = np.asarray(value, dtype=float)
            out = np.interp(v, x, y)
            below, above = v < x[0], v > x[-1]
            if np.any(below):
                out = np.where(below, y[0] + (y[1] - y[0]) / (x[1] - x[0]) * (v - x[0]), out)
            if np.any(above):
                out = np.where(above, y[-1] + (y[-1] - y[-2]) / (x[-1] - x[-2]) * (v - x[-1]), out)
            return out
        if value < x[0]:
            return y[0] + (y[1] - y[0]) / (x[1] - x[0]) * (value - x[0])
        if value > x[-1]:
            return y[-1] + (y[-1] - y[-2]) / (x[-1] - x[-2]) * (value - x[-1])
        return np.interp(value, x, y)

    return evaluate


def _mask_intervals(t, active):
    """Time intervals [first, last] of the blocks of samples where ``active`` is True.

    The bounds are samples of ``t``, not the interpolated crossing instants, so that every
    evaluation time placed in an interval is a time where the condition actually holds.
    An isolated sample gives a degenerate interval (first == last).
    """
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return []
    blocks = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    return [(float(t[b[0]]), float(t[b[-1]])) for b in blocks]


def _share(counts_total, weights):
    """Integer shares of ``counts_total`` over ``weights``: one each, the rest proportional.

    A zero weight (a window reduced to one sample) receives exactly one point: it holds one
    time and more would be duplicates. When there are fewer points than windows, the widest
    windows are served first and the caller reports the windows left out.
    """
    w = np.asarray(weights, dtype=float)
    n = len(w)
    order = np.argsort(-w)
    if counts_total <= n:                        # not enough points: the widest windows first
        base = np.zeros(n, dtype=int)
        base[order[:counts_total]] = 1
        return base
    base = np.ones(n, dtype=int)
    free = w > 0                                 # zero-width windows keep their single point
    spare = counts_total - n
    if spare and np.any(free):
        share = w[free] / w[free].sum()
        exact = spare * share
        extra = np.floor(exact).astype(int)
        for k in np.argsort(-(exact - extra))[:spare - int(extra.sum())]:
            extra[k] += 1
        base[free] += extra
    return base


def _active_output_times(t, intervals, n, spacing):
    """Evaluation times inside ``intervals``, and how many intervals got none.

    The first time of the run is always stored: it is the time the backward trajectories
    are integrated to. The last one is stored as well when it lies outside the intervals,
    so that the result covers the whole run, but only once every interval has a time.
    Without any interval the times are spread over the run as in :func:`_output_times`.
    """
    if not intervals:
        return _output_times(t[0], t[-1], n, spacing), 0
    k = n - 1                                   # the first time of the run is always stored
    if intervals[0][0] <= t[0]:
        k += 1                                  # one generated time will coincide with it
    # the last time of the run is stored too, but only once every window has a time
    ends_inactive = intervals[-1][1] < t[-1] and k - 1 >= len(intervals)
    k = max(k - (1 if ends_inactive else 0), 1)
    log_spacing = spacing == "log" and all(a > 0 for a, _ in intervals)
    weights = [np.log(b / a) if log_spacing else b - a for a, b in intervals]
    counts = _share(k, weights)
    points = []
    for (a, b), c in zip(intervals, counts):
        if c <= 0:
            continue                            # window left out: reported by the caller
        points.append(np.array([b]) if c == 1 or b <= a else
                      _output_times(a, b, c, "log" if log_spacing else "linear"))
    times = np.unique(np.concatenate([[t[0]]] + points + ([[t[-1]]] if ends_inactive else [])))
    return times, int(np.sum(counts == 0))


def hot_tail_estimate(t, Te, ne, J=None, E=None, config: Optional[FHTEConfig] = None, **options) -> FHTEResult:
    """Hot-tail runaway fraction for a prescribed evolution.

    Parameters
    ----------
    t : times [s] (increasing)
    Te : bulk electron temperature [eV], array or scalar
    ne : electron density [m^-3], array or scalar
    J : current density [A m^-2], array or scalar. Used when E is None.
    E : parallel electric field [V m^-1], array or scalar (overrides J)
    config : FHTEConfig (default values if None). Keyword ``options`` override its fields,
        e.g. ``hot_tail_estimate(t, Te, ne, J=1e6, n_output=20)``.
    """
    cfg = config if config is not None else FHTEConfig()
    if options:
        cfg = dataclasses.replace(cfg, **options)
    cfg.validate()
    t = np.asarray(t, dtype=float)
    if E is None and J is None:
        raise ValueError("Give the current density J or the electric field E.")

    def as_trace(x, name):
        a = np.broadcast_to(np.asarray(x, dtype=float), t.shape).copy()
        if a.shape != t.shape:
            raise ValueError(f"{name} must be a scalar or have the length of t.")
        return a

    T_t = np.nan_to_num(as_trace(Te, "Te"), nan=cfg.T_floor)
    T_t[T_t < cfg.T_floor] = cfg.T_floor
    n_t = np.nan_to_num(as_trace(ne, "ne"), nan=cfg.n_floor)
    n_t[n_t < cfg.n_floor] = cfg.n_floor
    T_interp = _linear_interpolator(t, T_t)
    n_interp = _linear_interpolator(t, n_t)
    if E is None:
        J_t = np.nan_to_num(as_trace(J, "J"), nan=0.0)
        J_interp = _linear_interpolator(t, J_t)

        def field_at(T_b, n_b, time):
            return eta_spitzer_fhte(T_b, cfg.zeff, n_b, cfg.neoclassical_factor) * J_interp(time)
    else:
        E_interp = _linear_interpolator(t, as_trace(E, "E"))

        def field_at(T_b, n_b, time):
            return E_interp(time)

    chand = cfg.chandrasekhar

    def momentum_rhs(time, y):
        """dgamma/dt of the test electron (see the module docstring).

        The operations are kept in this order and on 1-element arrays, which fixes their
        round-off. A non-finite force is replaced by zero: gamma <= 1 occurs during the
        Newton iterations of Radau.
        """
        y = np.atleast_1d(y)
        T_b = T_interp(time)
        n_b = n_interp(time)
        K = (y - 1) * ME * C**2 / CHARGE
        A = drag_coefficient(y, T_b, n_b, chand)
        A = np.where(K < T_b, -A, A)            # below the thermal energy, collisions accelerate
        E_f = field_at(T_b, n_b, time)
        beta_gamma = np.sqrt(y**2 - 1)
        force = -A * y / beta_gamma + CHARGE*E_f/(ME*C) * beta_gamma / y
        return np.where(np.isfinite(force), force, 0.0)

    def critical_at(time):
        """p_c, E_c, E and the bulk values at one time (fixed point on gamma_c)."""
        T_b = T_interp(time)
        n_b = n_interp(time)
        E_f = field_at(T_b, n_b, time)
        gamma_c, p = 2.0, P_NO_RUNAWAY
        for _ in range(cfg.critical_iterations):
            p = critical_momentum(E_f, n_b, T_b, gamma_c, chand)
            gamma_c = np.sqrt(1 + p**2)
        return p, critical_field(n_b, T_b, gamma_c, chand), E_f

    n_out, n_traj = cfg.n_output, cfg.n_trajectory
    active = cfg.output_window == "active"
    if active:
        # Runaways are created only where E > E_c. The evaluation times are placed there,
        # and no backward integration is done outside: the runaway population is simply kept.
        runaway = np.array([critical_at(tk)[0] < P_NO_RUNAWAY for tk in t])
        windows = _mask_intervals(t, runaway)
        t_out, missed = _active_output_times(t, windows, n_out, cfg.output_spacing)
        n_out = len(t_out)
    else:
        t_out = _output_times(t[0], t[-1], n_out, cfg.output_spacing)
        missed = 0
    p_c = np.ones(n_out) * P_NO_RUNAWAY
    p_limit = np.full(n_out, np.nan)
    fraction = np.zeros(n_out)
    E_out, Ec_out = np.full(n_out, np.nan), np.full(n_out, np.nan)
    p_traj = np.zeros((n_out, n_traj))
    t_traj = np.zeros((n_out, n_traj))
    status, message = 0, ""
    if missed:
        message = (f"{missed} of the {len(windows)} time windows where E > E_c received no "
                   f"evaluation time: increase fhte.n_output")
    last_valid = 0.0                            # last runaway fraction that was computed
    for i in range(1, n_out):
        p_c[i], Ec_out[i], E_out[i] = critical_at(t_out[i])
        if cfg.trajectory_spacing == "log" and t_out[0] > 0:
            t_traj[i, :] = 10**np.linspace(np.log10(t_out[i]), np.log10(t_out[0]), n_traj)
            t_traj[i, 0], t_traj[i, -1] = t_out[i], t_out[0]
        else:
            t_traj[i, :] = np.linspace(t_out[i], t_out[0], n_traj)
        if active and not p_c[i] < P_NO_RUNAWAY:
            # E <= E_c: no new runaway. The population already produced is not lost. The last
            # value that was actually computed is carried, so a failed point is not propagated.
            fraction[i] = last_valid
            p_traj[i, :] = np.nan
            continue
        with np.errstate(invalid="ignore", divide="ignore"):    # gamma < 1 in the Newton iterations
            sol = solve_ivp(momentum_rhs, (t_traj[i, 0], t_traj[i, -1]), [np.sqrt(1 + p_c[i]**2)],
                            method="Radau", rtol=cfg.rtol, atol=cfg.atol, t_eval=t_traj[i, :])
        if not sol.success or sol.y.shape[1] != n_traj:
            status, message = -1, f"backward integration failed at t = {t_out[i]:.3g} s: {sol.message}"
            p_traj[i, :] = np.nan
            fraction[i] = np.nan
            continue
        # A trajectory that reaches gamma = 1 before t0 means that electrons at rest
        # become runaways, so p_limit = 0 and the whole distribution is counted.
        p_traj[i, :] = np.sqrt(np.maximum(sol.y[0]**2 - 1, 0.0))
        p_limit[i] = p_traj[i, -1]
        if p_limit[i] == 0 and not message:
            message = f"the backward trajectory from t = {t_out[i]:.3g} s reaches p = 0: all electrons run away"
        fraction[i] = runaway_fraction(p_limit[i], T_t[0], cfg.gamma_max, cfg.gamma_points)
        last_valid = fraction[i]
    if cfg.accumulate:
        best = 0.0
        for i in range(n_out):                  # runaways already created are not lost
            if np.isfinite(fraction[i]):
                best = max(best, fraction[i])
                fraction[i] = best
    return FHTEResult(t_output=t_out, fraction=fraction, p_c=p_c, p_limit=p_limit, E=E_out, E_c=Ec_out,
                      T=T_interp(t_out), n=n_interp(t_out), p_trajectory=p_traj, t_trajectory=t_traj,
                      T0=float(T_t[0]), n0=float(n_t[0]), status=status, message=message)


def hot_tail_from_run(run, config: Optional[FHTEConfig] = None, atomic=None, provider=None) -> FHTEResult:
    """Hot-tail estimate on a TQ toy-model run (tqtoy.RunResult).

    Uses the temperature selected by ``config.temperature``, the free-electron density
    of the active populations (plus ``bound_electron_weight`` times the bound electrons),
    and either the current density of the run (field = spitzer) or its parallel field
    (field = model).
    """
    from .physics.resistivity import linear_current_ramp

    cfg = config if config is not None else run.config.fhte
    cfg.validate(two_populations=run.two_populations)
    if len(run.t) < 2:
        raise ValueError(f"The run has {len(run.t)} valid output time(s): no hot-tail estimate.")
    T = {"average": run.Te_avg, "cold": run.Te_cold, "hot": run.Te_hot}[cfg.temperature]
    n = run.ne_total
    if cfg.bound_electron_weight > 0:
        bound = np.zeros_like(n)
        for j, Z in enumerate(run.atomic_numbers):
            for i in range(Z + 1):
                bound = bound + (Z - i) * run.nij[:, i, j]
        n = n + cfg.bound_electron_weight * bound
    if cfg.field == "model":
        from .diagnostics import compute_terms
        E = compute_terms(run, atomic=atomic, provider=provider)["E_cold"]
        return hot_tail_estimate(run.t, T, n, E=E, config=cfg)
    p = run.config.plasma
    if p.J_final is None:
        J = np.full(len(run.t), p.J)
    else:
        J = np.array([linear_current_ramp(p.J, tk, p.J_final, p.J_ramp_time) for tk in run.t])
    return hot_tail_estimate(run.t, T, n, J=J, config=cfg)


# --------------------------------------------------------------------------- prescribed evolutions

TRACE_HELP = ("number (constant), list (one value per time), {exponential: {initial, final, tau}}, "
              "{linear: {initial, final}} or {column: name} (column of prescribed.file)")


@dataclass
class PrescribedConfig:
    """Prescribed evolution of the bulk plasma for a standalone FHTE run."""

    t_start: float = field(default=1.0e-8, metadata={"help": "First time", "unit": "s"})
    t_end: float = field(default=1.2e-3, metadata={"help": "Last time", "unit": "s"})
    n_points: int = field(default=200, metadata={"help": "Number of times", "unit": ""})
    spacing: str = field(default="linear", metadata={"help": "linear or log", "unit": ""})
    file: Optional[str] = field(default=None, metadata={
        "help": "Optional CSV (header t,Te,...) or NPZ file. Its column t replaces the time grid", "unit": ""})
    Te: Any = field(default_factory=lambda: {"exponential": {"initial": 3.1e3, "final": 31.0, "tau": 0.3e-3}},
                    metadata={"help": "Bulk electron temperature [eV]: " + TRACE_HELP, "unit": "eV"})
    ne: Any = field(default=2.8e19, metadata={"help": "Electron density [m^-3]: " + TRACE_HELP, "unit": "m^-3"})
    J: Any = field(default=1.5e6, metadata={"help": "Current density [A m^-2]: " + TRACE_HELP, "unit": "A m^-2"})
    E: Any = field(default=None, metadata={
        "help": "Parallel field [V m^-1] (null: Spitzer field from J): " + TRACE_HELP, "unit": "V m^-1"})


@dataclass
class FHTERunConfig:
    """Input file of ``tqtoy fhte``: a TQ results file (``results``) or a prescribed evolution."""

    results: Optional[str] = field(default=None, metadata={
        "help": "Results file (.h5) of tqtoy run: the estimate runs on every scan point and is stored in "
                "the file. Null: run on the prescribed evolution below", "unit": ""})
    prescribed: PrescribedConfig = field(default_factory=PrescribedConfig, metadata={
        "help": "Prescribed evolution of the bulk plasma (ignored when results is set)", "unit": ""})
    fhte: FHTEConfig = field(default_factory=lambda: FHTEConfig(n_output=20, n_trajectory=50), metadata={
        "help": "Settings of the estimator. The defaults below apply to a prescribed evolution. With a "
                "results file, only the entries actually written in the input file override the fhte "
                "section stored in it, so a scanned FHTE parameter keeps its value", "unit": ""})
    jobs: int = field(default=1, metadata={"help": "Number of parallel processes (results file only)", "unit": ""})
    output: Optional[str] = field(default=None, metadata={
        "help": "Output file (.h5). Null: results/fhte_results.h5 for a prescribed evolution, or update the "
                "results file in place", "unit": ""})

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "FHTERunConfig":
        cfg = _build(cls, data or {}, "")
        cfg.fhte.validate()
        if cfg.prescribed.spacing not in ("linear", "log"):
            raise ConfigError("prescribed.spacing must be linear or log.")
        if cfg.jobs < 1:
            raise ConfigError("jobs must be at least 1.")
        return cfg

    @classmethod
    def from_yaml(cls, path) -> "FHTERunConfig":
        data = read_yaml(path)
        try:
            return cls.from_dict(data)
        except ConfigError as exc:
            raise ConfigError(f"{path}: {exc}") from None

    def to_yaml(self) -> str:
        return yaml.safe_dump(dataclasses.asdict(self), sort_keys=False)


def _read_table(path) -> Dict[str, np.ndarray]:
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path)
        return {k: np.asarray(data[k], dtype=float) for k in data.files}
    arr = np.genfromtxt(path, delimiter=",", names=True)
    return {name: np.asarray(arr[name], dtype=float) for name in arr.dtype.names}


def _trace(spec, t, table, name):
    if spec is None:
        return None
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return np.full(t.shape, float(spec))
    if isinstance(spec, str):
        try:
            return np.full(t.shape, float(spec))
        except ValueError:
            raise ConfigError(f"prescribed.{name}: invalid value {spec!r}.") from None
    if isinstance(spec, (list, tuple)):
        a = np.asarray(spec, dtype=float)
        if a.shape != t.shape:
            raise ConfigError(f"prescribed.{name}: {a.size} values for {t.size} times.")
        return a
    if isinstance(spec, dict) and len(spec) == 1:
        (kind, args), = spec.items()
        if kind == "column":
            if table is None or args not in table:
                raise ConfigError(f"prescribed.{name}: column '{args}' not found (prescribed.file).")
            return table[args]
        if kind == "exponential":
            a = {k: float(v) for k, v in args.items()}
            return a["final"] + (a["initial"] - a["final"]) * np.exp(-(t - t[0]) / a["tau"])
        if kind == "linear":
            a = {k: float(v) for k, v in args.items()}
            return a["initial"] + (a["final"] - a["initial"]) * (t - t[0]) / (t[-1] - t[0])
    raise ConfigError(f"prescribed.{name}: expected {TRACE_HELP}.")


def build_traces(p: PrescribedConfig) -> Dict[str, Optional[np.ndarray]]:
    """Time grid and traces (t, Te, ne, J, E) of a prescribed evolution."""
    table = _read_table(p.file) if p.file else None
    if table is not None and "t" in table:
        t = table["t"]
    elif p.spacing == "log":
        t = 10**np.linspace(np.log10(p.t_start), np.log10(p.t_end), p.n_points)
    else:
        t = np.linspace(p.t_start, p.t_end, p.n_points)
    traces = {"t": t}
    for name in ("Te", "ne", "J", "E"):
        traces[name] = _trace(getattr(p, name), t, table, name)
    if traces["Te"] is None or traces["ne"] is None:
        raise ConfigError("prescribed: Te and ne are required.")
    if traces["E"] is None and traces["J"] is None:
        raise ConfigError("prescribed: give J or E.")
    return traces


def run_prescribed(cfg: FHTERunConfig):
    """Run the estimator on a prescribed evolution. Returns (traces, FHTEResult)."""
    tr = build_traces(cfg.prescribed)
    res = hot_tail_estimate(tr["t"], tr["Te"], tr["ne"], J=tr["J"], E=tr["E"], config=cfg.fhte)
    return tr, res


# --------------------------------------------------------------------------- files

FHTE_FORMAT = "tqtoy-fhte-1"
RESULT_ARRAYS = ("t_output", "fraction", "p_c", "p_limit", "E", "E_c", "T", "n", "p_trajectory", "t_trajectory")


def save_prescribed(path, cfg: FHTERunConfig, traces, res: FHTEResult) -> Path:
    """Write a standalone FHTE result (inputs, configuration and result) to HDF5."""
    import h5py
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["format"] = FHTE_FORMAT
        f.attrs["config"] = cfg.to_yaml()
        g = f.create_group("inputs")
        for k, v in traces.items():
            if v is not None:
                g.create_dataset(k, data=v)
        r = f.create_group("result")
        for k in RESULT_ARRAYS:
            r.create_dataset(k, data=getattr(res, k))
        r.attrs["T0"], r.attrs["n0"], r.attrs["status"], r.attrs["message"] = res.T0, res.n0, res.status, res.message
    return path


def load_prescribed(path):
    """Read a file written by :func:`save_prescribed`. Returns (config, traces, FHTEResult)."""
    import h5py
    with h5py.File(path, "r") as f:
        if f.attrs.get("format") != FHTE_FORMAT:
            raise ValueError(f"{path} is not a standalone FHTE file.")
        cfg = FHTERunConfig.from_dict(yaml.safe_load(f.attrs["config"]))
        traces = {k: f["inputs"][k][...] for k in f["inputs"]}
        for k in ("J", "E"):
            traces.setdefault(k, None)
        r = f["result"]
        res = FHTEResult(**{k: r[k][...] for k in RESULT_ARRAYS}, T0=float(r.attrs["T0"]), n0=float(r.attrs["n0"]),
                         status=int(r.attrs["status"]), message=str(r.attrs["message"]))
    return cfg, traces, res


def file_format(path) -> str:
    import h5py
    with h5py.File(path, "r") as f:
        return str(f.attrs.get("format", ""))


def summary_fhte(res: FHTEResult) -> Dict[str, float]:
    """Scalar results of the estimator.

    The momentum quantities are given at the last time where a backward trajectory was
    integrated (the last time with E > E_c), not at the last evaluation time, which can lie
    after the runaway window.
    """
    computed = np.flatnonzero(np.isfinite(res.p_limit))
    k = int(computed[-1]) if computed.size else -1
    return {
        "n_RE_hot_tail_fraction": res.final_fraction,
        "n_RE_hot_tail_density": res.final_density,
        "t_last_runaway": float(res.t_output[k]) if computed.size else np.nan,
        "p_c_last_runaway": float(res.p_c[k]) if computed.size else np.nan,
        "p_limit_last_runaway": float(res.p_limit[k]) if computed.size else np.nan,
        "E_over_Ec_last_runaway": (float(res.E[k] / res.E_c[k])
                                   if computed.size and np.isfinite(res.E_c[k]) else np.nan),
        "T0_fhte": res.T0,
        "n0_fhte": res.n0,
    }


def result_to_arrays(res: FHTEResult) -> Dict[str, Any]:
    out = {k: getattr(res, k) for k in RESULT_ARRAYS}
    out.update(T0=res.T0, n0=res.n0, status=res.status, message=res.message)
    return out


def result_from_arrays(d: Dict[str, Any]) -> FHTEResult:
    n = int(np.sum(np.isfinite(np.asarray(d["t_output"]))))
    kw = {k: np.asarray(d[k])[:n] for k in RESULT_ARRAYS}
    return FHTEResult(**kw, T0=float(d["T0"]), n0=float(d["n0"]), status=int(d.get("status", 0)),
                      message=str(d.get("message", "")))


__all__: List[str] = ["FHTEResult", "FHTERunConfig", "PrescribedConfig", "hot_tail_estimate", "hot_tail_from_run",
                      "run_prescribed", "runaway_fraction", "critical_momentum", "summary_fhte"]
