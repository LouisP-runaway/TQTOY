"""Derived quantities: power terms, characteristic times, energy balance, scan maps.

All power terms are recomputed with :meth:`tqtoy.model.TQModel.evaluate`, i.e.
with exactly the equations integrated by the solver.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.integrate import cumulative_trapezoid

from .atomic import AtomicData, get_atomic_data
from .constants import E_CHARGE
from .model import TQModel
from .physics.resistivity import critical_field
from .solver import RunResult

# Default threshold on the average electron temperature defining the end of the thermal quench.
T_TQ_THRESHOLD = 100.0  # [eV]


def compute_terms(run: RunResult, atomic: Optional[AtomicData] = None, provider=None) -> Dict[str, np.ndarray]:
    """All intermediate terms of the model along the run (see TQModel.evaluate)."""
    if atomic is None:
        atomic = get_atomic_data(run.config.element_names, provider)
    model = TQModel(run.config, atomic=atomic)
    return model.terms_along(run.t, run.fluid, run.nij)


def _first_time(t, mask) -> float:
    idx = np.flatnonzero(mask)
    return float(t[idx[0]]) if idx.size else np.nan


def thermal_quench_time(run: RunResult, threshold: float = T_TQ_THRESHOLD) -> float:
    """First time at which the density-weighted electron temperature falls below ``threshold`` [s]."""
    return _first_time(run.t, run.Te_avg < threshold)


def radiative_collapse_time(run: RunResult, terms: Dict[str, np.ndarray]) -> float:
    """Time of maximum radiated power of the cold (or single) population [s]."""
    p = terms["P_rad_cold"]
    return float(run.t[np.nanargmax(p)]) if np.any(np.isfinite(p)) else np.nan


def ohmic_radiative_balance_time(run: RunResult, terms: Dict[str, np.ndarray], tolerance: float = 0.1) -> float:
    """First time (after the first output) where |P_ohm / P_rad - 1| <= tolerance, cold population [s]."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = terms["P_ohm_cold"] / terms["P_rad_cold"]
    mask = np.abs(ratio - 1) <= tolerance
    mask[0] = False
    return _first_time(run.t, mask)


def runaway_onset_time(run: RunResult, terms: Dict[str, np.ndarray], population: str = "cold") -> float:
    """First time at which the parallel field exceeds the Connor-Hastie critical field [s]."""
    E = terms[f"E_{population}"]
    Ec = critical_field(run.fluid[f"ne_{population}"], run.fluid[f"Te_{population}"])
    return _first_time(run.t, E > Ec)


def thermal_energy(run: RunResult) -> np.ndarray:
    """Thermal energy density 3/2 e (ne Te + ni Ti) of the active populations [J m^-3]."""
    W = 1.5 * E_CHARGE * (run.ne_cold * run.Te_cold + run.ni_cold * run.Ti_cold)
    if run.two_populations:
        W = W + 1.5 * E_CHARGE * (run.ne_hot * run.Te_hot + run.ni_hot * run.Ti_hot)
    return W


def energy_balance(run: RunResult, terms: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
    """Time-integrated energy terms [J m^-3].

    Returns W_th, the cumulated ohmic, radiated, stochastic and linear-loss energies
    and ``residual`` = (W_th(0) - W_th) - (E_rad + E_stoch + E_lin - E_ohm).

    The model conserves energy, so the residual measures the integration error and
    the effect of the floors. By default the energies are integrated during the
    simulation on the internal solver steps (``run.energies``). Otherwise they are
    integrated from the output samples (``terms``), which can be inaccurate on a
    coarse output grid: the residual then indicates the sampling error.
    """
    if run.energies is not None:
        E_ohm, E_rad, E_stoch, E_lin = (run.energies[k] for k in ("E_ohm", "E_rad", "E_stoch", "E_lin"))
    else:
        if terms is None:
            terms = compute_terms(run)
        t = run.t

        def cum(p):
            return cumulative_trapezoid(np.nan_to_num(p), t, initial=0.0)

        E_ohm = cum(terms["P_ohm_hot"] + terms["P_ohm_cold"])
        E_rad = cum(terms["P_rad_hot"] + terms["P_rad_cold"])
        E_stoch = cum(terms["P_stoch_hot"] + terms["P_stoch_cold"])
        E_lin = cum(terms["P_loss_lin_e"] + terms["P_loss_lin_i"])
    W = thermal_energy(run)
    residual = (W[0] - W) - (E_rad + E_stoch + E_lin - E_ohm)
    return dict(W_th=W, E_ohm=E_ohm, E_rad=E_rad, E_stoch=E_stoch, E_lin=E_lin, residual=residual)


def summary(run: RunResult, terms: Optional[Dict[str, np.ndarray]] = None,
            tq_threshold: float = T_TQ_THRESHOLD) -> Dict[str, float]:
    """Scalar figures of merit of one run. ``tq_threshold`` [eV] defines t_TQ."""
    if len(run.t) < 2:
        raise ValueError(f"The run has {len(run.t)} valid output time(s) (status {run.status}: "
                         f"{run.message or 'no message'}). Nothing to summarise.")
    if terms is None:
        terms = compute_terms(run)
    eb = energy_balance(run, terms)
    W0 = eb["W_th"][0]
    lost = eb["E_rad"][-1] + eb["E_stoch"][-1] + eb["E_lin"][-1]
    return {
        "status": run.status,
        "t_last": float(run.t[-1]),
        "Te_avg_final": float(run.Te_avg[-1]),
        "t_TQ": thermal_quench_time(run, tq_threshold),
        "t_radiative_collapse": radiative_collapse_time(run, terms),
        "t_ohmic_radiative_balance": ohmic_radiative_balance_time(run, terms),
        "t_runaway_onset": runaway_onset_time(run, terms),
        "W_th_initial": float(W0),
        "W_th_final": float(eb["W_th"][-1]),
        "E_ohmic": float(eb["E_ohm"][-1]),
        "E_radiated": float(eb["E_rad"][-1]),
        "E_stochastic": float(eb["E_stoch"][-1]),
        "radiated_fraction": float(eb["E_rad"][-1] / lost) if lost > 0 else np.nan,
        "energy_residual_rel": float(np.max(np.abs(eb["residual"])) / W0) if W0 > 0 else np.nan,
        **(_fhte_summary(run.fhte) if run.fhte is not None else {}),
    }


def _fhte_summary(res):
    from .fhte import summary_fhte
    return summary_fhte(res)


SCAN_QUANTITIES = {
    "n_RE_hot_tail": "hot-tail runaway fraction n_RE/n0 at the last FHTE time",
    "n_RE_hot_tail_density": "hot-tail runaway density at the last FHTE time [m^-3]",
    "t_TQ": "thermal quench time: first time with <Te> < 100 eV [s]",
    "Te_avg_final": "final average electron temperature [eV]",
    "t_radiative_collapse": "time of maximum radiated power [s] (slower: recomputes the power terms)",
    "radiated_fraction": "radiated / (radiated + stochastic + linear) energy",
    "radiated_fraction_net": "(radiated - ohmic) / (radiated + stochastic + linear - ohmic) energy",
}


def scan_map(results, quantity: str = "t_TQ", progress: bool = False, **kwargs) -> np.ndarray:
    """Evaluate a scalar quantity on every scan point. See SCAN_QUANTITIES.

    The hot-tail quantities use the stored FHTE results (tqtoy fhte / fhte.enabled).
    Without them, the estimator is run on each point with its fhte settings.
    """
    if quantity.startswith("n_RE_hot_tail"):
        return _hot_tail_map(results, quantity, progress)
    out = np.full(results.shape, np.nan)
    for idx, run in results:
        if len(run.t) < 2:
            continue
        if quantity == "t_TQ":
            out[idx] = thermal_quench_time(run, **kwargs)
        elif quantity == "Te_avg_final":
            out[idx] = run.Te_avg[-1]
        elif quantity == "t_radiative_collapse":
            out[idx] = radiative_collapse_time(run, compute_terms(run))
        elif quantity == "radiated_fraction":
            eb = energy_balance(run)
            lost = eb["E_rad"][-1] + eb["E_stoch"][-1] + eb["E_lin"][-1]
            out[idx] = eb["E_rad"][-1] / lost if lost > 0 else np.nan
        elif quantity == "radiated_fraction_net":
            eb = energy_balance(run)
            net = eb["E_rad"][-1] + eb["E_stoch"][-1] + eb["E_lin"][-1] - eb["E_ohm"][-1]
            out[idx] = (eb["E_rad"][-1] - eb["E_ohm"][-1]) / net if net > 0 else np.nan
        else:
            raise KeyError(f"Unknown quantity '{quantity}'. Available: {list(SCAN_QUANTITIES)}.")
    return out


def _hot_tail_map(results, quantity, progress=False):
    if results.fhte is not None:
        frac = results.fhte["fraction"]
        n_valid = np.sum(np.isfinite(results.fhte["t_output"]), axis=-1)
        last = np.take_along_axis(frac, np.maximum(n_valid - 1, 0)[..., None], axis=-1)[..., 0]
        last = np.where(n_valid > 0, last, np.nan)
        return last * results.fhte["n0"] if quantity.endswith("density") else last
    import copy
    import sys
    from .scan import compute_fhte
    if progress:
        print("[tqtoy] no stored FHTE results: running the estimator on every point "
              "(store them with 'tqtoy fhte results.h5')", file=sys.stderr)
    tmp = copy.copy(results)
    tmp.messages = results.messages.copy()
    tmp = compute_fhte(tmp, progress=progress)
    return _hot_tail_map(tmp, quantity)
