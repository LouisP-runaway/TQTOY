"""Parameter scans (grid over any set of configuration parameters)."""

from __future__ import annotations

import sys
import time as _time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .atomic import get_atomic_data
from .config import Config, ConfigError
from .model import FLUID_NAMES, StateLayout
from .solver import ENERGY_KEYS, STATUS_FAILED, STATUS_NOT_RUN, RunResult, simulate


@dataclass
class ScanResults:
    """Results of a scan. Arrays have the scan shape as leading dimensions.

    time        : (*shape, n_t)          output times of each point (they differ when a
                                         time parameter is scanned), NaN padded
    fluid[name] : (*shape, n_t)          time traces (NaN after the last valid time)
    nij         : (*shape, n_t, Z_max+1, n_elements)
    n_valid     : (*shape)               number of valid output times
    status      : (*shape)               see tqtoy.solver.STATUS_*
    """

    config: Config
    axes: List[Tuple[str, np.ndarray]]
    time: np.ndarray
    fluid: Dict[str, np.ndarray]
    nij: np.ndarray
    n_valid: np.ndarray
    status: np.ndarray
    elapsed: np.ndarray
    messages: np.ndarray
    elements: List[str]
    atomic_numbers: List[int]
    r_p_samples: Optional[np.ndarray] = None
    energies: Optional[Dict[str, np.ndarray]] = None      # key -> (*shape, n_t)
    fhte: Optional[Dict[str, np.ndarray]] = None          # hot-tail estimates, see FHTE_* below
    fhte_overrides: Optional[Dict] = None                 # FHTE settings changed by a post-hoc computation
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(len(v) for _, v in self.axes)

    @property
    def axis_names(self) -> List[str]:
        return [name for name, _ in self.axes]

    @property
    def t(self) -> np.ndarray:
        """Output times shared by all points. Raises if a time parameter was scanned."""
        flat = self.time.reshape(-1, self.time.shape[-1])
        first = flat[0]
        if not all(np.array_equal(row, first, equal_nan=True) for row in flat[1:]):
            raise ValueError("The output times differ between scan points (a time parameter was scanned). "
                             "Use results.time[idx] or results.run(idx).t.")
        return first[np.isfinite(first)]

    # ------------------------------------------------------------------ selection
    def _axis_position(self, key: str) -> int:
        matches = [k for k, name in enumerate(self.axis_names) if name == key or name.endswith("." + key)]
        if len(matches) != 1:
            raise KeyError(f"'{key}' does not identify one scan axis among {self.axis_names}.")
        return matches[0]

    def index(self, coords: Optional[Dict[str, float]] = None, **kwargs) -> Tuple[int, ...]:
        """Grid index of the point closest to the requested values.

        Axes may be named by their full dotted path or by a unique suffix, e.g.
        ``index({'n_D': 1e22, 'neon.injected_atom_density': 5e19})``. The comparison is
        done in log space for positive axes. Unspecified axes default to index 0.
        """
        coords = dict(coords or {}, **kwargs)
        idx = [0] * len(self.axes)
        for key, value in coords.items():
            k = self._axis_position(key)
            raw = np.asarray(self.axes[k][1])
            if raw.dtype.kind in "iuf" and isinstance(value, str):
                try:                     # YAML 1.1 reads 1.0e19 (unsigned exponent) as a string
                    value = float(value)
                except ValueError:
                    pass
            if raw.dtype.kind in "iuf" and not isinstance(value, (bool, str)):
                values = raw.astype(float)
                if np.all(values > 0) and value > 0:
                    idx[k] = int(np.argmin(np.abs(np.log(values) - np.log(value))))
                else:
                    idx[k] = int(np.argmin(np.abs(values - value)))
            else:                                # boolean or text axis: exact match
                matches = [i for i, v in enumerate(raw.tolist()) if v == value or str(v) == str(value)]
                if not matches:
                    raise KeyError(f"{value!r} is not a value of the scan axis '{self.axes[k][0]}'.")
                idx[k] = matches[0]
        return tuple(idx)

    def coordinates(self, idx: Tuple[int, ...]) -> Dict[str, object]:
        """Scan-axis values of a grid point (native Python types)."""
        out = {}
        for (name, values), i in zip(self.axes, idx):
            v = np.asarray(values)[i]
            out[name] = v.item() if hasattr(v, "item") else v
        return out

    def point_config(self, idx: Tuple[int, ...]) -> Config:
        base = replace(self.config, scan={})
        return base.with_overrides(self.coordinates(idx)) if self.axes else base

    def run(self, idx=None, **kwargs) -> RunResult:
        """RunResult of one scan point, selected by index tuple or by values (see :meth:`index`)."""
        if idx is None or isinstance(idx, dict):
            idx = self.index(idx, **kwargs)
        idx = tuple(idx)
        n = int(self.n_valid[idx])
        fluid = {name: self.fluid[name][idx][:n] for name in FLUID_NAMES}
        rp = None
        if self.r_p_samples is not None:
            rp = self.r_p_samples[idx]
            rp = rp[np.isfinite(rp)]             # arrays are padded when n_shards is scanned
        en = None if self.energies is None else {k: v[idx][:n] for k, v in self.energies.items()}
        if en is not None and any(np.all(np.isnan(v)) for v in en.values()):
            en = None
        config = self.point_config(idx)
        if self.fhte_overrides:
            config = config.with_overrides(self.fhte_overrides)
        return RunResult(t=self.time[idx][:n], fluid=fluid, nij=self.nij[idx][:n], elements=self.elements,
                         atomic_numbers=self.atomic_numbers,
                         config=config, status=int(self.status[idx]),
                         message=str(self.messages[idx]), elapsed=float(self.elapsed[idx]), r_p_samples=rp,
                         energies=en, fhte=self.fhte_result(idx))

    def fhte_result(self, idx):
        """Stored hot-tail estimate (tqtoy.fhte.FHTEResult) of one point, or None."""
        from .fhte import FHTEResult
        if self.fhte is None or not np.isfinite(self.fhte["T0"][tuple(idx)]):
            return None
        d = {k: v[tuple(idx)] for k, v in self.fhte.items()}
        n = int(np.sum(np.isfinite(d["t_output"])))
        m = int(np.sum(np.isfinite(d["t_trajectory"][min(1, n - 1)]))) if n > 1 else 0
        kw = {k: d[k][:n] for k in FHTE_1D}
        kw.update({k: d[k][:n, :m] for k in FHTE_2D})
        return FHTEResult(**kw, T0=float(d["T0"]), n0=float(d["n0"]), status=int(d["status"]))

    def __iter__(self):
        for idx in np.ndindex(*self.shape):
            yield idx, self.run(idx)


# --------------------------------------------------------------------------- execution

FHTE_1D = ("t_output", "fraction", "p_c", "p_limit", "E", "E_c", "T", "n")
FHTE_2D = ("p_trajectory", "t_trajectory")
FHTE_0D = ("T0", "n0", "status")


def _alloc_fhte(shape, n_out, n_traj):
    d = {k: np.full(shape + (n_out,), np.nan) for k in FHTE_1D}
    d.update({k: np.full(shape + (n_out, n_traj), np.nan) for k in FHTE_2D})
    d.update({k: np.full(shape, np.nan) for k in FHTE_0D})
    return d


def _store_fhte(store, idx, res):
    n = len(res.t_output)
    for k in FHTE_1D:
        store[k][idx + (slice(0, n),)] = getattr(res, k)
    m = res.p_trajectory.shape[1]
    for k in FHTE_2D:
        store[k][idx + (slice(0, n), slice(0, m))] = getattr(res, k)
    store["T0"][idx], store["n0"][idx], store["status"][idx] = res.T0, res.n0, res.status


def _run_fhte(run, atomic=None, config=None):
    """Hot-tail estimate of a run, or (None, message) if it cannot be computed."""
    from .fhte import hot_tail_from_run
    if len(run.t) < 2:
        return None, "FHTE: fewer than 2 valid output times"
    try:
        return hot_tail_from_run(run, config=config, atomic=atomic), ""
    except Exception as exc:                              # keep the scan going, report the point
        return None, f"FHTE failed: {exc}"


def _simulate_point(config: Config, atomic=None) -> RunResult:
    """Run one point. A failing integration is reported as a failed point, not as an exception,
    so that one bad point does not stop a scan."""
    try:
        return simulate(config, atomic=atomic)
    except Exception as exc:
        if atomic is None:
            try:
                atomic = get_atomic_data(config.element_names)  # already loaded in this process
            except Exception:
                raise exc from None                             # the atomic data itself is unusable
        layout = StateLayout(tuple(atomic.atomic_numbers))
        return RunResult(t=np.zeros(0), fluid={name: np.zeros(0) for name in FLUID_NAMES},
                         nij=np.zeros((0, layout.z_max + 1, len(layout.atomic_numbers))),
                         elements=list(config.element_names), atomic_numbers=list(layout.atomic_numbers),
                         config=config, status=STATUS_FAILED,
                         message=f"{type(exc).__name__}: {exc}", elapsed=0.0)


def _worker(config_dict):
    r = _simulate_point(Config.from_dict(config_dict))
    fh = None
    if r.config.fhte.enabled:
        fh, msg = _run_fhte(r)
        if msg:
            r.message = (r.message + " | " if r.message else "") + msg
    return r.t, r.fluid, r.nij, r.status, r.message, r.elapsed, r.r_p_samples, r.energies, fh


def _print_progress(done: int, total: int, t0: float) -> None:
    tty = sys.stderr.isatty()
    if not tty and done != total and done % max(total // 10, 1):
        return                               # log files: about one line per 10 %
    elapsed = _time.time() - t0
    remaining = elapsed / max(done, 1) * (total - done)
    print(f"{chr(13) if tty else ''}[tqtoy] {done}/{total} runs ({100 * done / total:5.1f} %) - "
          f"elapsed {elapsed:6.0f} s - remaining ~{remaining // 60:.0f} min {remaining % 60:02.0f} s",
          end="" if (tty and done < total) else "\n", file=sys.stderr, flush=True)


def run_scan(config: Config, jobs: int = 1, provider=None, progress: bool = True,
             callback: Optional[Callable[[Tuple[int, ...], RunResult], None]] = None) -> ScanResults:
    """Run every point of ``config.scan`` (a single run if the scan is empty).

    Parameters
    ----------
    jobs : number of worker processes (1 = serial)
    provider : custom atomic data provider (serial runs only)
    progress : print progress and remaining time on stderr
    callback : optional function called as callback(index, RunResult) after each serial run
    """
    axes = config.scan_axes()
    shape = tuple(len(v) for _, v in axes)
    base = replace(config, scan={})
    for path, _ in axes:
        parts = path.split(".")
        if parts[0] == "impurities" and parts[1] not in base.impurities:
            raise ConfigError(f"scan.{path}: impurity '{parts[1]}' must also be defined in the impurities section.")
        if path == "output.file":
            raise ConfigError("scan.output.file: all scan points are written to one file.")

    # Configuration of every point, validated before any run
    points = []
    for idx, overrides in config.scan_points():
        try:
            cfg = base.with_overrides(overrides) if overrides else base
        except ConfigError as exc:
            raise ConfigError(f"scan point {dict(overrides)}: {exc}") from None
        points.append((idx, cfg))

    grids = [cfg.output_times() for _, cfg in points]
    n_t = max(len(g) for g in grids)
    layout = StateLayout(tuple(get_atomic_data(base.element_names, provider).atomic_numbers))
    n_el = len(layout.atomic_numbers)

    time = np.full(shape + (n_t,), np.nan)
    fluid = {name: np.full(shape + (n_t,), np.nan) for name in FLUID_NAMES}
    nij = np.full(shape + (n_t, layout.z_max + 1, n_el), np.nan)
    n_valid = np.zeros(shape, dtype=int)
    status = np.full(shape, STATUS_NOT_RUN, dtype=int)
    elapsed = np.zeros(shape)
    messages = np.full(shape, "", dtype=object)
    parks = [cfg.injection.ablation.n_shards for _, cfg in points
             if cfg.injection.source == "ablation" and cfg.injection.ablation.parks is not None]
    r_p_samples = np.full(shape + (max(parks),), np.nan) if parks else None
    energies = None
    if any(cfg.output.energy_integrals for _, cfg in points):
        energies = {k: np.full(shape + (n_t,), np.nan) for k in ENERGY_KEYS}
    fhte_cfgs = [cfg.fhte for _, cfg in points if cfg.fhte.enabled]
    fhte = None
    if fhte_cfgs:
        fhte = _alloc_fhte(shape, max(c.n_output for c in fhte_cfgs), max(c.n_trajectory for c in fhte_cfgs))

    total = len(points)
    t0 = _time.time()

    def store(idx, out):
        t, fl, nj, st, msg, el, rp, en, fh = out
        if fhte is not None and fh is not None:
            _store_fhte(fhte, idx, fh)
        n = len(t)
        time[idx + (slice(0, n),)] = t
        for name in FLUID_NAMES:
            fluid[name][idx + (slice(0, n),)] = fl[name]
        nij[idx + (slice(0, n),)] = nj
        n_valid[idx], status[idx], messages[idx], elapsed[idx] = n, st, msg, el
        if r_p_samples is not None and rp is not None:
            r_p_samples[idx + (slice(0, len(rp)),)] = rp
        if energies is not None and en is not None:
            for k in ENERGY_KEYS:
                energies[k][idx + (slice(0, n),)] = en[k]

    if jobs <= 1:
        atomic = get_atomic_data(base.element_names, provider)
        for done, (idx, cfg) in enumerate(points, start=1):
            r = _simulate_point(cfg, atomic=atomic)
            if cfg.fhte.enabled:
                r.fhte, msg = _run_fhte(r, atomic=atomic)
                if msg:
                    r.message = (r.message + " | " if r.message else "") + msg
            store(idx, (r.t, r.fluid, r.nij, r.status, r.message, r.elapsed, r.r_p_samples, r.energies, r.fhte))
            if callback is not None:
                callback(idx, r)
            if progress:
                _print_progress(done, total, t0)
    else:
        if provider is not None:
            raise ConfigError("A custom atomic data provider cannot be used with jobs > 1.")
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_worker, cfg.to_dict()): idx for idx, cfg in points}
            for done, fut in enumerate(as_completed(futures), start=1):
                store(futures[fut], fut.result())
                if progress:
                    _print_progress(done, total, t0)

    # Points that did not return any time keep their nominal grid
    for (idx, _), grid in zip(points, grids):
        if n_valid[idx] == 0:
            time[idx + (slice(0, len(grid)),)] = grid

    return ScanResults(config=config, axes=axes, time=time, fluid=fluid, nij=nij, n_valid=n_valid,
                       status=status, elapsed=elapsed, messages=messages, elements=base.element_names,
                       atomic_numbers=list(layout.atomic_numbers),
                       r_p_samples=r_p_samples, energies=energies, fhte=fhte)


# --------------------------------------------------------------------------- post-hoc hot-tail estimate

def _fhte_worker(run):
    return _run_fhte(run)


def compute_fhte(results: ScanResults, overrides: Optional[Dict] = None, jobs: int = 1, progress: bool = True,
                 provider=None) -> ScanResults:
    """Run the Fast Hot Tail Estimator on every point of stored results (in place).

    ``overrides`` changes FHTE settings for all points, e.g. {"fhte.n_output": 10}.
    The FHTE section of each point configuration is used otherwise (scanned FHTE
    parameters are honoured). Returns ``results`` with ``results.fhte`` filled.
    """
    overrides = dict(overrides or {})
    bad = [k for k in overrides if not k.startswith("fhte.")]
    if bad:
        raise ConfigError(f"Only fhte.* parameters can be changed after the run (got {bad}).")
    points = []
    for idx in np.ndindex(*results.shape):
        run = results.run(idx)
        run.fhte = None
        if overrides:
            run.config = run.config.with_overrides(overrides)
        run.config.fhte.validate(two_populations=run.two_populations)
        points.append((idx, run))
    store = _alloc_fhte(results.shape, max(r.config.fhte.n_output for _, r in points),
                        max(r.config.fhte.n_trajectory for _, r in points))
    total, t0 = len(points), _time.time()
    messages = []

    def keep(idx, out):
        res, msg = out
        if res is not None:
            _store_fhte(store, idx, res)
        if msg:
            messages.append((idx, msg))

    if jobs <= 1 or provider is not None:
        atomic = None
        for done, (idx, run) in enumerate(points, start=1):
            if run.config.fhte.field == "model" and atomic is None:
                atomic = get_atomic_data(results.elements, provider)
            keep(idx, _run_fhte(run, atomic=atomic))
            if progress:
                _print_progress(done, total, t0)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_fhte_worker, run): idx for idx, run in points}
            for done, fut in enumerate(as_completed(futures), start=1):
                keep(futures[fut], fut.result())
                if progress:
                    _print_progress(done, total, t0)
    for idx, msg in messages:
        results.messages[idx] = (str(results.messages[idx]) + " | " if results.messages[idx] else "") + msg
    results.fhte = store
    results.fhte_overrides = {**(results.fhte_overrides or {}), **overrides}
    return results
