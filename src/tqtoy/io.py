"""HDF5 storage of scan results.

File layout::

    /                  attrs: format, tqtoy_version, created, config (YAML), elements, atomic_numbers
    /time              output times of each point [s], shape (*shape, n_t), NaN padded
    /scan/<path>       values of each scan axis (attr 'order' on /scan gives the axis order)
    /results/<name>    Te_hot ... ni_cold (*shape, n_t), nij (*shape, n_t, Z_max+1, n_el),
                       n_valid, status, elapsed, message, r_p_samples (optional),
                       E_ohm, E_rad, E_stoch, E_lin (optional, cumulated energies [J m^-3])
    /fhte/<name>       hot-tail estimates (optional): t_output, fraction, p_c, p_limit, E, E_c, T, n
                       (*shape, n_out), p_trajectory, t_trajectory (*shape, n_out, n_traj), T0, n0, status

The complete configuration is stored, so a file is self-describing and a run can
be reproduced with ``tqtoy run`` on the extracted configuration (``tqtoy info --config``).
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import h5py
import numpy as np

from . import __version__
from .config import Config
from .model import FLUID_NAMES
from .scan import ScanResults
from .solver import ENERGY_KEYS

FORMAT = "tqtoy-results-2"


def save_results(results: ScanResults, path) -> Path:
    """Write ScanResults to an HDF5 file (overwritten if it exists)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    str_dt = h5py.string_dtype()
    with h5py.File(path, "w") as f:
        f.attrs["format"] = FORMAT
        f.attrs["tqtoy_version"] = __version__
        f.attrs["created"] = datetime.datetime.now().isoformat(timespec="seconds")
        f.attrs["config"] = results.config.to_yaml()
        f.attrs["elements"] = json.dumps(results.elements)
        f.attrs["atomic_numbers"] = json.dumps([int(z) for z in results.atomic_numbers])
        f.create_dataset("time", data=results.time)
        g = f.create_group("scan")
        g.attrs["order"] = json.dumps(results.axis_names)
        for name, values in results.axes:
            values = np.asarray(values)
            if values.dtype.kind in "USO":           # text axis (e.g. model.resistivity)
                g.create_dataset(name, data=values.astype(object), dtype=str_dt)
            else:
                g.create_dataset(name, data=values)
        r = f.create_group("results")
        for name in FLUID_NAMES:
            r.create_dataset(name, data=results.fluid[name], compression="gzip")
        r.create_dataset("nij", data=results.nij, compression="gzip")
        r.create_dataset("n_valid", data=results.n_valid)
        r.create_dataset("status", data=results.status)
        r.create_dataset("elapsed", data=results.elapsed)
        r.create_dataset("message", data=np.asarray(results.messages, dtype=object).astype(str).astype(object),
                         dtype=str_dt)
        if results.r_p_samples is not None:
            r.create_dataset("r_p_samples", data=results.r_p_samples)
        if results.energies is not None:
            for key, arr in results.energies.items():
                r.create_dataset(key, data=arr, compression="gzip")
        if results.fhte is not None:
            _write_fhte(f, results)
    return path


def _write_fhte(f, results: ScanResults) -> None:
    if "fhte" in f:
        del f["fhte"]
    g = f.create_group("fhte")
    g.attrs["overrides"] = json.dumps(results.fhte_overrides or {})
    for key, arr in results.fhte.items():
        g.create_dataset(key, data=arr, compression="gzip" if np.ndim(arr) > 0 else None)


def save_fhte(results: ScanResults, path) -> Path:
    """Add (or replace) the hot-tail results in an existing results file, leaving the rest unchanged."""
    with h5py.File(path, "a") as f:
        if f.attrs.get("format") != FORMAT:
            raise ValueError(f"{path} is not a tqtoy results file.")
        f["results"]["message"][...] = np.asarray(results.messages, dtype=object).astype(str).astype(object)
        _write_fhte(f, results)
    return Path(path)


def load_results(path) -> ScanResults:
    """Read a file written by :func:`save_results`."""
    with h5py.File(path, "r") as f:
        if f.attrs.get("format") != FORMAT:
            raise ValueError(f"{path} is not a tqtoy results file (format '{f.attrs.get('format')}').")
        config = Config.from_yaml_string(f.attrs["config"])
        elements = json.loads(f.attrs["elements"])
        time = f["time"][...]
        order = json.loads(f["scan"].attrs["order"])
        axes = []
        for name in order:
            ds = f["scan"][name]
            values = ds.asstr()[...] if h5py.check_string_dtype(ds.dtype) else ds[...]
            axes.append((name, np.asarray(values)))
        r = f["results"]
        fluid = {name: r[name][...] for name in FLUID_NAMES}
        messages = np.array([m.decode() if isinstance(m, bytes) else str(m) for m in r["message"][...].ravel()],
                            dtype=object).reshape(r["message"].shape)
        res = ScanResults(config=config, axes=axes, time=time, fluid=fluid, nij=r["nij"][...],
                          n_valid=r["n_valid"][...], status=r["status"][...], elapsed=r["elapsed"][...],
                          messages=messages, elements=elements,
                          atomic_numbers=json.loads(f.attrs["atomic_numbers"]),
                          r_p_samples=r["r_p_samples"][...] if "r_p_samples" in r else None,
                          energies={k: r[k][...] for k in ENERGY_KEYS} if all(k in r for k in ENERGY_KEYS) else None,
                          metadata={k: str(f.attrs[k]) for k in ("tqtoy_version", "created")})
        if "fhte" in f:
            res.fhte = {k: f["fhte"][k][...] for k in f["fhte"]}
            res.fhte_overrides = json.loads(f["fhte"].attrs.get("overrides", "{}")) or None
    return res
