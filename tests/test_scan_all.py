"""Every configuration parameter can be a scan axis (analytic atomic rates, no ADAS)."""

import numpy as np
import pytest

from tqtoy import Config
from tqtoy.config import ConfigError, parameter_table
from tqtoy.scan import run_scan

BASE = {"plasma": {"Te0": 3e3, "ne0": 5e19},
        "impurities": {"neon": {"injected_atom_density": 1e19}},
        "injection": {"n_D": 3e20},
        "time": {"t_end": 2e-4, "n_output": 20},
        "output": {"energy_refinement": 2}}

# Values of the parameters that cannot be derived from their default value
SPECIAL = {
    "model.populations": [1, 2],
    "model.resistivity": ["spitzer", "multi_species"],
    "injection.source": ["prescribed", "ablation"],
    "time.spacing": ["log", "linear"],
    "solver.method": ["Radau", "BDF"],
    "plasma.Ti0": [2e3, 4e3],
    "plasma.ni0": [5e19, 6e19],
    "plasma.J_final": [1e5, 5e5],
    "plasma.J_ramp_time": [5e-5, 1e-4],
    "injection.ablation.D2_fraction": [0.5, 0.9],
    "injection.ablation.seed": [1, 2],
    "injection.charge_state": [0, 1],
    "time.t_start": [1e-7, 1e-6],
    "model.linear_loss_coefficient": [0.0, 1e3],
    "impurities.<element>.background_density": [0.0, 1e17],
    "model.stochastic.r": [0.6, 0.7],                # r/a must stay below 1/sqrt(6)
    "model.stochastic.a": [2.0, 3.0],
    "fhte.field": ["spitzer", "model"],
    "fhte.temperature": ["average", "cold"],
    "fhte.output_spacing": ["linear", "log"],
    "fhte.trajectory_spacing": ["linear", "log"],
    "fhte.enabled": [False, True],
    "fhte.output_window": ["active", "full"],
    "solver.jacobian": ["numeric", "analytic"],
    "solver.state_variables": ["energy", "temperature"],
    "model.current_sharing": ["density_weighted", "spitzer"],
    "fhte.gamma_points": [20000, 40000],
    "fhte.bound_electron_weight": [0.0, 0.5],
}
# Base settings some parameters need to be active or valid
NEEDS = {
    "model.resistivity": {"model.populations": 2},
    "plasma.cold_T0": {"model.populations": 2},
    "plasma.cold_n0": {"model.populations": 2},
    "plasma.J_final": {"plasma.J_ramp_time": 1e-4},
    "plasma.J_ramp_time": {"plasma.J_final": 1e5},
    "injection.ablation.seed": {"injection.source": "ablation", "injection.ablation.parks.n_atoms": 1e24},
}
SKIP = {"scan", "output.file", "injection.ablation.parks"}


def _values(row):
    path = row["path"]
    if path in SPECIAL:
        return SPECIAL[path]
    if row["type"] == "bool":
        return [True, False]
    if row["type"] == "int":
        d = int(row["default"])
        return [d, 2 * d]
    d = float(row["default"])
    return [d, 1.5 * d]


def _needs(path):
    needs = dict(NEEDS.get(path, {}))
    if path.startswith("fhte.") and path != "fhte.enabled":
        needs.setdefault("fhte.enabled", True)
    if path.startswith("injection.ablation.") and path not in NEEDS:
        needs["injection.source"] = "ablation"
    return needs


ROWS = [r for r in parameter_table() if r["path"] not in SKIP]


def _get(cfg, path):
    node = cfg
    for key in path.split("."):
        node = node[key] if isinstance(node, dict) else getattr(node, key)
    return node


@pytest.mark.parametrize("row", ROWS, ids=[r["path"] for r in ROWS])
def test_parameter_can_be_scanned(row, fake_provider):
    path = row["path"].replace("<element>", "neon")
    values = _values(row)
    cfg = Config.from_dict(BASE).with_overrides({**_needs(row["path"]), f"scan.{path}": values})
    res = run_scan(cfg, provider=fake_provider, progress=False)
    assert res.shape == (2,)
    for i, v in enumerate(values):
        run = res.run((i,))
        assert _get(run.config, path) == v
        assert run.ok, run.message
        # the output times follow the configuration of each point
        np.testing.assert_allclose(run.t, run.config.output_times()[:len(run.t)])
        if run.config.fhte.enabled:
            assert run.fhte is not None and np.isfinite(run.fhte.final_fraction)


def test_all_parameters_are_covered():
    assert len(ROWS) == len(parameter_table()) - len(SKIP)


def test_output_file_cannot_be_scanned(fake_provider):
    cfg = Config.from_dict(BASE).with_overrides({"scan.output.file": ["a.h5", "b.h5"]})
    with pytest.raises(ConfigError, match="output.file"):
        run_scan(cfg, provider=fake_provider, progress=False)


def test_invalid_point_is_reported_before_running(fake_provider):
    cfg = Config.from_dict(BASE).with_overrides({"scan.model.resistivity": ["spitzer", "multi_species"]})
    with pytest.raises(ConfigError, match="scan point"):
        run_scan(cfg, provider=fake_provider, progress=False)


def test_time_grid_scan_round_trip(fake_provider, tmp_path):
    from tqtoy.io import load_results, save_results
    cfg = Config.from_dict(BASE).with_overrides({"scan.time.n_output": [10, 25]})
    res = load_results(save_results(run_scan(cfg, provider=fake_provider, progress=False), tmp_path / "t.h5"))
    assert len(res.run((0,)).t) == 10 and len(res.run((1,)).t) == 25
    with pytest.raises(ValueError, match="differ"):
        res.t
