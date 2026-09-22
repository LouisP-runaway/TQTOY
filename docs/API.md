# Python API

The objects of the package. Every function and class also has a docstring:
`help(tqtoy.simulate)`, `help(tqtoy.scan.ScanResults)`. Section 8 of
[TUTORIAL.md](TUTORIAL.md) is a worked example, and `examples/api_example.py` a longer script.

Units: SI, temperatures in eV, densities in m^-3, powers in W m^-3, energies in J m^-3.

## Configuration: `tqtoy.Config`

| Member | Description |
|---|---|
| `Config()` | Default configuration (the ITER-like single run) |
| `Config.from_yaml(path)`, `Config.from_dict(d)` | Read and validate a configuration |
| `cfg.with_overrides({"plasma.Te0": 1e4, ...})` | Copy with parameters replaced (dotted paths, also `scan.<path>`) |
| `cfg.plasma.Te0`, `cfg.injection.n_D`, ... | Parameters as attributes |
| `cfg.impurities["neon"].injected_atom_density` | Impurity parameters (dictionary keyed by element) |
| `cfg.scan_axes()` | List of `(path, values)` of the scan |
| `cfg.output_times()` | Output time grid |
| `cfg.to_yaml()`, `cfg.to_dict()` | Export |

`tqtoy.config.parameter_table()` returns the list of parameters with type, default, unit and
description.

## One simulation: `tqtoy.simulate(config) -> RunResult`

`simulate(config, atomic=None, provider=None, t_eval=None)` integrates one configuration. The
`scan` section is ignored. `t_eval` replaces the output grid of the configuration.

`RunResult` attributes:

| Attribute | Shape | Description |
|---|---|---|
| `t` | (n_t,) | Output times [s] |
| `Te_hot`, `Te_cold`, `Ti_hot`, `Ti_cold` | (n_t,) | Temperatures [eV] |
| `ne_hot`, `ne_cold`, `ni_hot`, `ni_cold` | (n_t,) | Densities [m^-3] |
| `nij` | (n_t, Z_max+1, n_elements) | Impurity charge-state densities [m^-3] |
| `n_imp` | (n_t, n_elements) | Total density of each impurity element |
| `Te_avg`, `ne_total` | (n_t,) | Density-weighted Te and free-electron density of the active populations |
| `energies` | dict of (n_t,) | Cumulated `E_ohm`, `E_rad`, `E_stoch`, `E_lin` [J m^-3] |
| `elements`, `atomic_numbers` | lists | Impurity elements |
| `config` | Config | Configuration of the run |
| `status`, `message`, `ok` | | 0 completed, 1 stopped (negative value), -1 failed |
| `r_p_samples` | (n_shards,) or None | Sampled shard radii (Parks) [m] |

With one population, the plasma is in the `*_cold` arrays.

## Scans: `tqtoy.scan.run_scan(config, jobs=1) -> ScanResults`

Runs every point of `config.scan`. `jobs > 1` uses several processes. On Windows and macOS,
protect the calling script with `if __name__ == "__main__":`.

`ScanResults` members:

| Member | Description |
|---|---|
| `axes`, `axis_names`, `shape` | Scan axes `(path, values)` and grid shape |
| `time` | Output times of each point, shape `(*shape, n_t)` (they differ if a time parameter is scanned) |
| `t` | Output times shared by all points (error if a time parameter is scanned) |
| `fluid["Te_cold"]`, ... | Arrays of shape `(*shape, n_t)`, NaN after the last valid time |
| `nij` | Array `(*shape, n_t, Z_max+1, n_elements)` |
| `status`, `n_valid`, `messages`, `elapsed` | Per-point run information |
| `index(Te0=5e3)` | Grid index of the closest point (full path or unique suffix of the axis name) |
| `run(Te0=5e3)` or `run((i, j))` | `RunResult` of one point |
| `coordinates(idx)`, `point_config(idx)` | Axis values and configuration of one point |
| `for idx, run in results:` | Iterate over all points |

`TQModel` (`tqtoy.model`) holds the right-hand side: `model.rhs(t, y)` and, with
`solver.jacobian: analytic`, `model.jacobian(t, y)`. `model.evaluate(t, vals, flags, nij)`
returns every intermediate term and is what the diagnostics call.

## Files: `tqtoy.io`

`save_results(results, path)` and `load_results(path) -> ScanResults`. A single run is a scan
with no axis (`results.run()` returns it).

## Diagnostics: `tqtoy.diagnostics`

| Function | Returns |
|---|---|
| `summary(run)` | Dict of scalars: `t_TQ`, `t_radiative_collapse`, `t_ohmic_radiative_balance`, `t_runaway_onset`, energies, `radiated_fraction`, `energy_residual_rel` |
| `compute_terms(run)` | Dict of time arrays of every term of the model (see below) |
| `energy_balance(run)` | `W_th`, `E_ohm`, `E_rad`, `E_stoch`, `E_lin`, `residual` |
| `thermal_quench_time(run, threshold=100.)` | First time with `Te_avg < threshold` [s] |
| `scan_map(results, "t_TQ")` | Array over the scan grid (`t_TQ`, `Te_avg_final`, `t_radiative_collapse`, `radiated_fraction`, `radiated_fraction_net`, `n_RE_hot_tail`, `n_RE_hot_tail_density`) |

Keys of `compute_terms`: `J`, `J_hot`, `J_cold`, `Zeff`, `eta_hot`, `eta_cold`, `E_hot`,
`E_cold` (parallel field), `P_ohm_*`, `P_rad_*`, `P_stoch_*` (`*` = `hot` or `cold`),
`P_loss_lin_e`, `P_loss_lin_i`, the exchange powers `P_<a>_<b>` from species a to b
(`P_ehot_ecold`, `P_ihot_icold`, `P_ehot_ihot`, `P_ehot_icold`, `P_ecold_icold`,
`P_ecold_ihot`) and the sources `S_D`, `S_imp`, `g_hot`, `g_cold`.

## Figures: `tqtoy.plotting`

`make_figures(run, names, outdir=None, style=None)` draws the figures listed in
`tqtoy.plotting.FIGURES` (`temperatures`, `densities`, `charge_states`, `zeff`, `mean_charge`,
`powers`, `energy`, `currents`, `efield`, `exchange`, `timescales`, `cooling_curve`,
`electron_budget`, `sources`, `shards`, `hot_tail`). Each `plot_*` function returns a Matplotlib
figure. `plot_radiation_rates("neon")` draws the ADAS radiation coefficients.

Over a scan:

| Function | Description |
|---|---|
| `plot_scan_map(results, quantity, style=None)` | Scalar quantity over the scan: a line (1D scan) or a map (2D scan) |
| `plot_scan_series(results, variable, axis=None, at=None, points=None, style=None)` | Time traces of the points of one scan axis on one figure, coloured by the scanned value |

`tqtoy.plotting.SERIES` lists the variables of `plot_scan_series` (`Te_avg`, `ne_total`, `n_imp`,
`Zeff`, `P_rad`, `P_ohm`, `P_stoch`, `E_parallel`, `W_th`, `n_RE_hot_tail`...).

`style` is a `PlotStyle`, a dict of its fields, or None. The fields are listed in
[PLOTTING.md](PLOTTING.md).

```python
from tqtoy.plotting import PlotStyle, plot_scan_map
st = PlotStyle(cmap="magma", contours=False, colors={"cold": "black"})
st = PlotStyle.from_yaml("mystyle.yaml").with_overrides({"filled": True})
fig = plot_scan_map(results, "t_TQ", style=st)
```

## Hot-tail runaways: `tqtoy.fhte`

| Function | Description |
|---|---|
| `hot_tail_estimate(t, Te, ne, J=None, E=None, config=None, **options)` | Estimate on prescribed traces (arrays or scalars). `options` override `FHTEConfig` fields |
| `hot_tail_from_run(run, config=None)` | Estimate on a `RunResult` (settings: `run.config.fhte` by default) |
| `run_prescribed(FHTERunConfig)` | Standalone run from a prescribed-evolution configuration (see `tqtoy init -t fhte_prescribed`) |
| `save_prescribed`, `load_prescribed` | Standalone result files |
| `tqtoy.scan.compute_fhte(results, overrides, jobs)` | Estimate on every point of `ScanResults`, stored in `results.fhte` |
| `tqtoy.io.save_fhte(results, path)` | Store these estimates in an existing results file |
| `runaway_fraction`, `critical_momentum`, `critical_field`, `coulomb_log`, `drag_coefficient` | Physics functions of the estimator |

The evaluation times are placed where E > E_c (`fhte.output_window`), and `fraction` is the
running maximum over them (`fhte.accumulate`). `p_limit` is NaN at the times where no backward
trajectory was integrated (E <= E_c).

`FHTEResult` attributes: `t_output`, `fraction` (n_RE/n0), `density` (n_RE [m^-3]), `p_c`,
`p_limit`, `E`, `E_c`, `T`, `n` (arrays over the evaluation times), `p_trajectory`,
`t_trajectory` (backward trajectories), `T0`, `n0`, `final_fraction`, `final_density`, `status`,
`message`. A `RunResult` of a scan with stored estimates carries it as `run.fhte`.

Scan-map quantities: `n_RE_hot_tail` (final n_RE/n0) and `n_RE_hot_tail_density`.

## Input files of the commands: `tqtoy.tasks`

| Member | Description |
|---|---|
| `PlotTask`, `MapTask`, `CompareSpec` | Entries of the `tqtoy plot` and `tqtoy map` input files |
| `from_yaml(cls, path)`, `from_dict(cls, data)` | Read and validate one |
| `with_overrides(task, {"compare.variable": "ne_total"})` | Copy with dotted-path entries replaced |
| `task_table(cls)`, `tasks_text(name, markdown)` | Entries with type, default and description (documentation) |

`tqtoy.fhte.FHTERunConfig` is the input file of `tqtoy fhte` and `tqtoy.Config` that of
`tqtoy run`.

## Physics functions: `tqtoy.physics`

Point functions used by the model, usable on their own: `collisions` (Coulomb logarithms,
collision times, exchange power), `resistivity` (Z_eff, Spitzer and multi-species resistivity,
current sharing, critical field), `impurities` (charge-state balance, radiated power), `power`
(ohmic, stochastic), `ablation` (ablation rate, Parks distribution).

`impurities.coronal_fractions(el, ne, Te)` and `impurities.coronal_cooling_rate(el, ne, Te)` give
the coronal equilibrium of one element, which the `cooling_curve` figure compares the run to.
