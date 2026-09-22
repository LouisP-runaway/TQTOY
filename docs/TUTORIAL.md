# Tutorial

Every command below has been run as written. Copy them in order into an empty folder.

The physics is in [MODEL_EQUATIONS.pdf](MODEL_EQUATIONS.pdf). This page is only about driving
the code.

## 1. Install

```bash
pip install -e .        # add [test] for pytest
tqtoy install-adas      # downloads the OpenADAS rate files once
```

`tqtoy install-adas` fetches the atomic data from the CHERAB repository. It takes a few minutes
and is needed only once per machine.

Without installing, add `src` to the path and use `python -m tqtoy` in place of `tqtoy`:

```bash
export PYTHONPATH=/path/to/tqtoy/src     # Windows: set PYTHONPATH=C:\path\to\tqtoy\src
python -m tqtoy install-adas
```

## 2. First run

Write an input file with every parameter and its default value, then run it:

```bash
tqtoy init                          # writes single_run_input.yaml
tqtoy run single_run_input.yaml     # writes results/tqtoy_results.h5
```

The default case is ITER-like: 20 keV, 1e20 m^-3, 1 MA/m^2, with 1e22 m^-3 of deuterium and
5e19 m^-3 of neon deposited over 10 microseconds. It takes about a second.

## 3. Read the result

```bash
tqtoy summary results/tqtoy_results.h5
```

```
  status                       0
  t_last                       0.01 s
  Te_avg_final                 3.476 eV
  t_TQ                         1.009e-05 s
  W_th_initial                 9.612e+05 J/m^3
  E_radiated                   1.075e+06 J/m^3
  radiated_fraction            0.7418
  energy_residual_rel          0.0002592
```

`status` is 0 when the run reached the last output time. `t_TQ` is the thermal-quench time, the
first output time at which the density-weighted electron temperature falls below 100 eV.
`energy_residual_rel` checks the energy balance and should stay small.

`tqtoy info results/tqtoy_results.h5` prints the scan axes and the status of every point.

Results files go to `results/` and figures to `figures/` by default. Both directories are
created when they do not exist. `output.file` and the `outdir` entry of a figure input file
change them.

## 4. Figures

```bash
tqtoy init -t plot                  # writes plot_input.yaml and style_input.yaml
tqtoy plot plot_input.yaml          # writes the figures in figures/
```

`plot_input.yaml` already points at `results/tqtoy_results.h5`, so nothing has to be edited for
a single run. It writes `temperatures.png`, `densities.png`, `charge_states.png`, `powers.png`
and `energy.png`, the five default figures. `-f` selects others, and `-f all` draws every one:

| Figure | What it shows |
|---|---|
| `temperatures`, `densities` | the fluid quantities of each population |
| `charge_states` | the density of every charge state of every element |
| `zeff`, `mean_charge` | the effective charge, and the mean charge of each element |
| `powers`, `energy` | the power terms, and their time integrals against the thermal energy |
| `timescales` | the characteristic time of each loss and exchange channel |
| `cooling_curve` | the radiative cooling rate of the run against coronal equilibrium |
| `electron_budget` | where the free electrons come from |
| `currents`, `efield` | the current shared between the populations, the field against `E_c` |
| `exchange` | the collisional energy exchange between every pair of species |
| `sources` | the deposition or ablation rates |
| `shards` | the sampled shard radii (Parks sampling only) |
| `hot_tail` | the runaway fraction and the momentum trajectories |

`style_input.yaml` holds every appearance option (colours, colormap, time unit, fonts). See
[PLOTTING.md](PLOTTING.md).

## 5. Change a parameter

Either edit the YAML file, or override any entry on the command line with its dotted path:

```bash
tqtoy run single_run_input.yaml --set plasma.Te0=10e3 model.populations=2 -o results/two_pop.h5
```

`--set` accepts any parameter listed by `tqtoy params`. The full table is in
[INPUT_PARAMETERS.md](INPUT_PARAMETERS.md).

## 6. Scan a parameter

Add a `scan` section. Each axis is a dotted path and a list of values:

```yaml
plasma:
  Te0: 20.0e3
  ne0: 1.0e20
  J: 1.0e6

impurities:
  neon:
    injected_atom_density: 5.0e19

injection:
  n_D: 1.0e22

model:
  populations: 2

time:
  t_end: 1.0e-2
  n_output: 200

output:
  file: results/scan.h5

scan:
  injection.n_D: {logspace: [21, 23, 5]}
```

```bash
tqtoy run scan_input.yaml -j 4      # -j runs the points in parallel
tqtoy info results/scan.h5
```

```
scan axes :
  injection.n_D                               5 values in [1e+21, 1e+23]
  completed                   : 5
```

Values are given as a list, or as `{linspace: [start, stop, num]}`,
`{geomspace: [start, stop, num]}` or `{logspace: [exp_start, exp_stop, num]}`. Two axes form a
grid, the first being the outer loop.

## 7. Look at a scan

A scalar quantity against the scanned parameter, a line for one axis and a coloured map for
two:

```bash
tqtoy map results/scan.h5 -q t_TQ
```

The time traces of every point on one figure, coloured by the scanned value:

```bash
tqtoy plot results/scan.h5 -c Te_avg
```

The traces of a single point, the closest to the requested value:

```bash
tqtoy plot results/scan.h5 --at n_D=1e22
```

`tqtoy map --help` lists the scalar quantities and `tqtoy plot --help-variables` the traces.

## 8. From Python

```python
from tqtoy import Config, simulate
from tqtoy.diagnostics import summary

cfg = Config.from_dict({
    "plasma": {"Te0": 20e3, "ne0": 1e20},
    "impurities": {"neon": {"injected_atom_density": 5e19}},
    "injection": {"n_D": 1e22},
    "model": {"populations": 2},
})
run = simulate(cfg)

print(run.status, len(run.t))
print(summary(run)["t_TQ"], "s")
print(run.Te_avg[-1], "eV")
```

```
0 500
7.477346305177581e-06 s
3.4778045162479456 eV
```

`run` carries `t`, the eight fluid traces (`Te_hot`, `Te_cold`, `ne_hot`, `ne_cold`, `Ti_hot`,
`Ti_cold`, `ni_hot`, `ni_cold`), the charge states `nij`, and the derived `Te_avg`, `ne_total`
and `n_imp`. `Config.from_yaml("file.yaml")` reads an input file instead.

See [API.md](API.md) for the objects, and `examples/api_example.py` for a longer script.

## 9. Hot-tail runaway electrons

The estimator post-processes a finished run:

```bash
tqtoy fhte results/scan.h5              # adds the estimate to the file
tqtoy map results/scan.h5 -q n_RE_hot_tail
```

Setting `fhte.enabled: true` in the configuration runs it on every point at the end of
`tqtoy run` instead. `tqtoy init -t fhte_results` writes a commented input file with every
option.

## 10. Practical notes

**Check the convergence.** The default `solver.rtol = 1e-2` is loose. Individual charge-state
densities can move by a few percent when it is reduced. Reduce `rtol` and confirm that the
quantity you care about does not move.

**The system is stiff.** Use an implicit method. `Radau`, the default, or `BDF`. Explicit
methods such as `RK23` were about a hundred times slower in a test.

**Faster runs.** `solver.jacobian: analytic` differentiates the impurity charge-state columns
analytically instead of by finite differences. It is 1.1 to 1.5 times faster, more so with many
charge states, and only works with an implicit method. It changes the step sequence but not the
solution.

**A stopped run is not a lost run.** `status: 1` means a fluid quantity became negative and the
integration stopped there. The trace up to that point is valid, and `t_TQ` is usually already
found, since the quench happens in microseconds while such a stop happens in milliseconds.

**Two populations.** The injected material forms the cold population, the pre-disruption plasma
is the hot one. They exchange energy collisionally and no particles. Section 9 of
[MODEL_EQUATIONS.pdf](MODEL_EQUATIONS.pdf) derives their coupling.

## 11. Tests

```bash
pip install -e .[test]
pytest                     # the whole suite
pytest -m "not adas"       # only the tests that need no atomic data
```

The suite covers the physics, the configuration, the input and output, the figures and the
analytic Jacobian.
