# TQTOY
A 0D model of the tokamak thermal quench triggered by a massive material injection, such as a shattered pellet.

The plasma is described by one or two thermal populations, each with electrons and main ions.
Impurity charge states evolve with non-coronal ADAS rates. Everything is local: densities,
temperatures, and powers per unit volume.

A single run takes about one second. A 400-point scan takes a few minutes and runs in parallel.

## Install

```bash
pip install -e .        # add [test] for pytest
tqtoy install-adas      # downloads the OpenADAS rate files once
```

Python >= 3.9, with NumPy, SciPy, h5py, Matplotlib, PyYAML and
[CHERAB](https://www.cherab.info) for the atomic data.

To run without installing, add `src` to `PYTHONPATH` and use `python -m tqtoy` in place of
`tqtoy`.

## Thirty seconds

```bash
tqtoy init                            # writes single_run_input.yaml, every parameter commented
tqtoy run single_run_input.yaml       # writes results/tqtoy_results.h5
tqtoy summary results/tqtoy_results.h5   # characteristic times and energies
```

```
  t_TQ                         1.009e-05 s
  W_th_initial                 9.612e+05 J/m^3
  radiated_fraction            0.7418
```

From Python:

```python
from tqtoy import Config, simulate

run = simulate(Config.from_yaml("single_run_input.yaml"))
print(run.Te_avg[-1], "eV")
```

**Next: [docs/TUTORIAL.md](docs/TUTORIAL.md)** walks through a first run, the figures, a
parameter scan and the Python API, with every command verified.

## Documentation

| | |
|---|---|
| [docs/TUTORIAL.md](docs/TUTORIAL.md) | how to drive the code, step by step |
| [docs/MODEL_EQUATIONS.pdf](docs/MODEL_EQUATIONS.pdf) | every equation of the model, typeset |
| [docs/INPUT_PARAMETERS.md](docs/INPUT_PARAMETERS.md) | every configuration parameter, with unit and default |
| [docs/INPUT_FILES.md](docs/INPUT_FILES.md) | the entries of each command's input file |
| [docs/PLOTTING.md](docs/PLOTTING.md) | the appearance options of the figures |
| [docs/API.md](docs/API.md) | the Python objects |

The same tables are printed by `tqtoy params`, `tqtoy tasks` and `tqtoy style`. Every command
has a `--help`.

`docs/MODEL_EQUATIONS.pdf` is built from `docs/MODEL_EQUATIONS.tex` with
`pdflatex MODEL_EQUATIONS.tex`.

## Commands

| Command | What it does |
|---|---|
| `tqtoy init` | write a commented input file (`-t plot`, `-t map`, `-t iter_scan`, ...) |
| `tqtoy run FILE` | run one simulation or a scan (`-j` for parallel) |
| `tqtoy summary FILE` | characteristic times, energies, energy balance |
| `tqtoy info FILE` | scan axes, run status, contents of a results file |
| `tqtoy plot FILE` | time traces of one point, or a comparison over a scan axis |
| `tqtoy map FILE` | a scalar quantity over a 1D or 2D scan |
| `tqtoy fhte FILE` | hot-tail runaway estimate |
| `tqtoy params`, `tasks`, `style` | the parameter, input-file and appearance tables |
| `tqtoy rates ELEMENT` | plot the ADAS rates of an element |
| `tqtoy install-adas` | download the atomic data |

Every command is driven by a YAML input file, and any entry can be overridden with `--set` and
its dotted path.

## Code layout

```
src/tqtoy/
  config.py        configuration dataclasses, YAML loading, validation, scans
  atomic.py        atomic data (OpenADAS through CHERAB, or any provider)
  physics/         point functions: collisions, resistivity, impurities, power, ablation
  model.py         state vector and right-hand side of the ODE system
  solver.py        time integration of one configuration (RunResult)
  scan.py          parameter scans (ScanResults), serial or parallel
  io.py            HDF5 input and output
  fhte.py          Fast Hot Tail Estimator (hot-tail runaway electrons)
  diagnostics.py   power terms, characteristic times, energy balance
  plotting.py      standard figures
  tasks.py         input files of the figure commands (plot, map)
  cli.py           command-line interface
  banner.py        the wordmark printed at the start of a run
```

`tests/` holds the test suite. `examples/` holds a copy of every input-file template and a
Python script.

## License

To be defined by the authors before publication.
