# Changelog

## 1.0.0

First release.

### The model

* 0D multi-fluid thermal quench: one or two thermal populations, each with electrons and main
  ions, coupled by collisional energy exchange. Every equation is in
  `docs/MODEL_EQUATIONS.pdf`.
* Non-coronal impurity charge-state balance on the OpenADAS rates, read through CHERAB. Several
  impurity elements at once, each with its own inventory.
* Radiated power (line and continuum) with an optional low-temperature limiter, ohmic heating,
  Ward-Wesson stochastic transport loss, and an optional loss proportional to the temperature.
* Spitzer or multi-species resistivity, Z_eff from the charge states, current shared between
  the two populations in proportion to their electron densities.
* Two injection sources: a constant deposition rate over a window, or the ablation law for
  shattered pellets, with identical shards or radii sampled from the Parks distribution.
* Fast Hot Tail Estimator (`tqtoy.fhte`, `tqtoy fhte`), on a finished run or on a prescribed
  evolution of the bulk plasma.

### Driving it

* Every parameter is read from a validated YAML input file, with its unit and description
  (`tqtoy params`, `docs/INPUT_PARAMETERS.md`). `tqtoy init` writes a commented example.
* Any scalar parameter can be a scan axis, including booleans, text options and the time grid.
  One or two axes. All points are validated before the first run, and `-j` runs them in
  parallel.
* One HDF5 file per scan, holding the full configuration, the run status and the number of
  valid output times of each point.
* Command-line interface: `init`, `run`, `info`, `summary`, `plot`, `map`, `fhte`, `tasks`,
  `params`, `style`, `rates`, `install-adas`. Also `python -m tqtoy`, without installing.
* Any entry of any input file can be overridden on the command line with `--set` and its dotted
  path.
* Results files go to `results/` and figures to `figures/` by default. Both directories are
  created when they do not exist.
* A wordmark is printed at the start of `tqtoy run` and `tqtoy fhte`, on the error stream so
  that a redirected standard output stays clean. `-q` and the environment variable
  `TQTOY_NO_BANNER` switch it off, and a terminal narrower than the wordmark gets one line.
* The cells of a 2D map are placed by edges taken in the space of the axis. An arithmetic
  midpoint on a coarse logarithmic grid puts the first edge at a non-positive value, which a
  logarithmic axis drops along with every cell beyond it.
* Sixteen figures of one run (`tqtoy plot -f all`), among them the characteristic time of
  each channel, the radiative cooling rate of the run against coronal equilibrium, the mean
  charge of each element and the origin of the free electrons. Also the traces of a whole scan
  axis on one figure coloured by the scanned value, and maps of a scalar quantity over a 1D or
  2D scan. The comparison covers 21 quantities, among them the six collisional exchange powers.
  Every appearance option is in a style file (`tqtoy style`, `docs/PLOTTING.md`), whose defaults
  are the `jet` colormap and no isocontour lines.
* Python API: `Config`, `simulate`, `run_scan`, `RunResult`, `ScanResults`, the diagnostics and
  the plotting functions (`docs/API.md`).

### Solver

* `solver.jacobian: analytic` differentiates the impurity charge-state columns of the Jacobian
  analytically instead of by finite differences, and keeps the first step of the difference
  rule of scipy for the eight fluid columns. Runs are 1.1 to 1.5 times faster, more so with
  many charge states. It changes the step sequence and not the solution.
* `solver.state_variables` integrates either the four temperatures or the four energy densities
  n T. The two are the same equations, one multiplied by n, and outputs are temperatures either
  way.
* A quantity is limited by its own floor `solver.floor`, and only the part of a derivative that
  would push it further below is removed.
* The ADAS rate vectors are cached on (ne, Te). The right-hand side asks for the charge-state
  balance and for the radiated power at the same point, and the solver perturbs the state
  variables one at a time when it estimates the Jacobian, so most calls repeat a point that was
  just evaluated.
* A point whose integration fails or stops is flagged and the scan continues. Its message is
  reported by `tqtoy info`, and its trace is padded with NaN.

### Verification

* 299 tests covering the physics, the configuration, the input and output, the figures, the
  analytic Jacobian and the estimator. `pytest -m "not adas"` runs the 292 that need no atomic
  data.
* Five configurations are pinned in `tests/data` (one and two populations, prescribed and
  ablation sources, neon and argon, Spitzer and multi-species resistivity). The right-hand side
  is checked against its reference to 1e-12 and the trajectories to a tolerance above the
  round-off that a change of machine produces.
* Charge neutrality and the energy balance are checked on every run
  (`energy_residual_rel` in `tqtoy summary`).
