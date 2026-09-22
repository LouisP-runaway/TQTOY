# Input files

The entries of the `tqtoy plot`, `tqtoy map` and `tqtoy fhte` input files, also printed by
`tqtoy tasks`. Those of `tqtoy run` are in [INPUT_PARAMETERS.md](INPUT_PARAMETERS.md).
[TUTORIAL.md](TUTORIAL.md) shows how the files are used.

`tqtoy init -t NAME` writes a commented example of each one. Indent with spaces, two per level.
Tabulations are expanded to spaces (YAML forbids them in the indentation) and a syntax error is
reported with its line and column.

## Overriding entries on the command line

Any entry can be changed for one call without editing the file, with `--set` and dotted paths:

```bash
tqtoy plot plot_input.yaml --set compare.variable=n_RE_hot_tail style.cmap=magma
tqtoy map map_input.yaml --set quantity=radiated_fraction style.contours=false
tqtoy run iter_scan_input.yaml --set plasma.Te0=10e3 'scan.injection.n_D=[1e21, 1e22]'
```

The most common entries also have their own option (`-q/--quantity`, `--at`, `-f/--figures`,
`-d/--outdir`, `--cmap`, `--no-show`...). The order is: input file, then `--set`, then the options.
A results file can also be given directly instead of an input file, which uses the default
entries: `tqtoy map results.h5 -q t_TQ`.

## Drawing a scan

`tqtoy map` draws a line versus the scanned parameter for a 1D scan, and a coloured map for a 2D
scan. `tqtoy plot` with a `compare` section overlays the time traces of the points of one axis,
coloured by the scanned value (`tqtoy plot --help-variables` lists the available quantities).
With a 2D scan, `compare.axis` chooses the axis that varies and `compare.at` fixes the other one.

## Plot style

The appearance of every figure is set by `style_input.yaml`, written next to the input file by
`tqtoy init -t plot | map` and read through the `style_file` entry, and by the `style` block of
the input file itself, which overrides it for that figure only. All the options are listed in
[PLOTTING.md](PLOTTING.md).

## `tqtoy plot`

Time traces of one scan point, or a comparison of the points of a scan axis.

| Entry | Type | Default | Unit | Description |
|---|---|---|---|---|
| `results` | str | `results/tqtoy_results.h5` |  | Results file (.h5) of tqtoy run, or a standalone FHTE file |
| `at` | Dict | `{}` |  | Scan point to plot, by axis name: {n_D: 1.0e22, injected_atom_density: 5.0e19}. The closest point is used. Empty: the first point |
| `figures` | List | `[temperatures, densities, charge_states, powers, energy]` |  | Figures to draw, or [all]: temperatures, densities, charge_states, zeff, mean_charge, powers, energy, currents, efield, exchange, timescales, cooling_curve, electron_budget, sources, shards, hot_tail |
| `compare` | section or null | `null` |  | Compare the points of a scan axis on one figure instead (see the compare section) |
| `compare.variable` | str | `Te_avg` |  | Quantity versus time (tqtoy plot --help-variables): Te_avg, Te_cold, Te_hot, Ti_cold, ne_total, ne_cold, ni_cold, n_imp, Zeff, P_rad, P_ohm, P_stoch, E_parallel, W_th, n_RE_hot_tail, P_ehot_ecold, P_ihot_icold, P_ehot_ihot, P_ehot_icold, P_ecold_icold, P_ecold_ihot |
| `compare.axis` | str or null | `null` |  | Scan axis to vary (null: the only axis of a 1D scan) |
| `compare.at` | Dict | `{}` |  | Values of the other scan axes, e.g. {n_D: 1.0e22} |
| `compare.points` | List or null | `null` |  | Values of the varied axis to draw (null: all of them) |
| `outdir` | str or null | `figures` |  | Directory where the figures are saved, created if it does not exist (null: not saved) |
| `format` | str | `png` |  | File format of the saved figures: png, pdf, svg... |
| `show` | bool | `true` |  | Open the figures in a window |
| `style_file` | str or null | `null` |  | Plot style file (tqtoy style -o mystyle.yaml) |
| `style` | Dict | `{}` |  | Plot style entries overriding style_file, e.g. {cmap: magma, time_unit: us} |

## `tqtoy map`

A scalar quantity over a 1D or 2D scan.

| Entry | Type | Default | Unit | Description |
|---|---|---|---|---|
| `results` | str | `results/tqtoy_results.h5` |  | Results file (.h5) of tqtoy run |
| `quantity` | str | `t_TQ` |  | Quantity to map: t_TQ, Te_avg_final, t_radiative_collapse, radiated_fraction, radiated_fraction_net, n_RE_hot_tail, n_RE_hot_tail_density |
| `outdir` | str or null | `figures` |  | Directory where the figure is saved, created if it does not exist (null: not saved) |
| `format` | str | `png` |  | File format of the saved figure |
| `show` | bool | `true` |  | Open the figure in a window |
| `style_file` | str or null | `null` |  | Plot style file (tqtoy style -o mystyle.yaml) |
| `style` | Dict | `{}` |  | Plot style entries overriding style_file, e.g. {cmap: jet, vmin: -25, vmax: 0, contours: false, overlay: {quantity: radiated_fraction_net, levels: [0.9]}} |

## `tqtoy fhte`

Hot-tail runaway estimate, on a results file or on a prescribed evolution.

| Entry | Type | Default | Unit | Description |
|---|---|---|---|---|
| `results` | str or null | `null` |  | Results file (.h5) of tqtoy run: the estimate runs on every scan point and is stored in the file. Null: run on the prescribed evolution below |
| `prescribed` | section | `(defaults below)` |  | Prescribed evolution of the bulk plasma (ignored when results is set) |
| `prescribed.t_start` | float | `1.0e-08` | s | First time |
| `prescribed.t_end` | float | `0.0012` | s | Last time |
| `prescribed.n_points` | int | `200` |  | Number of times |
| `prescribed.spacing` | str | `linear` |  | linear or log |
| `prescribed.file` | str or null | `null` |  | Optional CSV (header t,Te,...) or NPZ file. Its column t replaces the time grid |
| `prescribed.Te` | Any | `{exponential: {final: 31.0, initial: 3100.0, tau: 0.0003}}` | eV | Bulk electron temperature [eV]: number (constant), list (one value per time), {exponential: {initial, final, tau}}, {linear: {initial, final}} or {column: name} (column of prescribed.file) |
| `prescribed.ne` | Any | `2.8e+19` | m^-3 | Electron density [m^-3]: number (constant), list (one value per time), {exponential: {initial, final, tau}}, {linear: {initial, final}} or {column: name} (column of prescribed.file) |
| `prescribed.J` | Any | `1500000.0` | A m^-2 | Current density [A m^-2]: number (constant), list (one value per time), {exponential: {initial, final, tau}}, {linear: {initial, final}} or {column: name} (column of prescribed.file) |
| `prescribed.E` | Any | `null` | V m^-1 | Parallel field [V m^-1] (null: Spitzer field from J): number (constant), list (one value per time), {exponential: {initial, final, tau}}, {linear: {initial, final}} or {column: name} (column of prescribed.file) |
| `fhte` | section | `(defaults below)` |  | Settings of the estimator. The defaults below apply to a prescribed evolution. With a results file, only the entries actually written in the input file override the fhte section stored in it, so a scanned FHTE parameter keeps its value |
| `fhte.enabled` | bool | `false` |  | Run the FHTE on every point at the end of tqtoy run (also: tqtoy fhte results.h5) |
| `fhte.n_output` | int | `20` |  | Number of evaluation times from the first to the last output time (the last one gives the final runaway fraction). With output_window = active and only two times, the second one is the end of the runaway window rather than the last time of the run |
| `fhte.output_window` | str | `active` |  | Where the evaluation times are placed. active: inside the time windows where E > E_c, the only ones where runaways are created (no backward integration outside, the runaway population is kept constant there). full: over the whole run, with a backward integration at every time |
| `fhte.output_spacing` | str | `linear` |  | Spacing of the evaluation times: linear or log |
| `fhte.accumulate` | bool | `true` |  | Keep the runaway population once it is created: the reported fraction is the running maximum over the evaluation times. The estimator gives, at each time, the fraction of the initial distribution that is above the critical momentum at that time; this fraction falls when E/E_c decreases, although the electrons that already ran away are not lost. false: report the instantaneous fraction at each evaluation time |
| `fhte.n_trajectory` | int | `50` |  | Stored points per backward momentum trajectory |
| `fhte.trajectory_spacing` | str | `linear` |  | Spacing of the stored trajectory points: linear or log |
| `fhte.field` | str | `spitzer` |  | Parallel electric field: spitzer (from J with zeff and neoclassical_factor) or model (field of the TQ model, TQ runs only) |
| `fhte.zeff` | float | `1.0` |  | Effective charge in the Spitzer field of the FHTE |
| `fhte.neoclassical_factor` | float | `2.0` |  | Factor on the Spitzer resistivity of the FHTE (neoclassical correction) |
| `fhte.chandrasekhar` | bool | `false` |  | Multiply the collisional drag by the Chandrasekhar function |
| `fhte.temperature` | str | `average` |  | Bulk temperature from a TQ run: average (density weighted), cold or hot |
| `fhte.bound_electron_weight` | float | `0.0` |  | Weight of the electrons bound to impurities in the collisional density (0: free electrons only) |
| `fhte.T_floor` | float | `10.0` | eV | Floor on the bulk temperature |
| `fhte.n_floor` | float | `1000000000000000.0` | m^-3 | Floor on the density |
| `fhte.critical_iterations` | int | `4` |  | Fixed-point iterations on gamma_c for the critical momentum |
| `fhte.rtol` | float | `1.0e-05` |  | Relative tolerance of the backward momentum integration |
| `fhte.atol` | float | `1.0e-10` |  | Absolute tolerance of the backward momentum integration |
| `fhte.gamma_max` | float | `4.0` |  | Upper bound of the Maxwell-Juttner integration in gamma |
| `fhte.gamma_points` | int | `200000` |  | Number of points of the Maxwell-Juttner integration |
| `jobs` | int | `1` |  | Number of parallel processes (results file only) |
| `output` | str or null | `null` |  | Output file (.h5). Null: results/fhte_results.h5 for a prescribed evolution, or update the results file in place |

