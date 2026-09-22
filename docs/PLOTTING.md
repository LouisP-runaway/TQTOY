# Plot style

The appearance options of `tqtoy plot`, `tqtoy map` and `tqtoy rates`. The table below is also
printed by `tqtoy style --markdown`. The options are applied in this order, each level
overriding the previous one:

1. the default values below,
2. a style file: `tqtoy style -o mystyle.yaml` writes the defaults with comments, edit it and name
   it in the `style_file` entry of the plot or map input file (or use `--style mystyle.yaml`),
3. the `style` block of the plot or map input file (see [INPUT_FILES.md](INPUT_FILES.md)),
4. `--set style.KEY=VALUE`, or `--opt KEY=VALUE` for any option of the table, e.g.
   `--opt legend_loc="upper left" figsize=[6,4]`,
5. the direct options (`--cmap`, `--color`, `--levels`, `--contours`... see `tqtoy map --help`).

```yaml
# in map_input.yaml
style:
  cmap: seismic
  filled: true
  filled_levels: 16
  contours: true
```

In Python, pass `style=PlotStyle(...)` or a dict of options to `make_figures`, `plot_scan_map` and
the `plot_*` functions.

Examples:

```bash
tqtoy map scan.h5 --cmap magma --contours
tqtoy map scan.h5 --cmap seismic --filled --filled-levels 16
tqtoy map scan.h5 --levels 0.5 1 2 3 --contour-color black --vmin 0 --vmax 4
tqtoy plot scan.h5 --at n_D=1e22 --time-unit us --time-axis linear --xlim 0 50 --color cold=black
tqtoy plot scan.h5 --mpl-style seaborn-v0_8-paper --fontsize 14 --format pdf -d figures
```

| Option | Default | Description |
|---|---|---|
| `figsize` | `[8.0, 5.5]` | Figure size [width, height] in inches |
| `dpi` | `150` | Resolution of the saved figures |
| `fontsize` | `null` | Font size (null: matplotlib default) |
| `linewidth` | `null` | Line width (null: matplotlib default) |
| `mpl_style` | `null` | Matplotlib style sheet name or file, e.g. seaborn-v0_8-paper |
| `title` | `null` | Figure title (null: automatic, empty string: none) |
| `xlabel` | `null` | x-axis label (null: automatic) |
| `ylabel` | `null` | y-axis label (null: automatic) |
| `legend` | `true` | Show the legends |
| `legend_loc` | `best` | Legend position (matplotlib names: best, upper right...) |
| `legend_fontsize` | `null` | Legend font size (null: automatic) |
| `grid` | `true` | Show the grid |
| `colors` | `{}` | Curve colours overriding the defaults, e.g. {hot: black, cold: C0}. Keys: hot, cold, average, total, radiated, ohmic, stochastic, residual, t_TQ, collapse, balance, injection, D, impurity, samples, parks, line |
| `cmap` | `jet` | Colormap of the maps and of the charge-state curves |
| `time_unit` | `s` | Time unit of the time axis: s, ms or us |
| `time_axis` | `log` | Scale of the time axis: log or linear |
| `xlim` | `null` | Time-axis limits [min, max] in time_unit (null: full run) |
| `ylim` | `null` | y-axis limits [min, max] (null: automatic) |
| `yscale` | `null` | y-axis scale: log, linear or symlog (null: figure default) |
| `injection_band` | `true` | Shade the injection window |
| `markers` | `false` | Show the output times as markers |
| `tq_threshold` | `100.0` | Temperature defining t_TQ (first time with <Te> below) [eV] |
| `log` | `true` | Colour scale in log10 of the quantity |
| `vmin` | `null` | Lower limit of the colour scale (1D scans: of the y axis), in displayed units: log10 of the quantity if log is true (null: data) |
| `vmax` | `null` | Upper limit of the colour scale, in displayed units (null: data) |
| `filled` | `false` | Filled contours with discrete colour levels instead of a pixel map |
| `filled_levels` | `16` | Number of colour levels when filled is true |
| `contours` | `false` | Draw isocontour lines (2D scans) |
| `contour_levels` | `10` | Number of isocontours, or list of their values in displayed units |
| `contour_color` | `white` | Colour of the isocontours |
| `contour_linewidth` | `0.6` | Line width of the isocontours |
| `contour_labels` | `true` | Write the value on the isocontours |
| `contour_label_format` | `'%.2g'` | Format of the isocontour labels |
| `bad_color` | `'0.8'` | Colour of undefined points (e.g. threshold never reached) |
| `map_xscale` | `auto` | x-axis scale of the maps: auto, log or linear |
| `map_yscale` | `auto` | y-axis scale of the maps: auto, log or linear |
| `colorbar_label` | `null` | Colour-bar label (null: automatic) |
| `marker` | `o` | Marker of 1D scans |
| `overlay` | `null` | Isocontours of a second quantity on 2D maps, e.g. {quantity: radiated_fraction_net, levels: [0.9], color: lime, linewidth: 2} |
| `ratio_lines` | `null` | Lines y = k x on 2D maps for each k of the list (e.g. impurity fractions n_Ne / n_D) |
| `ratio_line_color` | `magenta` | Colour of the ratio lines |
