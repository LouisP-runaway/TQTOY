"""Command-line interface: ``tqtoy <command> ...`` (see ``tqtoy --help``)."""

from __future__ import annotations

import argparse
import os
import sys
from importlib import resources
from pathlib import Path

import numpy as np

from . import __version__
from .config import Config, ConfigError, parse_assignment

TEMPLATES = ("single_run", "iter_scan", "plot", "map", "fhte_results", "fhte_prescribed")
TEMPLATE_COMMAND = {"single_run": "run", "iter_scan": "run", "plot": "plot", "map": "map",
                    "fhte_results": "fhte", "fhte_prescribed": "fhte"}
STYLE_FILE = "style_input.yaml"          # written next to a plot or map input file by tqtoy init

SUMMARY_UNITS = {"W_th_initial": "J/m^3", "W_th_final": "J/m^3", "E_ohmic": "J/m^3", "E_radiated": "J/m^3",
                 "E_stochastic": "J/m^3", "Te_avg_final": "eV", "n_RE_hot_tail_density": "m^-3",
                 "p_c_last_runaway": "me c", "p_limit_last_runaway": "me c",
                 "T0_fhte": "eV", "n0_fhte": "m^-3"}


def _print_summary(values):
    for key, value in values.items():
        unit = SUMMARY_UNITS.get(key, "s" if key.startswith("t_") else "")
        unit = f" {unit}" if unit else ""
        print(f"  {key:28s} {value:.4g}{unit}" if isinstance(value, float) else f"  {key:28s} {value}")


def _template_text(name: str) -> str:
    return resources.files("tqtoy.templates").joinpath(f"{name}.yaml").read_text(encoding="utf-8")


def _parse_set(items):
    return dict(parse_assignment(s) for s in (items or []))


def _select(results, at_items):
    idx = results.index(_parse_set(at_items))
    return idx, results.run(idx)


def _add_style_args(s, kind: str):
    """Plot-style options. kind: 'plot', 'map' or 'rates'."""
    g = s.add_argument_group("appearance (see 'tqtoy style' for all options)")
    g.add_argument("--style", metavar="FILE", help="YAML plot-style file (written by 'tqtoy style -o FILE')")
    g.add_argument("--opt", nargs="+", metavar="KEY=VALUE", help="any style option, e.g. legend_loc='upper left'")
    g.add_argument("--cmap", help="colormap (jet, viridis, plasma, inferno, turbo, RdBu_r...)")
    g.add_argument("--color", nargs="+", metavar="KEY=COLOR", help="curve colours, e.g. hot=black cold=C0")
    g.add_argument("--title", help="figure title ('' for none)")
    g.add_argument("--figsize", nargs=2, type=float, metavar=("W", "H"), help="figure size in inches")
    g.add_argument("--dpi", type=int, help="resolution of the saved figures")
    g.add_argument("--fontsize", type=float)
    g.add_argument("--mpl-style", help="matplotlib style sheet, e.g. seaborn-v0_8-paper")
    g.add_argument("--no-legend", action="store_true")
    g.add_argument("--no-grid", action="store_true")
    if kind in ("plot", "map"):
        g.add_argument("--tq-threshold", type=float, metavar="EV", help="temperature defining t_TQ (default 100 eV)")
    if kind == "plot":
        g.add_argument("--time-unit", choices=["s", "ms", "us"])
        g.add_argument("--time-axis", choices=["log", "linear"])
        g.add_argument("--xlim", nargs=2, type=float, metavar=("MIN", "MAX"), help="time limits (in time unit)")
        g.add_argument("--ylim", nargs=2, type=float, metavar=("MIN", "MAX"))
        g.add_argument("--yscale", choices=["log", "linear", "symlog"])
        g.add_argument("--no-injection-band", action="store_true")
        g.add_argument("--markers", action="store_true", help="show the output times as markers")
    if kind == "map":
        g.add_argument("--contours", action=argparse.BooleanOptionalAction, default=None,
                       help="draw isocontour lines (default: no)")
        g.add_argument("--levels", nargs="+", type=float, metavar="L",
                       help="one number: count of isocontours, several: their values (displayed units)")
        g.add_argument("--contour-color")
        g.add_argument("--no-contour-labels", action="store_true")
        g.add_argument("--filled", action="store_true", help="filled contours with discrete colour levels")
        g.add_argument("--filled-levels", type=int, metavar="N")
        g.add_argument("--vmin", type=float, help="colour-scale minimum (displayed units, log10 unless --linear)")
        g.add_argument("--vmax", type=float)
        g.add_argument("--xscale", choices=["auto", "log", "linear"])
        g.add_argument("--yscale", choices=["auto", "log", "linear"])
        g.add_argument("--overlay", nargs="+", metavar="ARG",
                       help="isocontours of a second quantity: QUANTITY LEVEL [LEVEL...], "
                            "e.g. --overlay radiated_fraction_net 0.9")
        g.add_argument("--ratio-lines", nargs="+", type=float, metavar="K", help="draw the lines y = k x")


def _style_overrides(args):
    """Plot-style entries given on the command line (they override those of the input file)."""
    over = dict(_parse_set(getattr(args, "opt", None)))
    for item in getattr(args, "color", None) or []:
        key, value = item.split("=", 1) if "=" in item else (None, None)
        if key is None:
            raise ConfigError(f"--color {item}: expected KEY=COLOR.")
        over[f"colors.{key}"] = value
    simple = {"cmap": "cmap", "title": "title", "dpi": "dpi", "fontsize": "fontsize", "mpl_style": "mpl_style",
              "tq_threshold": "tq_threshold", "time_unit": "time_unit", "time_axis": "time_axis",
              "contour_color": "contour_color", "filled_levels": "filled_levels", "vmin": "vmin", "vmax": "vmax",
              "contours": "contours"}
    for attr, key in simple.items():
        value = getattr(args, attr, None)
        if value is not None:
            over[key] = value
    for attr in ("figsize", "xlim", "ylim"):
        if getattr(args, attr, None) is not None:
            over[attr] = list(getattr(args, attr))
    if getattr(args, "no_legend", False):
        over["legend"] = False
    if getattr(args, "no_grid", False):
        over["grid"] = False
    if getattr(args, "no_injection_band", False):
        over["injection_band"] = False
    if getattr(args, "markers", False):
        over["markers"] = True
    if getattr(args, "no_contour_labels", False):
        over["contour_labels"] = False
    if getattr(args, "filled", False):
        over["filled"] = True
    if getattr(args, "linear", False):
        over["log"] = False
    levels = getattr(args, "levels", None)
    if levels is not None:
        over["contour_levels"] = int(levels[0]) if len(levels) == 1 else list(levels)
    if args.command == "map":
        for attr, key in (("xscale", "map_xscale"), ("yscale", "map_yscale")):
            if getattr(args, attr, None) is not None:
                over[key] = getattr(args, attr)
        if args.overlay:
            try:
                levels = [float(v) for v in args.overlay[1:]] or [0.5]
            except ValueError:
                raise ConfigError("--overlay: expected QUANTITY LEVEL [LEVEL...].") from None
            over["overlay"] = {"quantity": args.overlay[0], "levels": levels}
        if args.ratio_lines:
            over["ratio_lines"] = list(args.ratio_lines)
    elif getattr(args, "yscale", None) is not None:
        over["yscale"] = args.yscale
    return over


def _style_of(task, args):
    """PlotStyle of a task: style_file, then the style section, then the command-line options.

    A relative ``style_file`` is looked for next to the input file first, then in the working
    directory, so that an input file and its style file travel together.
    """
    from .plotting import PlotStyle
    path = getattr(args, "style", None) or task.style_file
    path = _beside(path, getattr(args, "input", "") or ".")
    style = PlotStyle.from_yaml(path) if path else PlotStyle()
    over = {**task.style, **_style_overrides(args)}
    return style.with_overrides(over) if over else style


def _style_from_args(args):
    """PlotStyle of a command that has no input file (rates)."""
    from .plotting import PlotStyle
    style = PlotStyle.from_yaml(args.style) if getattr(args, "style", None) else PlotStyle()
    over = _style_overrides(args)
    return style.with_overrides(over) if over else style


def _setup_backend(show: bool):
    import matplotlib
    if not show:
        matplotlib.use("Agg")


def _is_input_file(path) -> bool:
    return Path(path).suffix.lower() in (".yaml", ".yml")


def _load_task(cls, args):
    """Input file of a figure command, then the command-line overrides.

    ``args.input`` is a YAML input file (``tqtoy init -t plot``) or, for convenience, a results
    file: the default entries are then used.
    """
    from .tasks import from_dict, from_yaml, with_overrides

    path = Path(args.input)
    task = from_yaml(cls, path) if _is_input_file(path) else from_dict(cls, {"results": str(path)})
    over = _parse_set(getattr(args, "set", None))
    for name in ("results", "quantity", "figures", "outdir", "format", "variable", "axis", "points"):
        value = getattr(args, name, None)
        if value is not None:
            over["compare." + name if name in ("variable", "axis", "points") else name] = value
    if getattr(args, "no_show", False):
        over["show"] = False
    task = with_overrides(task, over) if over else task
    # Scan axes are named by dotted paths, so --at entries are inserted as single keys.
    at = _parse_set(getattr(args, "at", None))
    if at:
        (task.compare.at if getattr(task, "compare", None) is not None else task.at).update(at)
    task.results = _beside(task.results, path)
    if not Path(task.results).exists():
        raise SystemExit(f"{task.results}: no such file (entry 'results' of the input file).")
    return task


def _beside(name, input_file):
    """A relative path is looked for next to the input file, then in the working directory."""
    if not name or Path(name).is_absolute() or Path(name).exists():
        return name
    candidate = Path(input_file).parent / name
    return str(candidate) if candidate.exists() else name


def _save_figures(figures: dict, task, style) -> None:
    if not task.outdir:
        return
    out = Path(task.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        fig.savefig(out / f"{name}.{task.format}", dpi=style.dpi)
    print(f"{len(figures)} figure(s) saved in {out}", file=sys.stderr)


def _show(task) -> None:
    if task.show:
        import matplotlib.pyplot as plt
        plt.show()


# --------------------------------------------------------------------------- commands

def cmd_init(args):
    from .plotting import PlotStyle
    path = Path(args.file or f"{args.template}_input.yaml")
    if path.exists() and not args.force:
        raise SystemExit(f"{path} exists (use --force to overwrite).")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_template_text(args.template), encoding="utf-8")
    command = TEMPLATE_COMMAND[args.template]
    print(f"Wrote {path}. Edit it, then run:  tqtoy {command} {path}")
    if args.template in ("plot", "map"):
        # the figure input files point to this file: write it too, so that every option is at hand
        style = path.parent / STYLE_FILE
        if style.exists():                      # never overwritten: it may have been edited
            print(f"{style} already there: it holds the appearance options (entry style_file).")
        else:
            style.write_text(PlotStyle().to_yaml(), encoding="utf-8")
            print(f"Wrote {style} (appearance options, entry style_file of {path.name}).")


def cmd_run(args):
    from .banner import print_banner
    from .io import save_results
    from .scan import run_scan

    if not args.quiet:
        print_banner("TQTOY", f"0D thermal quench toy model {__version__}")
    config = Config.from_yaml(args.config)
    overrides = _parse_set(args.set)
    if overrides:
        config = config.with_overrides(overrides)
    out = Path(args.output or config.output.file)
    shape = tuple(len(v) for _, v in config.scan_axes())
    n = int(np.prod(shape)) if shape else 1
    print(f"[tqtoy] {n} run(s), impurities: {', '.join(config.element_names)}, "
          f"{config.model.populations} population(s), source: {config.injection.source}", file=sys.stderr)
    results = run_scan(config, jobs=args.jobs, progress=not args.quiet)
    save_results(results, out)
    n_bad = int(np.sum(results.status < 0))
    print(f"[tqtoy] results written to {out}" + (f" ({n_bad} failed run(s), see 'tqtoy info')" if n_bad else ""),
          file=sys.stderr)


def cmd_info(args):
    from .io import load_results
    from .solver import STATUS_LABELS

    res = load_results(args.results)
    if args.config:
        print(res.config.to_yaml())
        return
    print(f"file      : {args.results}")
    print(f"created   : {res.metadata.get('created', '?')} with tqtoy {res.metadata.get('tqtoy_version', '?')}")
    print(f"elements  : {', '.join(res.elements)}")
    print(f"model     : {res.config.model.populations} population(s), source {res.config.injection.source}")
    print(f"time grid : up to {res.time.shape[-1]} points, {np.nanmin(res.time):.3g} s -> "
          f"{np.nanmax(res.time):.3g} s")
    if res.axes:
        print("scan axes :")
        for name, values in res.axes:
            v = np.asarray(values)
            print(f"  {name:40s} {len(v):4d} values in [{v.min():.3g}, {v.max():.3g}]")
    else:
        print("scan axes : none (single run)")
    status = np.atleast_1d(res.status)
    for code, label in STATUS_LABELS.items():
        count = int(np.sum(status == code))
        if count:
            print(f"  {label:28s}: {count}")
    if res.fhte is not None:
        extra = f" (post-hoc settings: {res.fhte_overrides})" if res.fhte_overrides else ""
        print(f"hot tail  : FHTE results stored{extra}")
    else:
        print("hot tail  : no FHTE results (compute them with: tqtoy fhte FILE)")
    msgs = [(idx, m) for idx, m in np.ndenumerate(np.atleast_1d(res.messages)) if m]
    for idx, m in msgs[:10]:
        print(f"  point {idx}: {m}")


def cmd_summary(args):
    from .diagnostics import summary
    from .fhte import file_format, load_prescribed, summary_fhte
    from .io import load_results

    if file_format(args.results) == "tqtoy-fhte-1":
        _, _, fres = load_prescribed(args.results)
        _print_summary(summary_fhte(fres))
        return
    res = load_results(args.results)
    idx, run = _select(res, args.at)
    if res.axes:
        print("point:", ", ".join(f"{k} = {_fmt(v)}" for k, v in res.coordinates(idx).items()))
    if args.fhte and run.fhte is None:
        from .fhte import hot_tail_from_run
        run.fhte = hot_tail_from_run(run)
    _print_summary(summary(run, tq_threshold=args.tq_threshold))


def _fmt(v):
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def cmd_plot(args):
    from .fhte import file_format
    from .io import load_results
    from .plotting import DEFAULT_FIGURES, FIGURES, SERIES, make_figures, plot_scan_series
    from .tasks import PlotTask

    if args.help_variables:
        print("Variables of the comparison figure (compare.variable):")
        for name, (label, _, _, _) in SERIES.items():
            print(f"  {name:16s} {label}")
        return
    if not args.input:
        raise SystemExit("give an input file (tqtoy init -t plot) or a results file (.h5).")
    task = _load_task(PlotTask, args)
    _setup_backend(task.show)
    style = _style_of(task, args)
    if file_format(task.results) == "tqtoy-fhte-1":
        _plot_prescribed(task, style)
        return
    res = load_results(task.results)
    if task.compare is not None:
        if args.figures:
            print("note: --figures is ignored, a comparison figure is drawn", file=sys.stderr)
        c = task.compare
        fig = plot_scan_series(res, c.variable, axis=c.axis, at=c.at, points=c.points, style=style)
        _save_figures({f"compare_{c.variable}": fig}, task, style)
        _show(task)
        return
    idx = res.index(task.at)
    run = res.run(idx)
    if task.figures == ["all"]:
        names = [n for n in FIGURES if not (n == "shards" and run.r_p_samples is None)]
    else:
        names = list(task.figures) or list(DEFAULT_FIGURES)
    if res.axes:
        print("point:", ", ".join(f"{k} = {_fmt(v)}" for k, v in res.coordinates(idx).items()), file=sys.stderr)
    prefix = "" if not res.axes else "point_" + "_".join(str(i) for i in idx) + "_"
    make_figures(run, names, outdir=task.outdir, fmt=task.format, prefix=prefix, style=style)
    if task.outdir:
        print(f"figures saved in {task.outdir}", file=sys.stderr)
    _show(task)


def _plot_prescribed(task, style):
    """Figures of a standalone FHTE result file."""
    from .fhte import load_prescribed
    from .plotting import plot_fhte_inputs, plot_hot_tail
    _, traces, fres = load_prescribed(task.results)
    _save_figures({"hot_tail": plot_hot_tail(fres, style=style),
                   "fhte_inputs": plot_fhte_inputs(traces, fres, style=style)}, task, style)
    _show(task)


def cmd_fhte(args):
    """Hot-tail estimate on a TQ results file (all points) or on a prescribed evolution."""
    from .banner import print_banner
    from .config import read_yaml
    from .fhte import FHTERunConfig, run_prescribed, save_prescribed, summary_fhte

    if not args.quiet:
        print_banner("TQTOY-FHTE", f"hot-tail runaway estimator {__version__}")
    path = Path(args.input)
    if _is_input_file(path):
        cfg = FHTERunConfig.from_yaml(path)
        written = (read_yaml(path) or {}).get("fhte") or {}
        settings = {f"fhte.{k}": v for k, v in written.items()}       # only the entries of the file
    else:
        cfg, settings = FHTERunConfig.from_dict({"results": str(path)}), {}
    over = _parse_set(args.set)
    settings.update({k: v for k, v in over.items() if k.startswith("fhte.")})
    if over or args.output or args.jobs is not None:
        import dataclasses

        from .config import _set_path
        data = dataclasses.asdict(cfg)
        for key, value in over.items():
            _set_path(data, key, value)
        if args.output:
            data["output"] = args.output
        if args.jobs is not None:
            data["jobs"] = args.jobs
        cfg = FHTERunConfig.from_dict(data)
    if not cfg.results:
        traces, fres = run_prescribed(cfg)
        out = save_prescribed(cfg.output or "results/fhte_results.h5", cfg, traces, fres)
        _print_summary(summary_fhte(fres))
        if fres.message:
            print(f"warning: {fres.message}", file=sys.stderr)
        print(f"[tqtoy] FHTE result written to {out}. Figures: tqtoy plot {out}", file=sys.stderr)
        return
    from .io import load_results, save_fhte, save_results
    from .scan import compute_fhte
    res = load_results(cfg.results)
    n = int(np.prod(res.shape)) if res.shape else 1
    print(f"[tqtoy] FHTE on {n} point(s) of {cfg.results}", file=sys.stderr)
    compute_fhte(res, settings, jobs=cfg.jobs, progress=not args.quiet)
    if cfg.output:
        save_results(res, cfg.output)
        out = cfg.output
    else:
        save_fhte(res, cfg.results)
        out = cfg.results
    frac = res.fhte["fraction"]
    last = frac[..., -1] if frac.ndim else frac
    print(f"[tqtoy] hot-tail results stored in {out} (final n_RE/n0 from {np.nanmin(last):.3g} to "
          f"{np.nanmax(last):.3g}). Map: tqtoy map {out} -q n_RE_hot_tail", file=sys.stderr)


def cmd_map(args):
    from .io import load_results
    from .plotting import plot_scan_map
    from .tasks import MapTask

    task = _load_task(MapTask, args)
    _setup_backend(task.show)
    style = _style_of(task, args)
    res = load_results(task.results)
    if not res.axes:
        raise SystemExit("This file contains a single run: nothing to map "
                         "(a scan is needed, see the scan section of the run input file).")
    fig = plot_scan_map(res, task.quantity, style=style)
    _save_figures({f"map_{task.quantity}": fig}, task, style)
    _show(task)


def cmd_rates(args):
    _setup_backend(not args.no_show)
    from .plotting import plot_radiation_rates

    style = _style_from_args(args)
    fig = plot_radiation_rates(args.element, ne=args.ne, style=style)
    if args.outdir:
        out = Path(args.outdir)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"rates_{args.element}.{args.format}", dpi=style.dpi)
    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()


def params_text(pattern: str = "", markdown: bool = False) -> str:
    """Parameter reference as plain text or as a Markdown table."""
    from .config import parameter_table
    rows = [r for r in parameter_table() if pattern.lower() in (r["path"] + " " + r["help"]).lower()]
    if markdown:
        lines = ["| Parameter | Type | Default | Unit | Description |", "|---|---|---|---|---|"]
        lines += [f"| `{r['path']}` | {r['type']} | `{r['default']}` | {r['unit']} | {r['help']} |" for r in rows]
        return "\n".join(lines) + "\n"
    lines = []
    for r in rows:
        unit = f" [{r['unit']}]" if r["unit"] else ""
        lines.append(f"{r['path']}  ({r['type']}, default {r['default']}){unit}\n    {r['help']}")
    return "\n".join(lines) + "\n" if lines else f"No parameter matches '{pattern}'.\n"


def cmd_params(args):
    print(params_text(args.pattern or "", args.markdown), end="")


def cmd_tasks(args):
    from .tasks import tasks_text
    print(tasks_text(args.name, args.markdown), end="")


def style_markdown() -> str:
    from .plotting import style_parameter_table
    lines = ["| Option | Default | Description |", "|---|---|---|"]
    lines += [f"| `{r['name']}` | `{r['default']}` | {r['help']} |" for r in style_parameter_table()]
    return "\n".join(lines) + "\n"


def cmd_style(args):
    from .plotting import PlotStyle
    if args.markdown:
        print(style_markdown(), end="")
        return
    text = PlotStyle().to_yaml()
    if args.output:
        path = Path(args.output)
        if path.exists() and not args.force:
            raise SystemExit(f"{path} exists (use --force to overwrite).")
        path.write_text(text, encoding="utf-8")
        print(f"Wrote {path}. Use it with the entry  style_file: {path}  of a plot or map input "
          f"file, or with:  tqtoy plot INPUT --style {path}")
    else:
        print(text, end="")


def cmd_install_adas(args):
    from .atomic import install_openadas
    install_openadas()


# --------------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tqtoy", description="0D multi-fluid toy model of the tokamak thermal quench.")
    p.add_argument("--version", action="version", version=f"tqtoy {__version__}")
    sub = p.add_subparsers(dest="command", required=True, metavar="command")

    s = sub.add_parser("init", help="write a commented input file (run, plot, map or fhte)")
    s.add_argument("file", nargs="?", help="output file (default: <template>_input.yaml)")
    s.add_argument("-t", "--template", choices=TEMPLATES, default="single_run",
                   help="single_run or iter_scan (tqtoy run), plot, map, fhte_results or fhte_prescribed")
    s.add_argument("--force", action="store_true", help="overwrite an existing file")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("run", help="run an input file (single run or scan)")
    s.add_argument("config", help="YAML input file (tqtoy init -t single_run)")
    s.add_argument("-o", "--output", help="HDF5 output file (default: output.file of the configuration)")
    s.add_argument("-j", "--jobs", type=int, default=1, help="number of parallel processes")
    s.add_argument("--set", nargs="+", metavar="KEY=VALUE", help="override parameters, e.g. plasma.Te0=10e3")
    s.add_argument("-q", "--quiet", action="store_true", help="no progress output")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("info", help="describe a results file")
    s.add_argument("results")
    s.add_argument("--config", action="store_true", help="print the stored configuration (YAML)")
    s.set_defaults(func=cmd_info)

    at_help = "select the scan point closest to these values, e.g. n_D=1e22 injected_atom_density=5e19"

    s = sub.add_parser("summary", help="characteristic times and energies of one run")
    s.add_argument("results")
    s.add_argument("--at", nargs="+", metavar="AXIS=VALUE", help=at_help)
    s.add_argument("--tq-threshold", type=float, default=100.0, metavar="EV",
                   help="temperature defining t_TQ (default 100 eV)")
    s.add_argument("--fhte", action="store_true", help="run the hot-tail estimate if it is not stored")
    s.set_defaults(func=cmd_summary)

    input_help = ("YAML input file (tqtoy init -t %s), or directly a results file (.h5) "
                  "to use the default entries")

    s = sub.add_parser("plot", help="time traces of one scan point, or a comparison over a scan axis")
    s.add_argument("input", nargs="?", help=input_help % "plot")
    s.add_argument("-r", "--results", help="results file, overriding the entry of the input file")
    s.add_argument("--set", nargs="+", metavar="KEY=VALUE",
                   help="override any entry of the input file, e.g. compare.variable=ne_total style.cmap=magma")
    s.add_argument("--at", nargs="+", metavar="AXIS=VALUE", help=at_help)
    s.add_argument("-f", "--figures", nargs="+", metavar="NAME",
                   help="figures to draw, or 'all' (default: temperatures densities charge_states powers energy)."
                        " Available: temperatures, densities, charge_states, zeff, powers, energy, currents,"
                        " efield, exchange, sources, shards, hot_tail")
    g = s.add_argument_group("comparison of scan points (1D)")
    g.add_argument("-c", "--variable", metavar="NAME",
                   help="compare the points of a scan axis on one figure: Te_avg, ne_total, n_imp, Zeff, P_rad, "
                        "E_parallel, W_th, n_RE_hot_tail... ('tqtoy plot --help-variables' lists them)")
    g.add_argument("--axis", metavar="NAME", help="scan axis to vary (default: the only axis of a 1D scan)")
    g.add_argument("--points", nargs="+", type=float, metavar="V", help="values of the axis to draw (default: all)")
    s.add_argument("-d", "--outdir", help="save the figures in this directory")
    s.add_argument("--format", help="figure format (png, pdf, svg...)")
    s.add_argument("--no-show", action="store_true", help="do not open windows")
    s.add_argument("--help-variables", action="store_true", help="list the comparison variables and exit")
    _add_style_args(s, "plot")
    s.set_defaults(func=cmd_plot)

    s = sub.add_parser("map", help="a scalar quantity over a 1D or 2D scan")
    s.add_argument("input", help=input_help % "map")
    s.add_argument("-r", "--results", help="results file, overriding the entry of the input file")
    s.add_argument("--set", nargs="+", metavar="KEY=VALUE",
                   help="override any entry of the input file, e.g. quantity=n_RE_hot_tail style.cmap=jet")
    s.add_argument("-q", "--quantity",
                   help="t_TQ (default), Te_avg_final, t_radiative_collapse, radiated_fraction, "
                        "radiated_fraction_net, n_RE_hot_tail, n_RE_hot_tail_density")
    s.add_argument("--linear", action="store_true", help="linear colour scale (default: log10)")
    s.add_argument("-d", "--outdir", help="save the figure in this directory")
    s.add_argument("--format")
    s.add_argument("--no-show", action="store_true")
    _add_style_args(s, "map")
    s.set_defaults(func=cmd_map)

    s = sub.add_parser("rates", help="plot the ADAS radiated power coefficients of an element")
    s.add_argument("element", help="element name or symbol, e.g. neon or Ne")
    s.add_argument("--ne", type=float, default=1e20, help="electron density [m^-3]")
    s.add_argument("-d", "--outdir")
    s.add_argument("--format", default="png")
    s.add_argument("--no-show", action="store_true")
    _add_style_args(s, "rates")
    s.set_defaults(func=cmd_rates)

    s = sub.add_parser("fhte", help="hot-tail runaway estimate (FHTE) on a results file or a prescribed evolution")
    s.add_argument("input", help="YAML input file (tqtoy init -t fhte_results | fhte_prescribed), or directly a "
                                 "results file (.h5): the estimate then runs on all its points with the stored "
                                 "settings")
    s.add_argument("-o", "--output", help="output file (.h5). Results file: write a copy instead of updating it")
    s.add_argument("-j", "--jobs", type=int, help="number of parallel processes (results file)")
    s.add_argument("--set", nargs="+", metavar="KEY=VALUE",
                   help="override entries, e.g. fhte.n_output=20 fhte.field=model results=scan.h5")
    s.add_argument("-q", "--quiet", action="store_true", help="no progress output")
    s.set_defaults(func=cmd_fhte)

    s = sub.add_parser("tasks", help="list the entries of the input files (run, plot, map, fhte)")
    s.add_argument("name", nargs="?", help="only this input file: run, plot, map or fhte")
    s.add_argument("--markdown", action="store_true", help="print Markdown tables")
    s.set_defaults(func=cmd_tasks)

    s = sub.add_parser("style", help="print (or write with -o) the default plot style with its options")
    s.add_argument("-o", "--output", metavar="FILE", nargs="?", const=STYLE_FILE,
                   help=f"write the style to this YAML file (default name: {STYLE_FILE})")
    s.add_argument("--force", action="store_true", help="overwrite an existing file")
    s.add_argument("--markdown", action="store_true", help="print the options as a Markdown table")
    s.set_defaults(func=cmd_style)

    s = sub.add_parser("params", help="list the configuration parameters (type, default, unit, description)")
    s.add_argument("pattern", nargs="?", help="only parameters whose path or description contains this text")
    s.add_argument("--markdown", action="store_true", help="print a Markdown table")
    s.set_defaults(func=cmd_params)

    s = sub.add_parser("install-adas", help="download the OpenADAS data used by CHERAB (once)")
    s.set_defaults(func=cmd_install_adas)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (ConfigError, KeyError, ValueError, FileNotFoundError) as exc:
        msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        raise SystemExit(f"error: {msg}") from None
    except BrokenPipeError:
        # The output was piped into a command that closed it early, such as 'head'.
        # Redirect the buffered stdout to /dev/null so that its flush at interpreter
        # exit does not raise a second time, and report the standard shell status.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        raise SystemExit(141) from None


if __name__ == "__main__":  # pragma: no cover
    main()
