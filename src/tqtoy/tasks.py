"""Input files of the figure commands (``tqtoy plot`` and ``tqtoy map``).

Every command reads a YAML input file. ``tqtoy init -t plot`` and ``tqtoy init -t map``
write commented examples. The entries are listed by ``tqtoy tasks`` and in
docs/INPUT_FILES.md. Command-line options override the entries of the file.

The appearance is set by the ``style`` block (any field of tqtoy.plotting.PlotStyle,
see ``tqtoy style``) or by a separate style file (``style_file``).
"""
# The input files of the other commands are documented in the same way:
#   run   -> tqtoy.config.Config      (tqtoy params)
#   fhte  -> tqtoy.fhte.FHTERunConfig (tqtoy tasks fhte)

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from .config import ConfigError, _build, _to_plain, read_yaml
from .plotting import DEFAULT_FIGURES, FIGURES, SERIES


def _t(default, help: str):
    return field(default=default, metadata={"help": help, "unit": ""})


def _tf(factory, help: str):
    return field(default_factory=factory, metadata={"help": help, "unit": ""})


@dataclass
class CompareSpec:
    """Time traces of several points of a scan on the same figure (1D comparison)."""

    variable: str = _t("Te_avg", "Quantity versus time (tqtoy plot --help-variables): "
                                 + ", ".join(SERIES))
    axis: Optional[str] = _t(None, "Scan axis to vary (null: the only axis of a 1D scan)")
    at: Dict[str, Any] = _tf(dict, "Values of the other scan axes, e.g. {n_D: 1.0e22}")
    points: Optional[List[float]] = _t(None, "Values of the varied axis to draw (null: all of them)")


@dataclass
class PlotTask:
    """Input file of ``tqtoy plot``: time traces of one point, or a comparison over a scan axis."""

    results: str = _t("results/tqtoy_results.h5", "Results file (.h5) of tqtoy run, or a standalone "
                                                  "FHTE file")
    at: Dict[str, Any] = _tf(dict, "Scan point to plot, by axis name: {n_D: 1.0e22, injected_atom_density: 5.0e19}. "
                                   "The closest point is used. Empty: the first point")
    figures: List[str] = _tf(lambda: list(DEFAULT_FIGURES),
                             "Figures to draw, or [all]: " + ", ".join(FIGURES))
    compare: Optional[CompareSpec] = _t(None, "Compare the points of a scan axis on one figure instead "
                                              "(see the compare section)")
    outdir: Optional[str] = _t("figures", "Directory where the figures are saved, created if it does "
                                          "not exist (null: not saved)")
    format: str = _t("png", "File format of the saved figures: png, pdf, svg...")
    show: bool = _t(True, "Open the figures in a window")
    style_file: Optional[str] = _t(None, "Plot style file (tqtoy style -o mystyle.yaml)")
    style: Dict[str, Any] = _tf(dict, "Plot style entries overriding style_file, e.g. {cmap: magma, time_unit: us}")


@dataclass
class MapTask:
    """Input file of ``tqtoy map``: a scalar quantity over a 1D or 2D scan."""

    results: str = _t("results/tqtoy_results.h5", "Results file (.h5) of tqtoy run")
    quantity: str = _t("t_TQ", "Quantity to map: t_TQ, Te_avg_final, t_radiative_collapse, radiated_fraction, "
                               "radiated_fraction_net, n_RE_hot_tail, n_RE_hot_tail_density")
    outdir: Optional[str] = _t("figures", "Directory where the figure is saved, created if it does "
                                          "not exist (null: not saved)")
    format: str = _t("png", "File format of the saved figure")
    show: bool = _t(True, "Open the figure in a window")
    style_file: Optional[str] = _t(None, "Plot style file (tqtoy style -o mystyle.yaml)")
    style: Dict[str, Any] = _tf(dict, "Plot style entries overriding style_file, e.g. {cmap: jet, vmin: -25, "
                                      "vmax: 0, contours: false, overlay: {quantity: radiated_fraction_net, "
                                      "levels: [0.9]}}")


def from_dict(cls, data: Optional[dict]):
    """Build a task from a mapping, with the same validation as the configurations."""
    return _build(cls, data or {}, "")


def from_yaml(cls, path):
    data = read_yaml(path)
    try:
        return from_dict(cls, data)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from None


def _section_class(klass, name: str, path: str):
    """Dataclass of a sub-section, or None when the field is a free mapping (dict)."""
    import typing
    tp = typing.get_type_hints(klass)[name]
    if getattr(tp, "__origin__", None) is typing.Union:
        args = [a for a in tp.__args__ if a is not type(None)]
        tp = args[0] if len(args) == 1 else tp
    if dataclasses.is_dataclass(tp):
        return tp
    if tp is Any or getattr(tp, "__origin__", None) is dict or tp is dict:
        return None
    raise ConfigError(f"'{path}' is not an entry of this input file ('{name}' is not a section).")


def _assign(klass, node: dict, keys: List[str], value, path: str) -> None:
    key = keys[0]
    if not key:
        raise ConfigError(f"'{path}' is not an entry of this input file (empty name).")
    if klass is not None and key not in {f.name for f in dataclasses.fields(klass)}:
        raise ConfigError(f"'{path}' is not an entry of this input file.")
    if len(keys) == 1:
        node[key] = _to_plain(value)
        return
    sub = _section_class(klass, key, path) if klass is not None else None
    current = node.get(key)
    if current is not None and not isinstance(current, dict):
        raise ConfigError(f"'{path}' is not an entry of this input file ('{key}' is not a section).")
    if current is None:
        node[key] = {}                              # optional section left empty, or a new mapping key
    _assign(sub, node[key], keys[1:], value, path)


def with_overrides(task, overrides: Dict[str, Any]):
    """Copy of a task with dotted-path entries replaced, e.g. {'compare.variable': 'ne_total'}."""
    data = _to_plain(dataclasses.asdict(task))
    for path, value in overrides.items():
        _assign(type(task), data, path.split("."), value, path)
    return from_dict(type(task), data)


def task_table(cls) -> List[Dict[str, str]]:
    """Entries of a task class: path, type, default, unit and description (for the documentation)."""
    rows: List[Dict[str, str]] = []

    def walk(klass, prefix, instance):
        import typing
        hints = typing.get_type_hints(klass)
        for f in dataclasses.fields(klass):
            value = getattr(instance, f.name)
            tp = hints[f.name]
            inner = tp
            optional = False
            if getattr(tp, "__origin__", None) is typing.Union:
                args = [a for a in tp.__args__ if a is not type(None)]
                optional = len(args) < len(tp.__args__)
                inner = args[0] if len(args) == 1 else tp
            name = getattr(inner, "__name__", str(inner)).replace("typing.", "")
            if optional and not dataclasses.is_dataclass(inner):
                name += " or null"
            help_text = f.metadata.get("help", "")
            unit = f.metadata.get("unit", "")
            if dataclasses.is_dataclass(inner):
                rows.append(dict(path=prefix + f.name, type="section or null" if optional else "section",
                                 default="null" if optional else "(defaults below)", unit="", help=help_text))
                walk(inner, prefix + f.name + ".", value if value is not None else inner())
                continue
            rows.append(dict(path=prefix + f.name, type=name, unit=unit,
                             default=yaml.safe_dump(_to_plain(value), default_flow_style=True).strip().
                             removesuffix("\n...").strip(),
                             help=help_text))

    walk(cls, "", cls())
    return rows


def task_class(name: str):
    """Class describing the input file of a command: 'run', 'plot', 'map' or 'fhte'."""
    if name == "plot":
        return PlotTask
    if name == "map":
        return MapTask
    if name == "fhte":
        from .fhte import FHTERunConfig
        return FHTERunConfig
    if name == "run":
        from .config import Config
        return Config
    raise ConfigError(f"Unknown input file '{name}'. Available: {', '.join(TASK_NAMES)}.")


#: Input file of each command, and the template written by ``tqtoy init``.
TASK_NAMES = ("run", "plot", "map", "fhte")
TASK_HELP = {
    "run": ("tqtoy run", "Physical model, time grid, solver and scan (see also 'tqtoy params')."),
    "plot": ("tqtoy plot", "Time traces of one scan point, or a comparison of the points of a scan axis."),
    "map": ("tqtoy map", "A scalar quantity over a 1D or 2D scan."),
    "fhte": ("tqtoy fhte", "Hot-tail runaway estimate, on a results file or on a prescribed evolution."),
}


def tasks_text(name=None, markdown: bool = False) -> str:
    """Reference of the input-file entries, as plain text or as Markdown tables.

    ``name`` is one command name, a sequence of them, or None for all of them.
    """
    from .config import parameter_table
    names = TASK_NAMES if not name else ((name,) if isinstance(name, str) else tuple(name))
    for key in names:
        if key not in TASK_NAMES:
            task_class(key)                               # raises a ConfigError listing the names
    out: List[str] = []
    for key in names:
        command, description = TASK_HELP[key]
        rows = parameter_table() if key == "run" else task_table(task_class(key))
        if markdown:
            out.append(f"## `{command}`\n\n{description}\n")
            out.append("| Entry | Type | Default | Unit | Description |\n|---|---|---|---|---|")
            out += [f"| `{r['path']}` | {r['type']} | `{r['default']}` | {r.get('unit', '')} | {r['help']} |"
                    for r in rows]
            out.append("")
        else:
            out.append(f"=== {command} ===   {description}")
            for r in rows:
                unit = f" [{r['unit']}]" if r.get("unit") else ""
                out.append(f"{r['path']}  ({r['type']}, default {r['default']}){unit}"
                           + (f"\n    {r['help']}" if r["help"] else ""))
            out.append("")
    return "\n".join(out) + "\n"
