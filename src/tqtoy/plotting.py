"""Standard figures. Each function returns a matplotlib Figure.

The appearance is controlled by a :class:`PlotStyle` (colours, colormap, isocontours,
axes, fonts...). It is set by the ``style`` block of the ``tqtoy plot`` and ``tqtoy map``
input files (see tqtoy.tasks), by a separate style file (entry ``style_file``, written by
``tqtoy style -o``) or by the command-line options.

The backend is never forced: use ``matplotlib.use(...)`` or the MPLBACKEND
environment variable if needed.
"""

from __future__ import annotations

import contextlib
import dataclasses
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import yaml
from scipy.integrate import trapezoid

from .config import ConfigError, parameter_unit, read_yaml
from .constants import E_CHARGE
from .diagnostics import (SCAN_QUANTITIES, T_TQ_THRESHOLD, compute_terms, energy_balance,
                          ohmic_radiative_balance_time, radiative_collapse_time, scan_map,
                          thermal_energy, thermal_quench_time)
from .physics.ablation import parks_kappa, parks_pdf
from .physics.impurities import coronal_cooling_rate
from .physics.resistivity import critical_field
from .solver import STATUS_LABELS, RunResult

# --------------------------------------------------------------------------- style

DEFAULT_COLORS = {
    "hot": "tab:red", "cold": "tab:blue", "average": "tab:purple", "total": "black",
    "radiated": "tab:green", "ohmic": "tab:red", "stochastic": "tab:pink", "residual": "0.5",
    "t_TQ": "black", "collapse": "tab:orange", "balance": "tab:purple",
    "injection": "0.85", "D": "tab:purple", "impurity": "tab:brown",
    "samples": "tab:red", "parks": "tab:blue", "line": "tab:blue",
}
TIME_UNITS = {"s": 1.0, "ms": 1e-3, "us": 1e-6}

#: Chemical symbols of the elements the model can use, for the figure labels.
ELEMENT_SYMBOLS = {
    "hydrogen": "H", "deuterium": "D", "tritium": "T", "helium": "He", "lithium": "Li",
    "beryllium": "Be", "boron": "B", "carbon": "C", "nitrogen": "N", "oxygen": "O",
    "fluorine": "F", "neon": "Ne", "aluminium": "Al", "aluminum": "Al", "silicon": "Si",
    "argon": "Ar", "iron": "Fe", "nickel": "Ni", "copper": "Cu", "krypton": "Kr",
    "molybdenum": "Mo", "xenon": "Xe", "tungsten": "W",
}


def _element_symbol(name: str) -> str:
    return ELEMENT_SYMBOLS.get(str(name).lower(), str(name).capitalize())


def _species_label(key: str) -> str:
    """Math label of a species of the exchange terms: 'ehot' -> 'e,hot' (inside $...$)."""
    for species, symbol in (("e", "e"), ("i", "i")):
        if key.startswith(species):
            rest = key[len(species):]
            return rf"{symbol},{{\rm {rest}}}" if rest else symbol
    return rf"{{\rm {key}}}"


#: Math labels of the parameters used as scan axes (see _axis_label).
AXIS_LABELS = {
    "plasma.Te0": r"$T_{e0}$", "plasma.Ti0": r"$T_{i0}$", "plasma.ne0": r"$n_{e0}$",
    "plasma.ni0": r"$n_{i0}$", "plasma.J": r"$J$", "plasma.B": r"$B$",
    "plasma.J_final": r"$J_{\rm final}$", "plasma.J_ramp_time": r"$t_{J}$",
    "plasma.cold_T0": r"$T_{0,\rm cold}$", "plasma.cold_n0": r"$n_{0,\rm cold}$",
    "injection.n_D": r"$n_D$", "injection.t_start": r"$t_{\rm inj}$",
    "injection.duration": r"$\Delta t_{\rm inj}$", "injection.charge_state": r"$q_{\rm inj}$",
    "injection.ablation.shard_radius": r"$r_s$", "injection.ablation.n_shards": r"$N_s$",
    "injection.ablation.volume": r"$V_{\rm abl}$", "injection.ablation.D2_fraction": r"$X$",
    "model.stochastic.deltaB_over_B": r"$\delta B / B$", "model.stochastic.r": r"$r$",
    "model.stochastic.a": r"$a$", "model.stochastic.coulomb_log": r"$\ln \Lambda$",
    "model.atomic_rate_multiplier": r"$r_{\rm ADAS}$", "model.linear_loss_coefficient": r"$c_{\rm loss}$",
    "model.main_ion_mass": r"$m_i / m_p$", "time.t_end": r"$t_{\rm end}$",
}


def _s(default, help: str):
    return field(default=default, metadata={"help": help})


def _sf(factory, help: str):
    return field(default_factory=factory, metadata={"help": help})


@dataclass
class PlotStyle:
    """Appearance of the figures. Every field has a default value."""

    # --- all figures
    figsize: List[float] = _sf(lambda: [8.0, 5.5], "Figure size [width, height] in inches")
    dpi: int = _s(150, "Resolution of the saved figures")
    fontsize: Optional[float] = _s(None, "Font size (null: matplotlib default)")
    linewidth: Optional[float] = _s(None, "Line width (null: matplotlib default)")
    mpl_style: Optional[str] = _s(None, "Matplotlib style sheet name or file, e.g. seaborn-v0_8-paper")
    title: Optional[str] = _s(None, "Figure title (null: automatic, empty string: none)")
    xlabel: Optional[str] = _s(None, "x-axis label (null: automatic)")
    ylabel: Optional[str] = _s(None, "y-axis label (null: automatic)")
    legend: bool = _s(True, "Show the legends")
    legend_loc: str = _s("best", "Legend position (matplotlib names: best, upper right...)")
    legend_fontsize: Optional[float] = _s(None, "Legend font size (null: automatic)")
    grid: bool = _s(True, "Show the grid")
    colors: Dict[str, str] = _sf(dict, "Curve colours overriding the defaults, e.g. {hot: black, cold: C0}. "
                                       f"Keys: {', '.join(DEFAULT_COLORS)}")
    cmap: str = _s("jet", "Colormap of the maps and of the charge-state curves")
    # --- time traces (tqtoy plot)
    time_unit: str = _s("s", "Time unit of the time axis: s, ms or us")
    time_axis: str = _s("log", "Scale of the time axis: log or linear")
    xlim: Optional[List[float]] = _s(None, "Time-axis limits [min, max] in time_unit (null: full run)")
    ylim: Optional[List[float]] = _s(None, "y-axis limits [min, max] (null: automatic)")
    yscale: Optional[str] = _s(None, "y-axis scale: log, linear or symlog (null: figure default)")
    injection_band: bool = _s(True, "Shade the injection window")
    markers: bool = _s(False, "Show the output times as markers")
    tq_threshold: float = _s(T_TQ_THRESHOLD, "Temperature defining t_TQ (first time with <Te> below) [eV]")
    # --- scan maps (tqtoy map)
    log: bool = _s(True, "Colour scale in log10 of the quantity")
    vmin: Optional[float] = _s(None, "Lower limit of the colour scale (1D scans: of the y axis), in displayed "
                                     "units: log10 of the quantity if log is true (null: data)")
    vmax: Optional[float] = _s(None, "Upper limit of the colour scale, in displayed units (null: data)")
    filled: bool = _s(False, "Filled contours with discrete colour levels instead of a pixel map")
    filled_levels: int = _s(16, "Number of colour levels when filled is true")
    contours: bool = _s(False, "Draw isocontour lines (2D scans)")
    contour_levels: Any = _s(10, "Number of isocontours, or list of their values in displayed units")
    contour_color: str = _s("white", "Colour of the isocontours")
    contour_linewidth: float = _s(0.6, "Line width of the isocontours")
    contour_labels: bool = _s(True, "Write the value on the isocontours")
    contour_label_format: str = _s("%.2g", "Format of the isocontour labels")
    bad_color: str = _s("0.8", "Colour of undefined points (e.g. threshold never reached)")
    map_xscale: str = _s("auto", "x-axis scale of the maps: auto, log or linear")
    map_yscale: str = _s("auto", "y-axis scale of the maps: auto, log or linear")
    colorbar_label: Optional[str] = _s(None, "Colour-bar label (null: automatic)")
    marker: str = _s("o", "Marker of 1D scans")
    overlay: Optional[Dict[str, Any]] = _s(None, "Isocontours of a second quantity on 2D maps, e.g. "
                                                 "{quantity: radiated_fraction_net, levels: [0.9], color: lime, "
                                                 "linewidth: 2}")
    ratio_lines: Optional[List[float]] = _s(None, "Lines y = k x on 2D maps for each k of the list (e.g. impurity "
                                                  "fractions n_Ne / n_D)")
    ratio_line_color: str = _s("magenta", "Colour of the ratio lines")

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "PlotStyle":
        data = dict(data or {})
        names = [f.name for f in dataclasses.fields(cls)]
        unknown = [k for k in data if k not in names]
        if unknown:
            raise ConfigError(f"plot style: unknown key(s) {unknown}. Valid keys: {names}.")
        style = cls(**data)
        style.validate()
        return style

    @classmethod
    def from_yaml(cls, path) -> "PlotStyle":
        return cls.from_dict(read_yaml(path))

    def with_overrides(self, overrides: Dict[str, Any]) -> "PlotStyle":
        data = dataclasses.asdict(self)
        for key, value in overrides.items():
            if key.startswith("colors."):
                data["colors"][key.split(".", 1)[1]] = value
            elif key == "colors" and isinstance(value, dict):
                data["colors"].update(value)           # merge, so an input file sets only the colours it changes
            else:
                if key not in data:
                    raise ConfigError(f"plot style: unknown key '{key}'. Valid keys: {list(data)}.")
                data[key] = value
        return PlotStyle.from_dict(data)

    #: Fields that hold a list or a mapping; every other one holds a single value.
    CONTAINER_FIELDS = ("colors", "overlay", "figsize", "xlim", "ylim", "contour_levels", "ratio_lines")

    def validate(self) -> None:
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if f.name not in self.CONTAINER_FIELDS and isinstance(value, (dict, list, tuple)):
                raise ConfigError(f"plot style: {f.name} must be a single value, got {value!r}.")
        checks = {"time_unit": (self.time_unit, tuple(TIME_UNITS)), "time_axis": (self.time_axis, ("log", "linear")),
                  "map_xscale": (self.map_xscale, ("auto", "log", "linear")),
                  "map_yscale": (self.map_yscale, ("auto", "log", "linear"))}
        for name, (value, allowed) in checks.items():
            if value not in allowed:
                raise ConfigError(f"plot style: {name} = '{value}' is invalid (allowed: {', '.join(allowed)}).")
        if self.yscale not in (None, "log", "linear", "symlog"):
            raise ConfigError("plot style: yscale must be log, linear, symlog or null.")
        for name in ("xlim", "ylim"):
            v = getattr(self, name)
            if v is not None and len(v) != 2:
                raise ConfigError(f"plot style: {name} must be [min, max].")
        if len(self.figsize) != 2:
            raise ConfigError("plot style: figsize must be [width, height].")
        unknown = [k for k in self.colors if k not in DEFAULT_COLORS]
        if unknown:
            raise ConfigError(f"plot style: unknown colour key(s) {unknown}. Keys: {list(DEFAULT_COLORS)}.")
        import matplotlib
        from matplotlib.colors import is_color_like
        # YAML reads grey levels such as 0.8 as numbers: matplotlib expects strings
        self.colors = {k: str(v) for k, v in self.colors.items()}
        self.contour_color, self.bad_color = str(self.contour_color), str(self.bad_color)
        self.ratio_line_color = str(self.ratio_line_color)
        if self.overlay is not None:
            unknown = set(self.overlay) - {"quantity", "levels", "color", "linewidth", "labels"}
            if "quantity" not in self.overlay or unknown:
                raise ConfigError("plot style: overlay needs 'quantity' and accepts levels, color, linewidth, labels.")
        for name, value in [(f"colors.{k}", v) for k, v in self.colors.items()] + \
                [("contour_color", self.contour_color), ("bad_color", self.bad_color)]:
            if not is_color_like(value):
                raise ConfigError(f"plot style: {name} = '{value}' is not a matplotlib colour.")
        if self.cmap not in matplotlib.colormaps:
            raise ConfigError(f"plot style: unknown colormap '{self.cmap}'. Examples: jet, viridis, plasma, "
                              "inferno, magma, cividis, turbo, seismic, RdBu_r (see matplotlib colormaps).")

    def to_yaml(self) -> str:
        """Default-style YAML with one comment per field."""
        lines = ["# tqtoy plot style. Use it with the entry  style_file: this_file.yaml  of a plot or map",
                 "# input file, or with  tqtoy plot|map INPUT --style this_file.yaml", ""]
        for f in dataclasses.fields(self):
            value = yaml.safe_dump({f.name: getattr(self, f.name)}, default_flow_style=True).strip()[1:-1]
            lines.append(f"{value:40s} # {f.metadata['help']}")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------ helpers
    def color(self, key: str) -> str:
        return self.colors.get(key, DEFAULT_COLORS[key])

    @property
    def time_factor(self) -> float:
        return TIME_UNITS[self.time_unit]

    @contextlib.contextmanager
    def context(self):
        """Matplotlib rc settings of the style (style sheet, font size, line width)."""
        import matplotlib.pyplot as plt
        rc = {}
        if self.fontsize is not None:
            rc["font.size"] = self.fontsize
        if self.linewidth is not None:
            rc["lines.linewidth"] = self.linewidth
        with contextlib.ExitStack() as stack:
            if self.mpl_style:
                stack.enter_context(plt.style.context(self.mpl_style))
            stack.enter_context(plt.rc_context(rc))
            yield


def style_parameter_table() -> List[Dict[str, str]]:
    """Fields of PlotStyle with their default value and description."""
    s = PlotStyle()

    def fmt(v):
        return yaml.safe_dump(v, default_flow_style=True).strip().removesuffix("\n...").strip()

    return [dict(name=f.name, default=fmt(getattr(s, f.name)), help=f.metadata["help"])
            for f in dataclasses.fields(PlotStyle)]


def _style(style) -> PlotStyle:
    if style is None:
        return PlotStyle()
    if isinstance(style, dict):
        return PlotStyle.from_dict(style)
    return style


# --------------------------------------------------------------------------- common layout

def _plt():
    import matplotlib.pyplot as plt
    return plt


def _new_axes(st: PlotStyle, title: str = ""):
    plt = _plt()
    fig, ax = plt.subplots(figsize=tuple(st.figsize), layout="constrained")
    _title(ax, st, title)
    return fig, ax


def _title(ax, st: PlotStyle, default: str):
    title = default if st.title is None else st.title
    if title:
        ax.set_title(title)


def _legend(ax, st: PlotStyle, **kwargs):
    if st.legend and ax.get_legend_handles_labels()[0]:
        kwargs.setdefault("fontsize", st.legend_fontsize)
        ax.legend(loc=st.legend_loc, **kwargs)


class _TimeAxis:
    """Conversion of times to the display unit."""

    def __init__(self, st: PlotStyle):
        self.f = st.time_factor
        self.unit = {"us": r"$\mu$s"}.get(st.time_unit, st.time_unit)

    def __call__(self, t):
        return np.asarray(t) / self.f


def _plot(ax, st: PlotStyle, x, y, **kwargs):
    if st.markers:
        kwargs.setdefault("marker", ".")
    return ax.plot(x, y, **kwargs)


def _decorate_time(ax, run: RunResult, st: PlotStyle, ylabel: str, yscale: Optional[str] = None):
    T = _TimeAxis(st)
    inj = run.config.injection
    if st.injection_band:
        ax.axvspan(T(inj.t_start), T(inj.t_start + inj.duration), color=st.color("injection"), zorder=0,
                   label="injection")
    ax.set_xscale(st.time_axis)
    if st.xlim is not None:
        ax.set_xlim(*st.xlim)
    elif len(run.t):
        ax.set_xlim(T(run.t[0]), T(run.t[-1]))
    scale = st.yscale or yscale
    if scale == "symlog":
        ax.set_yscale("symlog", linthresh=1e-3)
    elif scale:
        ax.set_yscale(scale)
    if st.ylim is not None:
        ax.set_ylim(*st.ylim)
    ax.set_xlabel(st.xlabel if st.xlabel is not None else f"Time [{T.unit}]")
    ax.set_ylabel(st.ylabel if st.ylabel is not None else ylabel)
    ax.grid(st.grid, which="major", alpha=0.4)


def _pops(run: RunResult, st: PlotStyle):
    """(label, variable suffix, colour) of the active populations."""
    if run.two_populations:
        return [("hot", "hot", st.color("hot")), ("cold", "cold", st.color("cold"))]
    return [("", "cold", st.color("cold"))]


def _vline(ax, st, t, color_key, label):
    ax.axvline(_TimeAxis(st)(t), color=st.color(color_key), ls="--", lw=1, label=label)


# --------------------------------------------------------------------------- time traces

def plot_temperatures(run: RunResult, style=None):
    st = _style(style)
    T = _TimeAxis(st)
    with st.context():
        fig, ax = _new_axes(st, rf"Temperatures, final $\langle T_e \rangle$ = {run.Te_avg[-1]:.3g} eV")
        for label, s, c in _pops(run, st):
            tag = f",{{\\rm {label}}}" if label else ""
            _plot(ax, st, T(run.t), run.fluid[f"Te_{s}"], color=c, label=rf"$T_{{e{tag}}}$")
            _plot(ax, st, T(run.t), run.fluid[f"Ti_{s}"], color=c, ls=":", label=rf"$T_{{i{tag}}}$")
        if run.two_populations:
            _plot(ax, st, T(run.t), run.Te_avg, color=st.color("average"), label=r"$\langle T_e \rangle$")
        t_tq = thermal_quench_time(run, st.tq_threshold)
        if np.isfinite(t_tq):
            _vline(ax, st, t_tq, "t_TQ", rf"$t_{{TQ}}$ = {t_tq * 1e6:.3g} $\mu$s")
        _decorate_time(ax, run, st, "Temperature [eV]", "log")
        _legend(ax, st)
    return fig


def plot_densities(run: RunResult, style=None):
    st = _style(style)
    T = _TimeAxis(st)
    plt = _plt()
    with st.context():
        fig, ax = _new_axes(st, "Densities")
        for label, s, c in _pops(run, st):
            tag = f",{{\\rm {label}}}" if label else ""
            _plot(ax, st, T(run.t), run.fluid[f"ne_{s}"], color=c, label=rf"$n_{{e{tag}}}$")
            _plot(ax, st, T(run.t), run.fluid[f"ni_{s}"], color=c, ls=":", label=rf"$n_{{i{tag}}}$")
        if run.two_populations:
            _plot(ax, st, T(run.t), run.ne_total, color=st.color("total"), label=r"$n_e$ (total)")
        cmap = plt.get_cmap(st.cmap)
        n_el = len(run.elements)
        for j, name in enumerate(run.elements):
            color = st.color("impurity") if n_el == 1 else cmap(j / max(n_el - 1, 1))
            _plot(ax, st, T(run.t), run.n_imp[:, j], ls="--", color=color,
                  label=rf"$n_{{\rm {_element_symbol(name)}}}$")
        _decorate_time(ax, run, st, r"Density [m$^{-3}$]", "log")
        _legend(ax, st, fontsize=st.legend_fontsize or 9)
    return fig


def plot_charge_states(run: RunResult, element: int = 0, style=None):
    st = _style(style)
    T = _TimeAxis(st)
    plt = _plt()
    name = run.elements[element]
    nij = run.nij[:, :, element]
    Z = run.atomic_numbers[element]
    with st.context():
        fig, ax = _new_axes(st, f"Charge states of {name}")
        cmap = plt.get_cmap(st.cmap)
        for i in range(Z + 1):
            symbol = _element_symbol(name)
            _plot(ax, st, T(run.t), nij[:, i], color=cmap(i / max(Z, 1)),
                  label=rf"${{\rm {symbol}}}^{{{i}+}}$")
        _plot(ax, st, T(run.t), np.sum(nij[:, :Z + 1], axis=1), color=st.color("total"),
              label=rf"$n_{{\rm {_element_symbol(name)}}}$ (total)")
        _decorate_time(ax, run, st, r"Density [m$^{-3}$]")
        _legend(ax, st, fontsize=st.legend_fontsize or 7, ncol=2)
    return fig


def plot_zeff(run: RunResult, terms: Dict[str, np.ndarray], style=None):
    st = _style(style)
    with st.context():
        fig, ax = _new_axes(st, "Effective charge")
        _plot(ax, st, _TimeAxis(st)(run.t), terms["Zeff"], color=st.color("total"))
        _decorate_time(ax, run, st, r"$Z_{\rm eff}$")
    return fig


def plot_powers(run: RunResult, terms: Dict[str, np.ndarray], style=None):
    st = _style(style)
    T = _TimeAxis(st)
    with st.context():
        fig, ax = _new_axes(st, "Power densities")
        _plot(ax, st, T(run.t), (terms["P_rad_hot"] + terms["P_rad_cold"]) * 1e-6, color=st.color("radiated"),
              label=r"$P_{\rm rad}$")
        _plot(ax, st, T(run.t), (terms["P_ohm_hot"] + terms["P_ohm_cold"]) * 1e-6, color=st.color("ohmic"),
              label=r"$P_{\rm ohm}$")
        _plot(ax, st, T(run.t), (terms["P_stoch_hot"] + terms["P_stoch_cold"]) * 1e-6,
              color=st.color("stochastic"), label=r"$P_{\rm stoch}$")
        if run.two_populations:
            _plot(ax, st, T(run.t), terms["P_rad_hot"] * 1e-6, color=st.color("hot"), ls=":",
                  label=r"$P_{\rm rad,hot}$")
            _plot(ax, st, T(run.t), terms["P_rad_cold"] * 1e-6, color=st.color("cold"), ls=":",
                  label=r"$P_{\rm rad,cold}$")
        t_c = radiative_collapse_time(run, terms)
        t_b = ohmic_radiative_balance_time(run, terms)
        t0 = run.config.injection.t_start
        if np.isfinite(t_c):
            _vline(ax, st, t_c, "collapse", rf"radiative collapse, $t - t_{{inj}}$ = {(t_c - t0) * 1e6:.3g} $\mu$s")
        if np.isfinite(t_b):
            _vline(ax, st, t_b, "balance",
                   rf"$P_{{\rm ohm}} \approx P_{{\rm rad}}$, $t - t_{{inj}}$ = {(t_b - t0) * 1e6:.3g} $\mu$s")
        _decorate_time(ax, run, st, r"Power density [MW m$^{-3}$]", "log")
        p_rad = np.nanmax(terms["P_rad_hot"] + terms["P_rad_cold"]) * 1e-6
        if st.ylim is None and ax.get_yscale() == "log" and np.isfinite(p_rad) and p_rad > 0:
            ax.set_ylim(bottom=1e-4 * p_rad)
        _legend(ax, st, fontsize=st.legend_fontsize or 9)
    return fig


def plot_energy(run: RunResult, terms: Dict[str, np.ndarray], style=None):
    st = _style(style)
    T = _TimeAxis(st)
    eb = energy_balance(run, terms)
    with st.context():
        fig, ax = _new_axes(st, "Energy balance")
        _plot(ax, st, T(run.t), (eb["W_th"][0] - eb["W_th"]) * 1e-3, color=st.color("total"),
              label=r"$W_{\rm th}(0) - W_{\rm th}$")
        _plot(ax, st, T(run.t), eb["E_rad"] * 1e-3, color=st.color("radiated"), label=r"$E_{\rm rad}$")
        _plot(ax, st, T(run.t), eb["E_ohm"] * 1e-3, color=st.color("ohmic"), label=r"$E_{\rm ohm}$")
        _plot(ax, st, T(run.t), eb["E_stoch"] * 1e-3, color=st.color("stochastic"), label=r"$E_{\rm stoch}$")
        _plot(ax, st, T(run.t), eb["residual"] * 1e-3, color=st.color("residual"), ls=":", label="residual")
        _decorate_time(ax, run, st, r"Energy density [kJ m$^{-3}$]")
        _legend(ax, st, fontsize=st.legend_fontsize or 9)
    return fig


def plot_currents(run: RunResult, terms: Dict[str, np.ndarray], style=None):
    st = _style(style)
    T = _TimeAxis(st)
    plt = _plt()
    with st.context():
        w, h = st.figsize
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(w, h * 1.3), sharex=True, layout="constrained")
        _title(ax1, st, "Current density and resistivity")
        if run.two_populations:
            _plot(ax1, st, T(run.t), terms["J_hot"], color=st.color("hot"), label=r"$J_{hot}$")
            _plot(ax1, st, T(run.t), terms["J_cold"], color=st.color("cold"), label=r"$J_{cold}$")
            _plot(ax2, st, T(run.t), terms["eta_hot"], color=st.color("hot"), label=r"$\eta_{hot}$")
        _plot(ax1, st, T(run.t), terms["J"], color=st.color("total"), ls="--", label=r"$J$")
        _plot(ax2, st, T(run.t), terms["eta_cold"], color=st.color("cold"),
              label=r"$\eta_{cold}$" if run.two_populations else r"$\eta$")
        _decorate_time(ax1, run, st, r"$J$ [A m$^{-2}$]")
        _decorate_time(ax2, run, st, r"$\eta$ [$\Omega$ m]", "log")
        ax1.set_xlabel("")
        for ax in (ax1, ax2):
            _legend(ax, st, fontsize=st.legend_fontsize or 9)
    return fig


def plot_electric_field(run: RunResult, terms: Dict[str, np.ndarray], style=None):
    st = _style(style)
    T = _TimeAxis(st)
    with st.context():
        fig, ax = _new_axes(st, "Parallel electric field and critical field")
        for label, s, c in _pops(run, st):
            tag = f" ({label})" if label else ""
            _plot(ax, st, T(run.t), terms[f"E_{s}"], color=c, label=rf"$E_\parallel${tag}")
            _plot(ax, st, T(run.t), critical_field(run.fluid[f"ne_{s}"], run.fluid[f"Te_{s}"]), color=c, ls=":",
                  label=rf"$E_c${tag}")
        _decorate_time(ax, run, st, r"$E$ [V m$^{-1}$]", "log")
        _legend(ax, st, fontsize=st.legend_fontsize or 9)
    return fig


def plot_exchange(run: RunResult, terms: Dict[str, np.ndarray], style=None):
    st = _style(style)
    T = _TimeAxis(st)
    plt = _plt()
    keys = ["P_ehot_ecold", "P_ihot_icold", "P_ehot_ihot", "P_ehot_icold", "P_ecold_icold", "P_ecold_ihot"]
    active = [k for k in keys if np.any(terms[k] != 0)]
    with st.context():
        fig, ax = _new_axes(st, r"Collisional energy exchange $P_{a \rightarrow b}$ (positive: a loses)")
        cmap = plt.get_cmap(st.cmap)
        for n, k in enumerate(active):
            a, b = (_species_label(s) for s in k[2:].split("_"))
            _plot(ax, st, T(run.t), terms[k] * 1e-6, color=cmap(n / max(len(active) - 1, 1)),
                  label=rf"$P_{{{a} \rightarrow {b}}}$")
        _decorate_time(ax, run, st, r"Power density [MW m$^{-3}$]", "symlog")
        _legend(ax, st, fontsize=st.legend_fontsize or 9)
    return fig


def plot_ablation(run: RunResult, terms: Dict[str, np.ndarray], style=None):
    st = _style(style)
    T = _TimeAxis(st)
    with st.context():
        fig, ax = _new_axes(st, "Material sources")
        _plot(ax, st, T(run.t), terms["S_D"], color=st.color("D"), label=r"$S_D$ (D atoms)")
        _plot(ax, st, T(run.t), terms["S_imp"], color=st.color("impurity"), label=r"$S_{\rm imp}$ (impurity atoms)")
        if run.config.injection.source == "ablation" and run.two_populations:
            _plot(ax, st, T(run.t), terms["g_hot"], color=st.color("hot"), ls="--", label=r"$g_{\rm hot}$ (ablation rate)")
            _plot(ax, st, T(run.t), terms["g_cold"], color=st.color("cold"), ls="--", label=r"$g_{\rm cold}$ (ablation rate)")
        _decorate_time(ax, run, st, r"Source [m$^{-3}$ s$^{-1}$]", "log")
        _legend(ax, st, fontsize=st.legend_fontsize or 9)
    return fig


def _mean_charge(run: RunResult) -> np.ndarray:
    """Mean charge <Z> of each impurity element along the run, shape (n_t, n_elements).

    Undefined where an element has no density, which happens before its injection.
    """
    k = np.arange(run.nij.shape[1])
    bound = np.einsum("k,tkj->tj", k, run.nij)
    total = np.sum(run.nij, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, bound / np.maximum(total, 1e-300), np.nan)


def plot_mean_charge(run: RunResult, terms: Dict[str, np.ndarray], style=None):
    """Mean charge of each impurity element and effective charge of the plasma."""
    st = _style(style)
    T = _TimeAxis(st)
    plt = _plt()
    Z_mean = _mean_charge(run)
    with st.context():
        fig, ax = _new_axes(st, "Mean impurity charge and effective charge")
        cmap = plt.get_cmap(st.cmap)
        n_el = len(run.elements)
        for j, name in enumerate(run.elements):
            color = st.color("impurity") if n_el == 1 else cmap(j / max(n_el - 1, 1))
            symbol = _element_symbol(name)
            _plot(ax, st, T(run.t), Z_mean[:, j], color=color,
                  label=rf"$\langle Z \rangle_{{\rm {symbol}}}$")
            ax.axhline(run.atomic_numbers[j], color=color, ls=":", lw=1,
                       label=rf"$Z_{{\rm {symbol}}}$ = {run.atomic_numbers[j]}")
        _plot(ax, st, T(run.t), terms["Zeff"], color=st.color("total"), label=r"$Z_{\rm eff}$")
        _decorate_time(ax, run, st, "Charge")
        _legend(ax, st, fontsize=st.legend_fontsize or 9)
    return fig


def plot_electron_budget(run: RunResult, style=None):
    """Where the free electrons come from: the main ions and the ionised impurities.

    Charge neutrality gives ne = ni + sum_{k,j} k n_{k,j}, so the two contributions add up
    to the total electron density. The third curve is the impurity share of that total.
    """
    st = _style(style)
    T = _TimeAxis(st)
    plt = _plt()
    k = np.arange(run.nij.shape[1])
    from_imp = np.einsum("k,tkj->tj", k, run.nij)
    ni = run.ni_cold + (run.ni_hot if run.two_populations else 0.0)
    with st.context():
        fig, ax = _new_axes(st, "Origin of the free electrons")
        # The main ions carry almost all the free electrons, so the two curves overlap.
        # The total is drawn first and thicker to stay visible underneath.
        _plot(ax, st, T(run.t), run.ne_total, color=st.color("total"),
              lw=(st.linewidth or 1.5) * 2.5, alpha=0.35, label=r"$n_e$ (total)")
        _plot(ax, st, T(run.t), ni, color=st.color("D"), label=r"$n_i$ (main ions)")
        cmap = plt.get_cmap(st.cmap)
        n_el = len(run.elements)
        for j, name in enumerate(run.elements):
            color = st.color("impurity") if n_el == 1 else cmap(j / max(n_el - 1, 1))
            _plot(ax, st, T(run.t), from_imp[:, j], color=color, ls="--",
                  label=rf"$\sum_k k\, n_{{k,\rm {_element_symbol(name)}}}$")
        _decorate_time(ax, run, st, r"Density [m$^{-3}$]", "log")
        _legend(ax, st, fontsize=st.legend_fontsize or 9)
    return fig


def _exchange_time(n, T_a, T_b, power):
    """Energy-exchange time [s] read back from an exchange power: P = 3 n e (T_a - T_b) / 2 tau.

    Undefined where the power vanishes, which is where the two temperatures are equal.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = 1.5 * n * E_CHARGE * (T_a - T_b) / power
    return np.where(np.isfinite(tau) & (tau > 0), tau, np.nan)


def plot_timescales(run: RunResult, terms: Dict[str, np.ndarray], style=None):
    """Characteristic times of every channel, which says what sets the pace of the quench.

    Each loss or heating channel gives W_th / P, the time it would take on its own to empty
    or refill the thermal energy. The collisional times are read back from the exchange
    powers, so they are the times the model actually uses.
    """
    st = _style(style)
    T = _TimeAxis(st)
    W = thermal_energy(run)
    with st.context():
        fig, ax = _new_axes(st, "Characteristic times")

        def channel(power, key, label):
            with np.errstate(divide="ignore", invalid="ignore"):
                tau = W / np.asarray(power)
            _plot(ax, st, T(run.t), np.where(np.isfinite(tau) & (tau > 0), tau, np.nan),
                  color=st.color(key), label=label)

        channel(terms["P_rad_hot"] + terms["P_rad_cold"], "radiated", r"$W_{\rm th} / P_{\rm rad}$")
        channel(terms["P_ohm_hot"] + terms["P_ohm_cold"], "ohmic", r"$W_{\rm th} / P_{\rm ohm}$")
        channel(terms["P_stoch_hot"] + terms["P_stoch_cold"], "stochastic",
                r"$W_{\rm th} / P_{\rm stoch}$")
        if run.two_populations:
            _plot(ax, st, T(run.t),
                  _exchange_time(run.ne_hot, run.Te_hot, run.Te_cold, terms["P_ehot_ecold"]),
                  color=st.color("average"), ls="--", label=r"$\tau_{ee}$ (hot to cold)")
            tau_ei = _exchange_time(run.ne_hot, run.Te_hot, run.Ti_hot, terms["P_ehot_ihot"])
        else:
            tau_ei = _exchange_time(run.ne_cold, run.Te_cold, run.Ti_cold, terms["P_ecold_icold"])
        _plot(ax, st, T(run.t), tau_ei, color=st.color("cold"), ls="--", label=r"$\tau_{ei}$")
        t_tq = thermal_quench_time(run, st.tq_threshold)
        if np.isfinite(t_tq):
            _vline(ax, st, t_tq, "t_TQ", rf"$t_{{TQ}}$ = {t_tq * 1e6:.3g} $\mu$s")
        _decorate_time(ax, run, st, "Time [s]", "log")
        _legend(ax, st, fontsize=st.legend_fontsize or 9)
    return fig


def plot_cooling_curve(run: RunResult, terms: Dict[str, np.ndarray], style=None, provider=None):
    """Effective radiative cooling rate of the run against its coronal equilibrium value.

    The run follows L_Z = P_rad / (n_e n_imp), the radiated power per electron and per
    impurity ion, plotted against the density-weighted electron temperature. The dotted
    curves are the coronal equilibrium of each element at the final electron density.

    The gap between the two is the non-coronal enhancement, the reason this model follows
    the charge states in time instead of assuming equilibrium. With several elements, the
    trajectory is their weighted mean while each coronal curve is its own. Below
    ``model.radiation_limit.Te_below`` the radiated power is capped, so the trajectory
    there is the limiter and not a cooling rate.
    """
    from .atomic import get_atomic_data
    st = _style(style)
    plt = _plt()
    n_imp = np.sum(run.n_imp, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        L_run = (terms["P_rad_hot"] + terms["P_rad_cold"]) / (run.ne_total * n_imp)
    ok = np.isfinite(L_run) & (L_run > 0) & (run.Te_avg > 0)
    Te_grid = np.logspace(0, np.log10(max(run.Te_avg[0], 10.0)), 200)
    rates = get_atomic_data(run.config.element_names, provider).rates
    ne_ref = float(run.ne_total[-1])
    with st.context():
        fig, ax = _new_axes(st, "Radiative cooling rate")
        ax.plot(run.Te_avg[ok], L_run[ok], color=st.color("radiated"), label="run")
        if np.any(ok):
            ax.plot(run.Te_avg[ok][0], L_run[ok][0], "o", color=st.color("radiated"), label="start")
        cmap = plt.get_cmap(st.cmap)
        for j, el in enumerate(rates):
            color = st.color("impurity") if len(rates) == 1 else cmap(j / max(len(rates) - 1, 1))
            L_cor = [coronal_cooling_rate(el, ne_ref, Te) for Te in Te_grid]
            ax.plot(Te_grid, L_cor, color=color, ls=":",
                    label=rf"${{\rm {_element_symbol(el.element.name)}}}$, coronal")
        cap = run.config.model.radiation_limit
        if cap.enabled:
            ax.axvline(cap.Te_below, color=st.color("residual"), ls="--", lw=1,
                       label=rf"limiter below {cap.Te_below:.0f} eV")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(st.xlabel if st.xlabel is not None else r"$\langle T_e \rangle$ [eV]")
        ax.set_ylabel(st.ylabel if st.ylabel is not None else r"$L_Z$ [W m$^3$]")
        ax.grid(st.grid, alpha=0.4)
        _legend(ax, st, fontsize=st.legend_fontsize or 9)
    return fig


def plot_shards(run: RunResult, bins: int = 30, style=None):
    st = _style(style)
    if run.r_p_samples is None:
        raise ValueError("This run has no sampled shard radii (injection.ablation.parks not set).")
    r = np.asarray(run.r_p_samples)
    ab = run.config.injection.ablation
    kappa = parks_kappa(ab.n_shards, ab.parks.n_atoms, ab.parks.solid_density)
    with st.context():
        fig, ax = _new_axes(st, f"Shard radii: mean {r.mean() * 1e3:.2f} mm, max {r.max() * 1e3:.2f} mm, "
                                f"std {r.std() * 1e3:.2f} mm")
        ax.hist(r * 1e3, bins=bins, density=True, color=st.color("samples"), alpha=0.6, label="samples")
        x = np.linspace(ab.parks.r_min, r.max(), 300)
        pdf = parks_pdf(x, kappa)
        ax.plot(x * 1e3, pdf / trapezoid(pdf, x * 1e3), color=st.color("parks"),
                label=rf"Parks, $\kappa_p^{{-1}}$ = {1e3 / kappa:.2f} mm")
        ax.set_xlabel(st.xlabel if st.xlabel is not None else r"$r_s$ [mm]")
        ax.set_ylabel(st.ylabel if st.ylabel is not None else "Probability density [mm$^{-1}$]")
        ax.grid(st.grid, alpha=0.4)
        _legend(ax, st)
    return fig


# --------------------------------------------------------------------------- comparison of scan points

def _series_fhte(run):
    fh = run.fhte
    if fh is None:
        from .fhte import hot_tail_from_run
        fh = hot_tail_from_run(run)
    return fh.t_output, fh.fraction


# name -> (label, needs the model terms, function(run, terms) -> (t, y), y scale)
SERIES = {
    "Te_avg": (r"$\langle T_e \rangle$ [eV]", False, lambda r, tm: (r.t, r.Te_avg), "log"),
    "Te_cold": (r"$T_{e,cold}$ [eV]", False, lambda r, tm: (r.t, r.Te_cold), "log"),
    "Te_hot": (r"$T_{e,hot}$ [eV]", False, lambda r, tm: (r.t, r.Te_hot), "log"),
    "Ti_cold": (r"$T_{i,cold}$ [eV]", False, lambda r, tm: (r.t, r.Ti_cold), "log"),
    "ne_total": (r"$n_e$ [m$^{-3}$]", False, lambda r, tm: (r.t, r.ne_total), "log"),
    "ne_cold": (r"$n_{e,cold}$ [m$^{-3}$]", False, lambda r, tm: (r.t, r.ne_cold), "log"),
    "ni_cold": (r"$n_{i,cold}$ [m$^{-3}$]", False, lambda r, tm: (r.t, r.ni_cold), "log"),
    "n_imp": (r"$n_{imp}$ [m$^{-3}$]", False, lambda r, tm: (r.t, np.sum(r.n_imp, axis=1)), "log"),
    "Zeff": (r"$Z_{\rm eff}$", True, lambda r, tm: (r.t, tm["Zeff"]), "linear"),
    "P_rad": (r"$P_{rad}$ [MW m$^{-3}$]", True,
              lambda r, tm: (r.t, (tm["P_rad_hot"] + tm["P_rad_cold"]) * 1e-6), "log"),
    "P_ohm": (r"$P_{ohm}$ [MW m$^{-3}$]", True,
              lambda r, tm: (r.t, (tm["P_ohm_hot"] + tm["P_ohm_cold"]) * 1e-6), "log"),
    "P_stoch": (r"$P_{stoch}$ [MW m$^{-3}$]", True,
                lambda r, tm: (r.t, (tm["P_stoch_hot"] + tm["P_stoch_cold"]) * 1e-6), "log"),
    "E_parallel": (r"$E_\parallel$ [V m$^{-1}$]", True, lambda r, tm: (r.t, tm["E_cold"]), "log"),
    "W_th": (r"$W_{th}$ [kJ m$^{-3}$]", False,
             lambda r, tm: (r.t, _thermal_energy(r) * 1e-3), "log"),
    "n_RE_hot_tail": (r"$n_{RE}/n_0$ (hot tail)", False, lambda r, tm: _series_fhte(r), "log"),
}

# The six collisional exchange powers, from species a to species b, positive when a loses.
# They change sign when the two temperatures cross, hence the symmetric-log scale.
for _key in ("P_ehot_ecold", "P_ihot_icold", "P_ehot_ihot", "P_ehot_icold", "P_ecold_icold",
             "P_ecold_ihot"):
    _a, _b = (_species_label(_s) for _s in _key[2:].split("_"))
    SERIES[_key] = (rf"$P_{{{_a} \rightarrow {_b}}}$ [MW m$^{{-3}}$]", True,
                    (lambda key: lambda r, tm: (r.t, tm[key] * 1e-6))(_key), "symlog")
del _key, _a, _b


def _thermal_energy(run):
    from .diagnostics import thermal_energy
    return thermal_energy(run)


def plot_scan_series(results, variable: str = "Te_avg", axis: Optional[str] = None,
                     at: Optional[Dict[str, Any]] = None, points=None, style=None):
    """Time traces of several points of a scan on one figure, coloured by the scanned value.

    axis : scan axis to vary (default: the only axis of a 1D scan)
    at : values of the other axes (nearest point), e.g. {"n_D": 1e22}
    points : values of the varied axis to draw (default: all)
    """
    st = _style(style)
    T = _TimeAxis(st)
    plt = _plt()
    if variable not in SERIES:
        raise KeyError(f"Unknown variable '{variable}'. Available: {list(SERIES)}.")
    label, needs_terms, getter, yscale = SERIES[variable]
    names = results.axis_names
    if not names:
        raise ValueError("This file contains a single run: nothing to compare.")
    if axis is None:
        if len(names) != 1:
            raise ValueError(f"Give the axis to vary among {names} (compare.axis).")
        axis = names[0]
    k = results._axis_position(axis)
    full_name = names[k]
    base = list(results.index(dict(at or {})))
    values = np.asarray(results.axes[k][1])
    if points is None:
        selected = list(range(len(values)))
    else:
        selected = sorted({int(results.index({full_name: v})[k]) for v in np.atleast_1d(points)})
    numeric = values.dtype.kind in "iuf"
    vals = values[selected].astype(float) if numeric else np.arange(len(selected), dtype=float)
    log_colour = numeric and np.all(vals > 0) and vals.max() / vals.min() >= 10
    norm = plt.matplotlib.colors.LogNorm(vals.min(), vals.max()) if log_colour else \
        plt.matplotlib.colors.Normalize(vals.min(), vals.max() if vals.max() > vals.min() else vals.min() + 1)
    cmap = plt.get_cmap(st.cmap)
    fixed = {n: results.coordinates(tuple(base))[n] for i, n in enumerate(names) if i != k}
    subtitle = ", ".join(f"{_axis_symbol(n)} = {v:.3g}" if isinstance(v, float) else f"{_axis_symbol(n)} = {v}"
                         for n, v in fixed.items())
    with st.context():
        fig, ax = _new_axes(st, f"{label} versus {_axis_symbol(full_name)}"
                                + (f" at {subtitle}" if subtitle else ""))
        for i, value in zip(selected, vals):
            idx = list(base)
            idx[k] = i
            run = results.run(tuple(idx))
            if len(run.t) < 2:
                continue
            terms = compute_terms(run) if needs_terms else None
            t, y = getter(run, terms)
            _plot(ax, st, T(t), y, color=cmap(norm(value)), lw=1.5)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(mappable, ax=ax, label=_axis_label(full_name) if numeric else full_name)
        ax.set_ylabel(st.ylabel if st.ylabel is not None else label)
        ax.set_xlabel(st.xlabel if st.xlabel is not None else f"Time [{T.unit}]")
        ax.set_xscale(st.time_axis)
        scale = st.yscale or yscale
        if scale == "symlog":
            ax.set_yscale("symlog", linthresh=1e-3)
        else:
            ax.set_yscale(scale)
        if st.xlim is not None:
            ax.set_xlim(*st.xlim)
        if st.ylim is not None:
            ax.set_ylim(*st.ylim)
        ax.grid(st.grid, alpha=0.4)
    return fig


# --------------------------------------------------------------------------- hot tail (FHTE)

def plot_hot_tail(res, style=None, t0_injection=None):
    """Hot-tail estimate: runaway fraction and backward momentum trajectories (tqtoy.fhte.FHTEResult)."""
    st = _style(style)
    T = _TimeAxis(st)
    plt = _plt()
    with st.context():
        w, h = st.figsize
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(w, h * 1.3), sharex=True, layout="constrained")
        _title(ax1, st, rf"Hot-tail runaways (FHTE): final $n_{{RE}}/n_0$ = {res.final_fraction:.3g}")
        frac = np.where(res.fraction > 0, res.fraction, np.nan)
        ax1.plot(T(res.t_output), frac, marker="o", color=st.color("hot"), label=r"$n_{RE}/n_0$")
        ax1.set_yscale("log")
        ax1.set_ylabel(st.ylabel if st.ylabel is not None else r"$n_{RE}/n_0$")
        cmap = plt.get_cmap(st.cmap)
        n = len(res.t_output)
        for i in range(1, n):
            if not res.p_c[i] < 100:                      # E <= E_c: no runaway, no trajectory
                continue
            ax2.plot(T(res.t_trajectory[i]), res.p_trajectory[i], color=cmap(i / max(n - 1, 1)), lw=1.2)
            ax2.plot(T(res.t_trajectory[i, -1]), res.p_limit[i], "x", color=cmap(i / max(n - 1, 1)))
        pc = np.where(res.p_c < 100, res.p_c, np.nan)
        ax2.plot(T(res.t_output), pc, "o--", color=st.color("total"), label=r"$p_c$ (critical momentum)")
        ax2.plot([], [], color=cmap(0.5), label="backward trajectories")
        ax2.plot([], [], "x", color=cmap(0.5), label=r"$p_{limit}$ (at the initial time)")
        ax2.set_yscale("log")
        ax2.set_ylabel(r"$p / m_e c$")
        for ax in (ax1, ax2):
            ax.set_xscale(st.time_axis)
            if st.xlim is not None:
                ax.set_xlim(*st.xlim)
            ax.grid(st.grid, alpha=0.4)
            _legend(ax, st, fontsize=st.legend_fontsize or 9)
        if st.time_axis == "log" and T(res.t_output[0]) > 0:
            ax2.set_xlim(left=T(res.t_output[0]))
        ax2.set_xlabel(st.xlabel if st.xlabel is not None else f"Time [{T.unit}]")
    return fig


def plot_fhte_inputs(traces, res, style=None):
    """Prescribed evolution of a standalone FHTE run: temperature, density and fields."""
    st = _style(style)
    T = _TimeAxis(st)
    plt = _plt()
    t = traces["t"]
    with st.context():
        w, h = st.figsize
        fig, axes = plt.subplots(3, 1, figsize=(w, h * 1.6), sharex=True, layout="constrained")
        _title(axes[0], st, "FHTE inputs")
        axes[0].plot(T(t), traces["Te"], color=st.color("cold"))
        axes[0].set_ylabel(r"$T_e$ [eV]")
        axes[0].set_yscale("log")
        axes[1].plot(T(t), traces["ne"], color=st.color("total"))
        axes[1].set_ylabel(r"$n_e$ [m$^{-3}$]")
        axes[2].plot(T(res.t_output), res.E, "o-", color=st.color("hot"), label=r"$E_\parallel$")
        axes[2].plot(T(res.t_output), res.E_c, "s--", color=st.color("total"), label=r"$E_c$")
        axes[2].set_yscale("log")
        axes[2].set_ylabel(r"$E$ [V m$^{-1}$]")
        _legend(axes[2], st)
        for ax in axes:
            ax.set_xscale(st.time_axis)
            ax.grid(st.grid, alpha=0.4)
        axes[2].set_xlabel(st.xlabel if st.xlabel is not None else f"Time [{T.unit}]")
    return fig


# --------------------------------------------------------------------------- scans

#: Axis label of each quantity that can be mapped over a scan (see diagnostics.SCAN_QUANTITIES).
MAP_LABELS = {
    "t_TQ": r"$t_{\rm TQ}$ [$\mu$s]",
    "t_radiative_collapse": r"$t_{\rm collapse}$ [$\mu$s]",
    "Te_avg_final": r"final $\langle T_e \rangle$ [eV]",
    "n_RE_hot_tail": r"$n_{RE} / n_0$ (hot tail)",
    "n_RE_hot_tail_density": r"$n_{RE}$ (hot tail) [m$^{-3}$]",
    "radiated_fraction": r"$E_{\rm rad} / (E_{\rm rad} + E_{\rm stoch} + E_{\rm lin})$",
    "radiated_fraction_net": r"$(E_{\rm rad} - E_{\rm ohm}) / "
                             r"(E_{\rm rad} + E_{\rm stoch} + E_{\rm lin} - E_{\rm ohm})$",
}

#: Figure title of each mapped quantity (SCAN_QUANTITIES holds the plain-text descriptions).
MAP_TITLES = {
    "t_TQ": "Thermal quench time",
    "t_radiative_collapse": "Time of maximum radiated power",
    "Te_avg_final": "Final average electron temperature",
    "n_RE_hot_tail": "Hot-tail runaway fraction at the last evaluation time",
    "n_RE_hot_tail_density": "Hot-tail runaway density at the last evaluation time",
    "radiated_fraction": "Radiated share of the lost energy",
    "radiated_fraction_net": "Radiated share of the lost energy, net of the ohmic input",
}


def _math_unit(unit: str) -> str:
    return re.sub(r"\^(-?\d+)", r"$^{\1}$", unit)       # m^-3 -> m$^{-3}$


def _axis_symbol(name: str) -> str:
    """Math label of a scan axis, without its unit."""
    return _axis_label(name).split(" [")[0]


def _axis_label(name: str) -> str:
    """Math label and unit of a scan axis, from its parameter path."""
    symbol = AXIS_LABELS.get(name)
    if symbol is None:
        parts = name.split(".")
        if len(parts) == 3 and parts[0] == "impurities":
            element = _element_symbol(parts[1])
            symbol = (rf"$n_{{\rm {element}}}$" if parts[2] == "injected_atom_density" else
                      rf"$n_{{\rm {element},0}}$" if parts[2] == "background_density" else None)
        if symbol is None:
            symbol = parts[-1].replace("_", " ")        # no math symbol defined: the parameter name
    unit = _math_unit(parameter_unit(name))
    return f"{symbol} [{unit}]" if unit else symbol


def _axis_scale(values, setting: str) -> str:
    if setting != "auto":
        return setting
    v = np.asarray(values)
    # Two decades or more of positive values are taken as a logarithmic axis. The relative
    # tolerance accepts a span of exactly two decades, which round-off can place just below.
    if v.dtype.kind in "iuf" and np.all(v > 0) and v.max() / v.min() > 100.0 * (1.0 - 1e-9):
        return "log"
    return "linear"


def _cell_edges(values, scale: str):
    """Cell edges around the sampled values, for pcolormesh with shading="flat".

    The midpoints are taken in the space of the axis, geometric on a logarithmic one and
    arithmetic otherwise. An arithmetic midpoint on a coarse logarithmic grid puts the first
    edge at a non-positive value, which a logarithmic axis then drops along with the cells
    that lie beyond it.
    """
    v = np.asarray(values, dtype=float)
    if scale == "log":
        v = np.log10(v)
    if v.size == 1:
        half = 0.5 if scale == "log" else (abs(float(v[0])) * 0.1 or 0.5)
        edges = np.array([v[0] - half, v[0] + half])
    else:
        inner = 0.5 * (v[:-1] + v[1:])
        edges = np.concatenate(([2 * v[0] - inner[0]], inner, [2 * v[-1] - inner[-1]]))
    return 10.0 ** edges if scale == "log" else edges


def plot_scan_map(results, quantity: str = "t_TQ", log: Optional[bool] = None,
                  values: Optional[np.ndarray] = None, style=None):
    """Map of a scalar quantity over a 1D or 2D scan (see diagnostics.SCAN_QUANTITIES)."""
    st = _style(style)
    if log is not None:
        st = dataclasses.replace(st, log=log)
    if values is None:
        kwargs = {"threshold": st.tq_threshold} if quantity == "t_TQ" else {}
        data = scan_map(results, quantity, **kwargs)
    else:
        data = np.asarray(values, dtype=float)
    unit_label = MAP_LABELS.get(quantity, quantity.replace("_", " "))
    if quantity.startswith("t_"):
        data = data * 1e6
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.log10(data) if st.log else data
    zero = np.isneginf(z)                     # zero values on a log scale (e.g. no runaway)
    z = np.where(np.isfinite(z), z, np.nan)
    label = st.colorbar_label if st.colorbar_label is not None else (r"$\log_{10}$ " if st.log else "") + unit_label
    default_title = MAP_TITLES.get(quantity, SCAN_QUANTITIES.get(quantity, quantity).replace(" [s]", ""))
    if quantity == "t_TQ":
        default_title = rf"Thermal quench time: first time with $\langle T_e \rangle$ < {st.tq_threshold:g} eV"
    plt = _plt()
    names = results.axis_names
    with st.context():
        fig, ax = _new_axes(st, default_title)
        if len(names) == 1:
            x = np.asarray(results.axes[0][1])
            ax.plot(x, data, marker=st.marker, color=st.color("line"))
            ax.set_xlabel(st.xlabel if st.xlabel is not None else _axis_label(names[0]))
            ax.set_ylabel(st.ylabel if st.ylabel is not None else unit_label)
            ax.set_xscale(_axis_scale(x, st.map_xscale))
            ax.set_yscale("log" if st.log else "linear")
            if st.vmin is not None or st.vmax is not None:      # displayed units: log10 if st.log
                lim = [None if v is None else (10**v if st.log else v) for v in (st.vmin, st.vmax)]
                ax.set_ylim(*lim)
            ax.grid(st.grid, alpha=0.4)
            return fig
        if len(names) != 2:
            raise ValueError("plot_scan_map supports 1D and 2D scans.")
        x, y = np.asarray(results.axes[0][1]), np.asarray(results.axes[1][1])
        # The scales are set before the mesh is drawn, so that the limits it computes are
        # the ones the axes keep.
        xscale = _axis_scale(x, st.map_xscale)
        yscale = _axis_scale(y, st.map_yscale)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        finite = np.isfinite(z)
        vmin = st.vmin if st.vmin is not None else (np.nanmin(z) if finite.any() else 0.0)
        vmax = st.vmax if st.vmax is not None else (np.nanmax(z) if finite.any() else 1.0)
        cmap = plt.get_cmap(st.cmap).copy()
        cmap.set_bad(st.bad_color)
        cmap.set_under(cmap(0.0))
        if zero.any():                        # zeros drawn with the lowest colour ("below vmin")
            z = np.where(zero, vmin - 1.0 - abs(vmin), z)
            finite = np.isfinite(z)
        if st.filled and finite.sum() > 3:
            ax.set_facecolor(st.bad_color)
            levels = np.linspace(vmin, vmax, st.filled_levels + 1)
            mappable = ax.contourf(x, y, z.T, levels=levels, cmap=cmap, extend="both")
        else:
            mappable = ax.pcolormesh(_cell_edges(x, xscale), _cell_edges(y, yscale),
                                     np.ma.masked_invalid(z.T), shading="flat", cmap=cmap,
                                     vmin=vmin, vmax=vmax)
        z_lines = np.where(z < vmin, vmin, z)
        if st.contours and min(len(x), len(y)) >= 3 and finite.sum() > 3 and vmax > vmin:
            levels = st.contour_levels
            if isinstance(levels, (int, float)) and not isinstance(levels, bool):
                levels = np.linspace(vmin, vmax, int(levels) + 2)[1:-1]
            cs = ax.contour(x, y, z_lines.T, levels=levels, colors=st.contour_color, linewidths=st.contour_linewidth,
                            linestyles="solid")
            if st.contour_labels:
                ax.clabel(cs, fontsize="small", fmt=st.contour_label_format)
        _map_overlays(ax, results, x, y, st)
        fig.colorbar(mappable, ax=ax, label=label, format="%.3g", extend="min" if zero.any() else "neither")
        ax.set_xlabel(st.xlabel if st.xlabel is not None else _axis_label(names[0]))
        ax.set_ylabel(st.ylabel if st.ylabel is not None else _axis_label(names[1]))
    return fig


def _map_overlays(ax, results, x, y, st: PlotStyle):
    """Isocontours of a second quantity and ratio lines on a 2D map."""
    if st.overlay:
        ov = st.overlay
        data = scan_map(results, ov["quantity"])
        levels = ov.get("levels", [0.5])
        levels = list(levels) if isinstance(levels, (list, tuple)) else [levels]
        finite = data[np.isfinite(data)]
        if finite.size and not any(finite.min() <= lv <= finite.max() for lv in levels):
            import warnings
            warnings.warn(f"overlay {ov['quantity']}: levels {levels} outside the data range "
                          f"[{finite.min():.3g}, {finite.max():.3g}], nothing drawn.")
        if np.sum(np.isfinite(data)) > 3:
            cs = ax.contour(x, y, data.T, levels=sorted(levels), colors=str(ov.get("color", "lime")),
                            linewidths=float(ov.get("linewidth", 2.0)), linestyles="solid")
            if ov.get("labels", True):
                ax.clabel(cs, fontsize="small", fmt=f"{ov['quantity']} = %.3g")
    if st.ratio_lines:
        xs = np.geomspace(np.min(x), np.max(x), 400) if np.all(np.asarray(x) > 0) else np.linspace(np.min(x), np.max(x), 400)
        for k in st.ratio_lines:
            ys = k * xs
            keep = (ys >= np.min(y)) & (ys <= np.max(y))
            if keep.any():
                ax.plot(xs[keep], ys[keep], "--", color=st.ratio_line_color, lw=1.5)
                i = int(np.flatnonzero(keep)[len(np.flatnonzero(keep)) // 5])
                ax.annotate(f"{k:g}", (xs[i], ys[i]), color=st.ratio_line_color, fontsize="small",
                            xytext=(3, 3), textcoords="offset points")


def plot_radiation_rates(element, ne: float = 1e20, Te=None, provider=None, style=None):
    """Total radiated power coefficient (PLT + PRB) of each charge state versus Te."""
    from .atomic import AtomicData
    st = _style(style)
    plt = _plt()
    Te = np.logspace(0, 4.5, 300) if Te is None else np.asarray(Te)
    el = AtomicData([element], provider).rates[0]
    with st.context():
        fig, ax = _new_axes(st, f"Radiated power coefficients of {el.element.name} (ne = {ne:.1e} m$^{{-3}}$)")
        cmap = plt.get_cmap(st.cmap)
        for i in range(el.Z + 1):
            L = np.array([(el.line[i](ne, T) if el.line[i] else 0.0) +
                          (el.continuum[i](ne, T) if el.continuum[i] else 0.0) for T in Te])
            ax.plot(Te, L, color=cmap(i / el.Z), label=f"{i}+")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(st.xlabel if st.xlabel is not None else r"$T_e$ [eV]")
        ax.set_ylabel(st.ylabel if st.ylabel is not None else r"$L_Z$ [W m$^3$]")
        ax.grid(st.grid, alpha=0.4)
        _legend(ax, st, fontsize=st.legend_fontsize or 7, ncol=2)
    return fig


# --------------------------------------------------------------------------- batch

FIGURES = {
    "temperatures": "electron and ion temperatures",
    "densities": "electron, ion and impurity densities",
    "charge_states": "impurity charge-state densities",
    "zeff": "effective charge",
    "mean_charge": "mean charge of each impurity element and effective charge",
    "powers": "ohmic, radiated and stochastic powers",
    "energy": "time-integrated energy balance",
    "currents": "current density and resistivity",
    "efield": "parallel electric field vs critical field",
    "exchange": "thermal exchange between species",
    "timescales": "characteristic time of each loss and exchange channel",
    "cooling_curve": "radiative cooling rate of the run against coronal equilibrium",
    "electron_budget": "origin of the free electrons: main ions and ionised impurities",
    "sources": "injection / ablation sources",
    "shards": "shard size distribution (Parks sampling only)",
    "hot_tail": "hot-tail runaway fraction and momentum trajectories (FHTE)",
}
DEFAULT_FIGURES = ("temperatures", "densities", "charge_states", "powers", "energy")


def make_figures(run: RunResult, names: Iterable[str] = DEFAULT_FIGURES, outdir=None, fmt: str = "png",
                 terms: Optional[Dict[str, np.ndarray]] = None, prefix: str = "", style=None):
    """Create the requested figures for one run. Saves them in ``outdir`` if given.

    ``style`` is a PlotStyle, a dict of PlotStyle fields, or None (defaults).
    Returns a dict name -> Figure.
    """
    st = _style(style)
    names = list(names)
    if len(run.t) < 2:
        raise ValueError(f"The run has {len(run.t)} valid output time(s) (status {run.status}): nothing to plot.")
    unknown = [n for n in names if n not in FIGURES]
    if unknown:
        raise KeyError(f"Unknown figure(s) {unknown}. Available: {list(FIGURES)}.")
    needs_terms = set(names) - {"temperatures", "densities", "charge_states", "shards", "hot_tail",
                               "electron_budget"}
    if needs_terms and terms is None:
        terms = compute_terms(run)
    figs = {}
    with_terms = {"zeff": plot_zeff, "powers": plot_powers, "energy": plot_energy, "currents": plot_currents,
                  "efield": plot_electric_field, "exchange": plot_exchange, "sources": plot_ablation,
                  "mean_charge": plot_mean_charge, "timescales": plot_timescales,
                  "cooling_curve": plot_cooling_curve}
    for name in names:
        if name == "temperatures":
            figs[name] = plot_temperatures(run, style=st)
        elif name == "densities":
            figs[name] = plot_densities(run, style=st)
        elif name == "charge_states":
            for j, el in enumerate(run.elements):
                figs[name if j == 0 else f"{name}_{el}"] = plot_charge_states(run, j, style=st)
        elif name == "shards":
            figs[name] = plot_shards(run, style=st)
        elif name == "electron_budget":
            figs[name] = plot_electron_budget(run, style=st)
        elif name == "hot_tail":
            fh = run.fhte
            if fh is None or fh.p_trajectory.shape[1] < 10:
                # Recompute with more evaluation times and trajectory points for the figure.
                # The value at the last time does not depend on these two settings.
                import dataclasses as _dc
                from .fhte import hot_tail_from_run
                cfg = run.config.fhte
                spacing = "log" if st.time_axis == "log" else "linear"
                cfg = _dc.replace(cfg, n_output=max(cfg.n_output, 15), n_trajectory=max(cfg.n_trajectory, 60),
                                  output_spacing=spacing, trajectory_spacing=spacing)
                fh = hot_tail_from_run(run, config=cfg)
            figs[name] = plot_hot_tail(fh, style=st)
        else:
            figs[name] = with_terms[name](run, terms, style=st)
    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(outdir / f"{prefix}{name}.{fmt}", dpi=st.dpi)
    return figs


def status_text(run: RunResult) -> str:
    return STATUS_LABELS.get(run.status, str(run.status)) + (f" ({run.message})" if run.message else "")

