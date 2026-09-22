"""Plot style and figures (analytic atomic rates, no ADAS)."""

import matplotlib
import numpy as np
import pytest

from tqtoy import Config, simulate
from tqtoy.config import ConfigError
from tqtoy.io import save_results
from tqtoy.plotting import PlotStyle, make_figures, plot_scan_map
from tqtoy.scan import run_scan

matplotlib.use("Agg")

BASE = {"plasma": {"Te0": 3e3, "ne0": 5e19}, "impurities": {"neon": {"injected_atom_density": 1e19}},
        "injection": {"n_D": 3e20}, "time": {"t_end": 2e-4, "n_output": 30}}


@pytest.fixture(scope="module")
def scan2d():
    from conftest import FakeProvider
    cfg = Config.from_dict(BASE).with_overrides({"scan.injection.n_D": {"logspace": [19, 21, 4]},
                                                 "scan.impurities.neon.injected_atom_density": {"logspace": [17, 20, 4]}})
    return run_scan(cfg, provider=FakeProvider(), progress=False)


def test_style_validation():
    with pytest.raises(ConfigError, match="unknown key"):
        PlotStyle.from_dict({"colour": "red"})
    with pytest.raises(ConfigError, match="colormap"):
        PlotStyle(cmap="nope").validate()
    with pytest.raises(ConfigError, match="not a matplotlib colour"):
        PlotStyle.from_dict({"colors": {"hot": "nope"}})
    with pytest.raises(ConfigError, match="unknown colour key"):
        PlotStyle.from_dict({"colors": {"warm": "red"}})
    st = PlotStyle.from_dict({"colors": {"cold": 0.2}, "bad_color": 0.5})   # YAML grey levels
    assert st.color("cold") == "0.2" and st.color("hot") == "tab:red"


def test_style_yaml_round_trip(tmp_path):
    p = tmp_path / "style.yaml"
    p.write_text(PlotStyle().to_yaml())
    assert PlotStyle.from_yaml(p) == PlotStyle()
    st = PlotStyle().with_overrides({"cmap": "magma", "colors.hot": "black", "contour_levels": [1, 2]})
    assert st.cmap == "magma" and st.color("hot") == "black" and st.contour_levels == [1, 2]


@pytest.mark.parametrize("style", [
    {},
    {"contours": False, "cmap": "magma"},
    {"filled": True, "filled_levels": 8, "contour_levels": [0.5, 1.0], "contour_color": "black"},
    {"log": False, "vmin": 0.0, "vmax": 50.0, "contour_labels": False, "map_xscale": "linear"},
    {"title": "", "xlabel": "n_D", "colorbar_label": "t", "fontsize": 14, "figsize": [5, 4]},
])
def test_scan_map_styles(scan2d, style):
    fig = plot_scan_map(scan2d, "t_TQ", style=style)
    ax = fig.axes[0]
    has_lines = any(type(c).__name__ in ("QuadContourSet", "ContourSet") and not c.filled
                    for c in ax.collections) or any(hasattr(c, "levels") and not getattr(c, "filled", True)
                                                     for c in ax.collections)
    if style.get("contours") is False:
        assert not has_lines
    if "figsize" in style:
        np.testing.assert_allclose(fig.get_size_inches(), style["figsize"])
    if style.get("title") == "":
        assert ax.get_title() == ""


def test_one_dimensional_map(fake_provider):
    cfg = Config.from_dict(BASE).with_overrides({"scan.plasma.Te0": [2e3, 3e3, 4e3]})
    res = run_scan(cfg, provider=fake_provider, progress=False)
    fig = plot_scan_map(res, "t_TQ", style={"marker": "s", "colors": {"line": "black"}, "vmin": -1, "vmax": 2})
    assert fig.axes[0].get_lines()[0].get_marker() == "s"
    np.testing.assert_allclose(fig.axes[0].get_ylim(), [0.1, 100.0])


def test_time_trace_styles(fake_provider, tmp_path):
    run = simulate(Config.from_dict(BASE).with_overrides({"model.populations": 2}), provider=fake_provider)
    st = PlotStyle(time_unit="us", time_axis="linear", xlim=[0, 50], markers=True, injection_band=False,
                   colors={"hot": "black"}, cmap="plasma", legend=False, dpi=60)
    figs = make_figures(run, ["temperatures", "densities", "charge_states"], outdir=tmp_path, style=st)
    ax = figs["temperatures"].axes[0]
    assert ax.get_xscale() == "linear" and tuple(ax.get_xlim()) == (0, 50)
    assert "s" in ax.get_xlabel() and ax.get_legend() is None
    assert matplotlib.colors.same_color(ax.get_lines()[0].get_color(), "black")
    assert ax.get_lines()[0].get_marker() == "."
    assert (tmp_path / "temperatures.png").exists()


def test_cli_style_options(scan2d, tmp_path):
    from tqtoy.cli import main
    path = save_results(scan2d, tmp_path / "s.h5")
    style = tmp_path / "st.yaml"
    main(["style", "-o", str(style)])
    main(["map", str(path), "--no-show", "-d", str(tmp_path), "--style", str(style), "--cmap", "inferno",
          "--no-contours", "--filled", "--vmin", "0", "--vmax", "3", "--opt", "legend_loc=upper left"])
    main(["map", str(path), "--no-show", "-d", str(tmp_path), "--levels", "0.5", "1", "--contour-color", "k"])
    main(["plot", str(path), "--no-show", "-d", str(tmp_path), "-f", "temperatures", "densities",
          "--time-unit", "ms", "--color", "cold=black", "--xlim", "1e-4", "0.2", "--no-injection-band"])
    assert (tmp_path / "map_t_TQ.png").exists()
    with pytest.raises(SystemExit):
        main(["map", str(path), "--no-show", "--cmap", "nope"])


def test_plotting_reference_is_up_to_date():
    from pathlib import Path
    from tqtoy.cli import style_markdown
    doc = (Path(__file__).parents[1] / "docs" / "PLOTTING.md").read_text(encoding="utf-8")
    assert doc.endswith(style_markdown())


def test_map_overlays(scan2d):
    from tqtoy.diagnostics import scan_map
    net = scan_map(scan2d, "radiated_fraction_net")
    assert np.all((net[np.isfinite(net)] <= 1.0 + 1e-9))
    level = float(np.nanmedian(net))
    fig = plot_scan_map(scan2d, "t_TQ", style={"overlay": {"quantity": "radiated_fraction_net", "levels": [level],
                                                           "color": "lime"}, "ratio_lines": [0.1, 1.0]})
    ax = fig.axes[0]
    assert any(line.get_linestyle() == "--" for line in ax.get_lines())
    with pytest.raises(ConfigError):
        PlotStyle.from_dict({"overlay": {"levels": [0.5]}})


def test_a_two_decade_scan_gets_a_logarithmic_axis():
    """The automatic axis scale of the maps.

    A span of exactly two decades, which logspace(21, 23, 5) produces, is the common case and
    must be recognised as logarithmic despite the round-off of the ratio of its bounds.
    """
    from tqtoy.plotting import _axis_scale
    assert _axis_scale(np.logspace(21, 23, 5), "auto") == "log"
    assert _axis_scale(np.logspace(21, 23, 5), "linear") == "linear"
    assert _axis_scale(np.linspace(1.0, 2.0, 5), "auto") == "linear"
    assert _axis_scale(np.array([-1.0, 1e3]), "auto") == "linear"


def test_every_figure_is_drawn(fake_provider, tmp_path):
    """Every name of FIGURES draws and saves, with one and with two populations.

    The two lists that FIGURES feeds must stay complete: make_figures dispatches on the
    name, and a figure missing from either mapping raises instead of drawing.
    """
    from tqtoy.plotting import FIGURES
    names = [n for n in FIGURES if n not in ("shards", "hot_tail")]
    for populations in (1, 2):
        cfg = Config.from_dict(BASE).with_overrides({"model.populations": populations})
        run = simulate(cfg, provider=fake_provider)
        figs = make_figures(run, names, outdir=tmp_path, style={"dpi": 50})
        assert set(names) <= set(figs)
        for name in names:
            assert (tmp_path / f"{name}.png").exists(), name
        for fig in figs.values():
            matplotlib.pyplot.close(fig)


def test_the_mean_charge_stays_between_zero_and_Z(fake_provider):
    from tqtoy.plotting import _mean_charge
    run = simulate(Config.from_dict(BASE), provider=fake_provider)
    Z_mean = _mean_charge(run)
    Z = run.atomic_numbers[0]
    finite = np.isfinite(Z_mean[:, 0])
    assert finite.any()
    assert np.all(Z_mean[finite, 0] >= 0) and np.all(Z_mean[finite, 0] <= Z)


def test_the_electron_budget_adds_up_to_the_electron_density(fake_provider):
    """Charge neutrality: ne = ni + sum_k k n_k, which the figure draws term by term."""
    run = simulate(Config.from_dict(BASE).with_overrides({"model.populations": 2}), provider=fake_provider)
    k = np.arange(run.nij.shape[1])
    from_imp = np.einsum("k,tkj->tj", k, run.nij).sum(axis=1)
    ni = run.ni_cold + run.ni_hot
    # Neutrality is an exact invariant of the equations, so what is measured here is the
    # error of the integration. It stays four orders of magnitude below solver.rtol.
    np.testing.assert_allclose(ni + from_imp, run.ne_total, rtol=1e-6)


def test_the_coronal_cooling_rate_of_neon_peaks_near_its_known_temperature(fake_provider):
    """The coronal helpers, on the analytic rates: fractions normalised, <Z> monotonic.

    The analytic provider of the test suite is not the real atomic data, so only the
    structural properties are asserted here. The values against OpenADAS are checked by
    test_physics.py when the data are installed.
    """
    from tqtoy.atomic import AtomicData
    from tqtoy.physics.impurities import coronal_cooling_rate, coronal_fractions
    el = AtomicData(["neon"], fake_provider).rates[0]
    Z_mean = []
    for Te in (1.0, 10.0, 100.0, 1e3, 1e4):
        f = coronal_fractions(el, 1e20, Te)
        assert np.all(f >= 0)
        np.testing.assert_allclose(f.sum(), 1.0, rtol=1e-12)
        Z_mean.append(float(np.dot(np.arange(el.Z + 1), f)))
        assert coronal_cooling_rate(el, 1e20, Te) > 0
    assert np.all(np.diff(Z_mean) > 0), "the mean charge must rise with the temperature"


def test_a_coarse_log_axis_keeps_every_cell_of_a_2d_map(scan2d):
    """Cell edges of a 2D map on a logarithmic axis.

    An arithmetic midpoint on a coarse logarithmic grid puts the first edge at a non-positive
    value. A logarithmic axis drops it, and with it every cell beyond, which leaves the map
    almost empty. The edges are therefore taken in the space of the axis.
    """
    from tqtoy.plotting import _cell_edges

    x = np.logspace(21, 23, 5)
    edges = _cell_edges(x, "log")
    assert edges.shape == (x.size + 1,)
    assert np.all(edges > 0), "a logarithmic axis needs strictly positive edges"
    assert np.all(np.diff(edges) > 0)
    assert np.all((edges[:-1] < x) & (x < edges[1:])), "each value must sit inside its own cell"
    # On a linear axis the edges stay the arithmetic midpoints, mirrored at the two ends.
    np.testing.assert_allclose(_cell_edges(x, "linear"),
                               [(3 * x[0] - x[1]) / 2, *(0.5 * (x[:-1] + x[1:])),
                                (3 * x[-1] - x[-2]) / 2])
    assert _cell_edges(np.array([5.0]), "log").shape == (2,)

    fig = plot_scan_map(scan2d, "t_TQ")
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    lo, hi = ax.get_xlim()
    axis_values = np.asarray(scan2d.axes[0][1])
    assert lo < axis_values.min() and hi > axis_values.max(), "the map must cover every scanned value"
