"""Input files of the commands (tqtoy.tasks) and the figure commands driven by them."""

import os
from pathlib import Path

import pytest
import yaml

from tqtoy.cli import TEMPLATES, _template_text, main
from tqtoy.config import ConfigError
from tqtoy.tasks import MapTask, PlotTask, from_dict, task_table, tasks_text, with_overrides

ROOT = Path(__file__).parents[1]


def test_defaults():
    task = from_dict(PlotTask, {})
    assert task.compare is None and task.format == "png" and task.show is True
    assert from_dict(MapTask, {}).quantity == "t_TQ"


def test_unknown_entry_is_rejected():
    with pytest.raises(ConfigError):
        from_dict(PlotTask, {"nope": 1})


@pytest.mark.parametrize("name", ["plot", "map"])
def test_template_matches_the_dataclass(name, tmp_path):
    """Every entry of the commented template must be a real entry of the task."""
    cls = PlotTask if name == "plot" else MapTask
    data = yaml.safe_load(_template_text(name))
    task = from_dict(cls, data)
    assert task.results.endswith(".h5")


@pytest.mark.parametrize("name", TEMPLATES)
def test_init_writes_a_template(name, tmp_path, capsys):
    path = tmp_path / "in.yaml"
    main(["init", str(path), "-t", name])
    assert path.read_text(encoding="utf-8") == _template_text(name)


def test_overrides():
    task = with_overrides(from_dict(PlotTask, {}), {
        "compare.variable": "ne_total", "compare.axis": "n_D", "style.cmap": "magma",
        "at.n_D": 1e22, "figures": ["powers"], "show": False})
    assert task.compare.variable == "ne_total" and task.compare.axis == "n_D"
    assert task.style == {"cmap": "magma"} and task.at == {"n_D": 1e22}
    assert task.figures == ["powers"] and task.show is False


def test_override_of_an_unknown_entry_is_rejected():
    task = from_dict(MapTask, {})
    for path in ("nope", "compare.variable", "style_file.x"):
        with pytest.raises(ConfigError):
            with_overrides(task, {path: 1})


def test_nested_style_override():
    task = with_overrides(from_dict(MapTask, {}), {"style.overlay.quantity": "radiated_fraction"})
    assert task.style["overlay"]["quantity"] == "radiated_fraction"


def test_a_scalar_entry_is_not_a_section():
    task = from_dict(MapTask, {"style": {"cmap": "jet"}})
    for path in ("style.cmap.x", "at."):
        with pytest.raises(ConfigError):
            with_overrides(task, {path: 1})


@pytest.mark.parametrize("value", ["all", 5, {"a": 1}])
def test_a_list_entry_must_be_a_list(value):
    with pytest.raises(ConfigError):
        from_dict(PlotTask, {"figures": value})


def test_task_table_is_documented():
    for cls in (PlotTask, MapTask):
        rows = task_table(cls)
        assert rows and all(r["help"] for r in rows)
    assert any(r["path"] == "compare.variable" for r in task_table(PlotTask))


def test_tasks_text_covers_every_command():
    text = tasks_text()
    for command in ("tqtoy run", "tqtoy plot", "tqtoy map", "tqtoy fhte"):
        assert command in text
    with pytest.raises(ConfigError):
        tasks_text("nope")


def test_input_files_reference_is_up_to_date():
    """docs/INPUT_FILES.md must be regenerated when an entry changes (see tqtoy tasks --markdown)."""
    doc = (ROOT / "docs" / "INPUT_FILES.md").read_text(encoding="utf-8")
    assert doc.endswith(tasks_text(("plot", "map", "fhte"), markdown=True))


def test_fhte_input_file_selects_the_mode():
    from tqtoy.fhte import FHTERunConfig
    assert FHTERunConfig.from_dict({}).results is None
    cfg = FHTERunConfig.from_dict({"results": "a.h5", "jobs": 4, "fhte": {"n_output": 5}})
    assert (cfg.results, cfg.jobs, cfg.fhte.n_output, cfg.output) == ("a.h5", 4, 5, None)
    with pytest.raises(ConfigError):
        FHTERunConfig.from_dict({"jobs": 0})


# --------------------------------------------------------------------------- commands

@pytest.fixture()
def results_file(tmp_path, fake_provider):
    """Small 1D scan written to an HDF5 file."""
    from tqtoy.config import Config
    from tqtoy.io import save_results
    from tqtoy.scan import run_scan
    cfg = Config.from_dict({
        "time": {"t_end": 1e-4, "n_output": 20},
        "scan": {"impurities.neon.injected_atom_density": [1e18, 1e19, 1e20]},
    })
    path = tmp_path / "scan.h5"
    save_results(run_scan(cfg, provider=fake_provider, progress=False), path)
    return path


def test_plot_from_an_input_file(tmp_path, results_file):
    task = tmp_path / "plot.yaml"
    task.write_text(yaml.safe_dump({
        "results": str(results_file), "figures": ["temperatures"], "outdir": str(tmp_path / "figs"),
        "show": False, "style": {"time_unit": "us", "cmap": "magma"}}), encoding="utf-8")
    main(["plot", str(task)])
    assert (tmp_path / "figs" / "point_0_temperatures.png").exists()


def test_plot_comparison_of_a_1d_scan(tmp_path, results_file):
    task = tmp_path / "plot.yaml"
    task.write_text(yaml.safe_dump({
        "results": str(results_file), "compare": {"variable": "Te_avg"},
        "outdir": str(tmp_path / "figs"), "show": False}), encoding="utf-8")
    main(["plot", str(task)])
    assert (tmp_path / "figs" / "compare_Te_avg.png").exists()
    main(["plot", str(task), "--set", "compare.variable=ne_total", "-d", str(tmp_path / "f2")])
    assert (tmp_path / "f2" / "compare_ne_total.png").exists()


def test_plot_accepts_a_results_file_directly(tmp_path, results_file):
    main(["plot", str(results_file), "-f", "temperatures", "-d", str(tmp_path / "figs"), "--no-show"])
    assert (tmp_path / "figs" / "point_0_temperatures.png").exists()


@pytest.mark.parametrize("axis", ["injected_atom_density", "impurities.neon.injected_atom_density"])
def test_at_accepts_a_dotted_axis_name(tmp_path, results_file, axis):
    """Scan axes are named by dotted paths: --at must not split them into sections."""
    main(["plot", str(results_file), "--at", f"{axis}=1e20", "-f", "temperatures",
          "-d", str(tmp_path / "figs"), "--no-show"])
    assert (tmp_path / "figs" / "point_2_temperatures.png").exists()


def test_at_accepts_yaml_numbers_written_without_a_sign(tmp_path, results_file):
    """YAML 1.1 reads 1.0e20 (unsigned exponent) as a string: the axis value must still match."""
    from tqtoy.io import load_results
    res = load_results(results_file)
    assert res.index({"injected_atom_density": "1.0e20"}) == res.index({"injected_atom_density": 1e20}) == (2,)


def test_at_selects_the_points_of_a_comparison(tmp_path, results_file):
    task = tmp_path / "plot.yaml"
    task.write_text(yaml.safe_dump({"results": str(results_file), "compare": {"variable": "Te_avg"}}),
                    encoding="utf-8")
    from tqtoy.cli import _load_task
    from tqtoy.tasks import PlotTask
    args = type("A", (), {"input": str(task), "at": ["impurities.neon.injected_atom_density=1e20"],
                          "no_show": True, "set": None})()
    loaded = _load_task(PlotTask, args)
    assert loaded.compare.at == {"impurities.neon.injected_atom_density": 1e20} and loaded.at == {}


def test_map_from_an_input_file(tmp_path, results_file):
    task = tmp_path / "map.yaml"
    task.write_text(yaml.safe_dump({
        "results": str(results_file), "quantity": "t_TQ", "outdir": str(tmp_path / "figs"),
        "show": False, "style": {"cmap": "jet"}}), encoding="utf-8")
    main(["map", str(task)])
    assert (tmp_path / "figs" / "map_t_TQ.png").exists()
    main(["map", str(task), "--set", "quantity=Te_avg_final", "-d", str(tmp_path / "f2")])
    assert (tmp_path / "f2" / "map_Te_avg_final.png").exists()


def test_missing_results_file_is_reported(tmp_path):
    task = tmp_path / "map.yaml"
    task.write_text("results: nowhere.h5\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="no such file"):
        main(["map", str(task)])


def test_fhte_input_file_on_results(tmp_path, results_file, fake_provider):
    from tqtoy.io import load_results
    task = tmp_path / "fhte.yaml"
    task.write_text(yaml.safe_dump({
        "results": str(results_file), "fhte": {"n_output": 3, "n_trajectory": 4}}), encoding="utf-8")
    main(["fhte", str(task), "-q"])
    res = load_results(results_file)
    assert res.fhte is not None and res.fhte["fraction"].shape[-1] == 3


def test_tabulations_in_an_input_file_are_accepted(tmp_path, results_file):
    """Editors insert tabulations, which YAML forbids in the indentation: they are expanded."""
    task = tmp_path / "map_input.yaml"
    task.write_text(f"results: {results_file}\nquantity: t_TQ\nshow: false\nstyle:\n\tcmap: magma\n\tlog: true\n",
                    encoding="utf-8")
    main(["map", str(task), "-d", str(tmp_path / "figs")])
    assert (tmp_path / "figs" / "map_t_TQ.png").exists()


def test_a_yaml_syntax_error_names_its_position(tmp_path):
    task = tmp_path / "map_input.yaml"
    task.write_text("results: a.h5\n  quantity: t_TQ\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="line 2"):
        main(["map", str(task)])


def test_init_writes_the_style_file_next_to_a_figure_input_file(tmp_path, monkeypatch):
    from tqtoy.cli import STYLE_FILE
    from tqtoy.plotting import PlotStyle
    monkeypatch.chdir(tmp_path)
    main(["init", "-t", "plot"])
    assert (tmp_path / "plot_input.yaml").exists() and (tmp_path / STYLE_FILE).exists()
    assert PlotStyle.from_yaml(tmp_path / STYLE_FILE) == PlotStyle()
    assert from_dict(PlotTask, yaml.safe_load((tmp_path / "plot_input.yaml").read_text())).style_file == STYLE_FILE


def test_math_labels():
    """Scan axes and mapped quantities are labelled with their symbol, in math mode."""
    from tqtoy.plotting import _axis_label, _axis_symbol
    assert _axis_label("injection.n_D") == "$n_D$ [m$^{-3}$]"
    assert _axis_label("impurities.neon.injected_atom_density") == r"$n_{\rm Ne}$ [m$^{-3}$]"
    assert _axis_symbol("plasma.Te0") == "$T_{e0}$"
    assert _axis_label("solver.rtol") == "rtol"          # no symbol defined: the parameter name


def test_relative_paths_are_resolved_next_to_the_input_file(tmp_path, results_file):
    """An input file and the files it names travel together, whatever the working directory."""
    import os
    import re
    folder = tmp_path / "inputs"
    folder.mkdir()
    main(["init", str(folder / "map_input.yaml"), "-t", "map"])
    text = (folder / "map_input.yaml").read_text(encoding="utf-8")
    text, n = re.subn(r"^results:.*$", f"results: {os.path.relpath(results_file, folder)}", text,
                      count=1, flags=re.MULTILINE)
    assert n == 1, "the map template no longer has a 'results' entry"
    (folder / "map_input.yaml").write_text(text, encoding="utf-8")
    cwd = Path.cwd()
    os.chdir(tmp_path.parent)                    # neither the input nor the style file is here
    try:
        main(["map", str(folder / "map_input.yaml"), "--no-show", "-d", str(tmp_path / "figs")])
    finally:
        os.chdir(cwd)
    assert (tmp_path / "figs" / "map_t_TQ.png").exists()


def test_init_never_overwrites_an_edited_style_file(tmp_path, monkeypatch):
    from tqtoy.cli import STYLE_FILE
    monkeypatch.chdir(tmp_path)
    main(["init", "-t", "plot"])
    (tmp_path / STYLE_FILE).write_text("cmap: magma\n", encoding="utf-8")
    main(["init", "-t", "plot", "--force"])      # --force is about the input file
    assert (tmp_path / STYLE_FILE).read_text(encoding="utf-8") == "cmap: magma\n"


def test_a_tabulation_inside_a_value_is_kept():
    from tqtoy.config import parse_yaml
    assert parse_yaml('title: "a\tb"') == {"title": "a\tb"}
    assert parse_yaml("a:\n\tb: 1\n") == {"a": {"b": 1}}


# --------------------------------------------------------------------------- banner

def test_the_banner_draws_both_wordmarks():
    from tqtoy.banner import HEIGHT, wordmark, width
    for text in ("TQTOY", "TQTOY-FHTE"):
        lines = wordmark(text).splitlines()
        assert len(lines) == HEIGHT
        assert len({len(line) for line in lines}) == 1, "the rows of a wordmark must align"
        assert len(lines[0]) == width(text)
    assert width("TQTOY-FHTE") <= 80, "the wordmark must fit in a standard terminal"


def test_the_banner_falls_back_and_can_be_switched_off(monkeypatch, capsys):
    import io

    from tqtoy.banner import print_banner
    out = io.StringIO()
    assert print_banner("TQTOY", "subtitle", out) is True
    assert "subtitle" in out.getvalue()

    monkeypatch.setattr("shutil.get_terminal_size", lambda default=None: os.terminal_size((20, 24)))
    narrow = io.StringIO()
    assert print_banner("TQTOY", "subtitle", narrow) is False
    assert narrow.getvalue().strip() == "TQTOY subtitle"

    monkeypatch.setenv("TQTOY_NO_BANNER", "1")
    quiet = io.StringIO()
    assert print_banner("TQTOY", "subtitle", quiet) is False
    assert quiet.getvalue() == ""


def test_a_run_prints_the_banner_once_and_honours_quiet(tmp_path, monkeypatch, capsys, fake_provider):
    """The banner goes to the error stream, so a redirected standard output stays clean."""
    import tqtoy.scan
    real_run_scan = tqtoy.scan.run_scan            # captured before the patch, not after
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("tqtoy.scan.run_scan",
                        lambda cfg, **kw: real_run_scan(cfg, provider=fake_provider, progress=False))
    main(["init", "-t", "single_run"])
    main(["run", "single_run_input.yaml", "--set", "time.n_output=5", "time.t_end=1e-5"])
    from tqtoy.banner import wordmark
    art = wordmark("TQTOY")
    err = capsys.readouterr().err
    assert err.count(art) == 1
    assert (tmp_path / "results" / "tqtoy_results.h5").exists(), "results go to results/ by default"

    main(["run", "single_run_input.yaml", "-q", "--set", "time.n_output=5", "time.t_end=1e-5"])
    assert art not in capsys.readouterr().err
