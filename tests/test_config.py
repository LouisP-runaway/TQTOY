import numpy as np
import pytest

from tqtoy.cli import TEMPLATES, _template_text
from tqtoy.config import Config, ConfigError, parse_assignment, parse_values


def test_defaults_are_valid():
    cfg = Config()
    cfg.validate()
    assert cfg.element_names == ["neon"]
    assert cfg.output_times()[0] == cfg.injection.t_start


@pytest.mark.parametrize("name", ["single_run", "iter_scan"])
def test_templates_load(name):
    cfg = Config.from_yaml_string(_template_text(name))
    assert cfg.impurities


@pytest.mark.parametrize("name", ["fhte_prescribed", "fhte_results"])
def test_fhte_template_loads(name):
    import yaml
    from tqtoy.fhte import FHTERunConfig
    cfg = FHTERunConfig.from_dict(yaml.safe_load(_template_text(name)))
    assert cfg.fhte.n_output == 20
    assert (cfg.results is None) == (name == "fhte_prescribed")


def test_single_run_template_lists_default_values():
    """The fully commented template must stay in sync with the dataclass defaults."""
    cfg = Config.from_yaml_string(_template_text("single_run"))
    assert cfg.to_dict() == Config().to_dict()


def test_iter_template_matches_former_launcher():
    cfg = Config.from_yaml_string(_template_text("iter_scan"))
    axes = dict(cfg.scan_axes())
    np.testing.assert_array_equal(axes["injection.n_D"], np.logspace(np.log10(1e20), 23, 21))
    np.testing.assert_array_equal(axes["impurities.neon.injected_atom_density"], np.logspace(18, 21, 20))
    assert cfg.model.populations == 1 and cfg.solver.rtol == 1e-2


def test_unknown_key_is_reported():
    with pytest.raises(ConfigError, match="unknown key"):
        Config.from_dict({"plasma": {"Te": 1.0}})


def test_yaml_numbers_written_as_strings_are_accepted():
    cfg = Config.from_yaml_string("plasma: {Te0: 1e4, ne0: 5e19}")
    assert cfg.plasma.Te0 == 1e4 and cfg.plasma.ne0 == 5e19


@pytest.mark.parametrize("bad", [
    {"model": {"populations": 3}},
    {"model": {"resistivity": "multi_species"}},                 # needs 2 populations
    {"injection": {"source": "pellet"}},
    {"time": {"spacing": "log", "t_start": 0.0}},
    {"plasma": {"J_final": 1.0}},                                # J_ramp_time missing
    {"scan": {"plasma.Tx": [1, 2]}},
    {"scan": {"plasma.Te0": {"logspace": [1, 2]}}},
    {"plasma": {"Te0": -1.0}},
    {"injection": {"n_D": -1.0}},
    {"impurities": {"neon": {"injected_atom_density": -1.0}}},
    {"solver": {"method": "Euler"}},
    {"output": {"energy_refinement": 0}},
])
def test_invalid_configurations(bad):
    with pytest.raises(ConfigError):
        Config.from_dict(bad)


def test_scan_specs():
    np.testing.assert_allclose(parse_values({"geomspace": [1e18, 1e20, 3]}), [1e18, 1e19, 1e20])
    np.testing.assert_allclose(parse_values({"linspace": [0, 1, 3]}), [0, 0.5, 1])
    np.testing.assert_allclose(parse_values([1, "2e3"]), [1, 2e3])


def test_scan_points_order():
    cfg = Config.from_dict({"scan": {"injection.n_D": [1e21, 1e22], "plasma.Te0": [1e3, 2e3, 3e3]}})
    pts = list(cfg.scan_points())
    assert len(pts) == 6
    assert pts[1] == ((0, 1), {"injection.n_D": 1e21, "plasma.Te0": 2e3})


def test_overrides():
    cfg = Config().with_overrides({"injection.ablation.parks.n_atoms": 1e24,
                                   "impurities.argon.injected_atom_density": 1e18})
    assert cfg.injection.ablation.parks.n_atoms == 1e24
    assert cfg.element_names == ["neon", "argon"]
    cfg = Config().with_overrides({"scan.injection.n_D": [1e21, 1e22]})
    assert len(cfg.scan_axes()[0][1]) == 2
    with pytest.raises(ConfigError):
        Config().with_overrides({"plasma.nope": 1})


def test_parse_assignment():
    assert parse_assignment("plasma.Te0=1e4") == ("plasma.Te0", 1e4)
    assert parse_assignment("scan.plasma.Te0=[1, 2]") == ("scan.plasma.Te0", [1, 2])
    assert parse_assignment("model.stochastic.enabled=false") == ("model.stochastic.enabled", False)


def test_yaml_round_trip():
    cfg = Config.from_dict({"model": {"populations": 2}, "scan": {"plasma.Te0": {"linspace": [1e3, 2e3, 2]}}})
    assert Config.from_yaml_string(cfg.to_yaml()).to_dict() == cfg.to_dict()


@pytest.mark.parametrize("name", TEMPLATES)
def test_example_files_match_templates(name):
    from pathlib import Path
    example = Path(__file__).parents[1] / "examples" / f"{name}.yaml"
    assert example.read_text(encoding="utf-8") == _template_text(name), \
        "examples/ and src/tqtoy/templates/ must hold the same files"


def test_configuration_reference_is_up_to_date():
    """docs/INPUT_PARAMETERS.md must be regenerated when a parameter changes (tqtoy params --markdown)."""
    from pathlib import Path
    from tqtoy.cli import params_text
    doc = (Path(__file__).parents[1] / "docs" / "INPUT_PARAMETERS.md").read_text(encoding="utf-8")
    assert doc.endswith(params_text(markdown=True))


def test_every_parameter_is_documented():
    from tqtoy.config import parameter_table
    rows = parameter_table()
    assert all(r["help"] for r in rows)
    paths = {r["path"] for r in rows}
    assert {"plasma.Te0", "impurities.<element>.injected_atom_density", "injection.ablation.parks.n_atoms"} <= paths


def test_params_command(capsys):
    from tqtoy.cli import main
    main(["params", "Te0"])
    out = capsys.readouterr().out
    assert "plasma.Te0" in out and "eV" in out


def test_single_run_template_lists_every_parameter():
    import yaml
    data = yaml.safe_load(_template_text("single_run"))

    def has(path):
        node = data
        for key in path.replace("<element>", "neon").split("."):
            if not isinstance(node, dict) or key not in node:
                return False
            node = node[key]
        return True

    from tqtoy.config import parameter_table
    missing = [r["path"] for r in parameter_table()
               if not r["path"].startswith("injection.ablation.parks.") and not has(r["path"])]
    assert not missing, f"parameters missing from templates/single_run.yaml: {missing}"


def test_examples_hold_the_style_file():
    """examples/style_input.yaml is the file tqtoy init writes next to a plot or map input file."""
    from pathlib import Path
    from tqtoy.cli import STYLE_FILE
    from tqtoy.plotting import PlotStyle
    path = Path(__file__).parents[1] / "examples" / STYLE_FILE
    assert path.read_text(encoding="utf-8") == PlotStyle().to_yaml()
    assert PlotStyle.from_yaml(path) == PlotStyle()
