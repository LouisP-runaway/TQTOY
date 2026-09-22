"""Model tests with analytic atomic rates (no ADAS data needed)."""

import numpy as np
import pytest

from tqtoy import Config, simulate
from tqtoy.diagnostics import energy_balance, summary, thermal_quench_time
from tqtoy.io import load_results, save_results
from tqtoy.scan import run_scan
from tqtoy.solver import STATUS_COMPLETED

BASE = {"plasma": {"Te0": 5e3, "ne0": 5e19, "J": 1e6},
        "impurities": {"neon": {"injected_atom_density": 2e19}},
        "injection": {"n_D": 5e20},
        "time": {"t_end": 2e-3, "n_output": 120}}


def cfg(**over):
    c = Config.from_dict(BASE)
    return c.with_overrides(over) if over else c


def test_single_population_conservation(fake_provider):
    c = cfg()
    r = simulate(c, provider=fake_provider)
    assert r.status == STATUS_COMPLETED
    after = r.t > c.injection.t_start + c.injection.duration
    # Inventories are constant after the injection (linear invariants of the ODE system,
    # preserved up to the Newton convergence of the implicit solver).
    # The injected amount itself is accurate to the solver tolerance (discontinuous source).
    np.testing.assert_allclose(r.n_imp[after, 0], r.n_imp[after, 0][-1], rtol=1e-7)
    np.testing.assert_allclose(r.n_imp[after, 0], 2e19, rtol=c.solver.rtol)
    np.testing.assert_allclose(r.ni_cold[after], 5e19 + 5e20, rtol=c.solver.rtol)
    # charge neutrality: ne = ni + sum_i i n_i
    Zbar_n = np.sum(np.arange(11)[None, :] * r.nij[:, :, 0], axis=1)
    np.testing.assert_allclose(r.ne_cold, r.ni_cold + Zbar_n, rtol=1e-6)


def test_energy_conservation(fake_provider):
    r = simulate(cfg(), provider=fake_provider)
    eb = energy_balance(r)
    assert np.max(np.abs(eb["residual"])) / eb["W_th"][0] < 1e-3


def _neutrality(r):
    """ne / (ni + sum_i i n_ij), summed over both populations. Exactly 1 in the model."""
    Zbar_n = np.sum(np.arange(r.nij.shape[1])[None, :, None] * r.nij, axis=(1, 2))
    ni = r.ni_cold + (r.ni_hot if r.two_populations else 0.0)
    return r.ne_total / (ni + Zbar_n)


def test_two_population_conservation(fake_provider):
    """The injected particles are collected by the cold population and none is lost.

    The cold population starts at cold_n0 = 1e10 m^-3, so the deuterium source dilutes it to
    the temperature floor within a few femtoseconds. Only the temperature stops there. The
    density goes on collecting the injected deuterons and the electrons released by impurity
    ionisation.
    """
    c = cfg(**{"model.populations": 2})
    r = simulate(c, provider=fake_provider)
    assert r.status == STATUS_COMPLETED
    np.testing.assert_allclose(_neutrality(r), 1.0, rtol=1e-6)
    after = r.t > c.injection.t_start + c.injection.duration
    ni = r.ni_cold + r.ni_hot
    np.testing.assert_allclose(ni[after], 5e19 + 5e20, rtol=c.solver.rtol)
    # The cold population holds every injected deuteron, not a residue of them.
    assert r.ni_cold[-1] > 0.5 * 5e20


def test_no_particle_is_transferred_between_the_populations(fake_provider):
    """The two populations exchange energy and nothing else (docs/MODEL_EQUATIONS.pdf).

    The hot population is closed: it receives no source and gives nothing away, so its particle
    content is exactly its initial value for the whole run.
    """
    c = cfg(**{"model.populations": 2})
    r = simulate(c, provider=fake_provider)
    np.testing.assert_allclose(r.ne_hot, r.ne_hot[0], rtol=1e-12)
    np.testing.assert_allclose(r.ni_hot, r.ni_hot[0], rtol=1e-12)
    assert r.ne_hot[0] == pytest.approx(5e19)


def test_the_exchange_relaxes_the_two_electron_temperatures(fake_provider):
    """The collisional exchange is the only coupling, and it is restoring on both sides.

    d(Te_hot - Te_cold)/dt from the exchange is -2/(3e) P_ee (1/ne_hot + 1/ne_cold), with P_ee
    proportional to (Te_hot - Te_cold). The difference must therefore fall monotonically once
    the injection is over, whichever population is the hotter.
    """
    c = cfg(**{"model.populations": 2})
    r = simulate(c, provider=fake_provider)
    gap = np.abs(r.Te_hot - r.Te_cold) / np.maximum(r.Te_hot, r.Te_cold)
    assert gap[0] > 0.9, "the two populations start genuinely apart"
    assert gap[-1] < 1e-3, "the exchange brings them together"
    # and they stay together once the injection is over
    after = r.t > c.injection.t_start + c.injection.duration
    assert np.max(gap[after]) < 1e-2


@pytest.mark.parametrize("populations", [1, 2])
def test_the_two_state_formulations_are_the_same_equations(populations, fake_provider):
    """solver.state_variables: the energy form is the temperature form multiplied by n.

    They are integrated differently, so they differ by the local error the tolerance allows.
    A difference that falls with rtol is that error. A difference that does not would mean the
    two forms are not the same equations.
    """
    previous = None
    for rtol in (1e-2, 1e-4, 1e-6):
        runs = [simulate(cfg(**{"model.populations": populations, "solver.rtol": rtol,
                                "solver.atol": 1e-12, "solver.state_variables": s}),
                         provider=fake_provider) for s in ("temperature", "energy")]
        a, b = runs
        assert a.ok and b.ok
        difference = float(np.max(np.abs(b.Te_avg - a.Te_avg) / np.maximum(np.abs(a.Te_avg), 1e-3)))
        assert difference < 20 * rtol, f"rtol = {rtol:.0e}: {difference:.2e}"
        if previous is not None:
            assert difference < 0.5 * previous, "the difference must fall with rtol"
        previous = difference


@pytest.mark.parametrize("extra", [{}, {"injection.source": "ablation"},
                                   {"model.block_electron_ion_exchange": True}])
def test_the_two_population_options_are_inert_for_one_population(fake_provider, extra):
    """A single population never reaches the floor and has no current to share."""
    runs = [simulate(cfg(**dict(extra, **over)), provider=fake_provider) for over in (
        {}, {"model.current_sharing": "spitzer"})]
    for name in runs[0].fluid:
        np.testing.assert_array_equal(runs[0].fluid[name], runs[1].fluid[name])
    np.testing.assert_array_equal(runs[0].nij, runs[1].nij)


def test_two_elements(fake_provider):
    """Each element keeps its own inventory, with no leakage between their charge states."""
    c = cfg(**{"impurities.argon.injected_atom_density": 1e18})
    r = simulate(c, provider=fake_provider)
    assert r.nij.shape[1:] == (19, 2)
    after = r.t > c.injection.t_start + c.injection.duration
    np.testing.assert_allclose(r.n_imp[after, 0], 2e19, rtol=c.solver.rtol)
    np.testing.assert_allclose(r.n_imp[after, 1], 1e18, rtol=c.solver.rtol)
    assert np.all(r.nij[:, 11:, 0] == 0)       # neon has no state above 10+


@pytest.mark.parametrize("extra", [
    {"model.populations": 2},
    {"model.populations": 2, "model.current_sharing": "spitzer"},
    {"model.populations": 2, "model.resistivity": "multi_species"},
    {"model.block_electron_ion_exchange": True},
    {"plasma.J_final": 1e5, "plasma.J_ramp_time": 1e-3},
    {"injection.source": "ablation", "injection.ablation.D2_fraction": 0.9},
    {"injection.source": "ablation", "injection.ablation.parks.n_atoms": 1e24, "injection.ablation.seed": 3},
    {"impurities.neon.background_density": 1e17, "injection.charge_state": 1},
])
def test_options_run(fake_provider, extra):
    r = simulate(cfg(**extra), provider=fake_provider)
    assert r.ok and len(r.t) > 10
    assert np.all(np.isfinite(r.Te_cold))
    s = summary(r)
    assert np.isfinite(s["W_th_initial"])


def test_background_impurity_initial_state(fake_provider):
    c = cfg(**{"impurities.neon.background_density": 1e18, "injection.charge_state": 2})
    r = simulate(c, provider=fake_provider)
    assert r.nij[0, 2, 0] == pytest.approx(1e18)
    assert r.ne_cold[0] == pytest.approx(5e19 + 2e18)


def test_scan_and_io_round_trip(fake_provider, tmp_path):
    c = cfg(**{"scan.injection.n_D": [1e20, 1e21], "scan.impurities.neon.injected_atom_density": [1e18, 1e19, 1e20]})
    res = run_scan(c, provider=fake_provider, progress=False)
    assert res.shape == (2, 3)
    assert np.all(res.status == STATUS_COMPLETED)
    idx = res.index(n_D=9e20, injected_atom_density=1.2e19)
    assert idx == (1, 1)
    single = simulate(res.point_config(idx), provider=fake_provider, t_eval=res.t)
    np.testing.assert_array_equal(single.Te_cold, res.run(idx).Te_cold)

    path = save_results(res, tmp_path / "scan.h5")
    back = load_results(path)
    assert back.axis_names == res.axis_names
    np.testing.assert_array_equal(back.fluid["Te_cold"], res.fluid["Te_cold"])
    np.testing.assert_array_equal(back.nij, res.nij)
    np.testing.assert_array_equal(back.energies["E_rad"], res.energies["E_rad"])
    assert back.config.to_dict() == res.config.to_dict()
    r = back.run(idx)
    assert thermal_quench_time(r) == thermal_quench_time(res.run(idx))


def test_single_run_io(fake_provider, tmp_path):
    res = run_scan(cfg(), provider=fake_provider, progress=False)
    assert res.shape == ()
    back = load_results(save_results(res, tmp_path / "one.h5"))
    np.testing.assert_array_equal(back.run().Te_cold, res.run().Te_cold)


def test_plots(fake_provider, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from tqtoy.plotting import FIGURES, make_figures, plot_scan_map
    c = cfg(**{"model.populations": 2, "injection.source": "ablation", "injection.ablation.parks.n_atoms": 1e24,
               "injection.ablation.seed": 0})
    r = simulate(c, provider=fake_provider)
    from tqtoy.diagnostics import compute_terms
    terms = compute_terms(r, provider=fake_provider)
    figs = make_figures(r, list(FIGURES), outdir=tmp_path, terms=terms)
    assert (tmp_path / "powers.png").exists() and len(figs) >= len(FIGURES)
    res = run_scan(cfg(**{"scan.injection.n_D": [1e20, 1e21, 1e22]}), provider=fake_provider, progress=False)
    plot_scan_map(res, "t_TQ")


def test_scan_of_boolean_and_text_parameters(fake_provider, tmp_path):
    c = cfg(**{"model.populations": 2, "time.n_output": 40,
               "scan.model.stochastic.enabled": [True, False],
               "scan.model.resistivity": ["spitzer", "multi_species"]})
    res = run_scan(c, provider=fake_provider, progress=False)
    back = load_results(save_results(res, tmp_path / "s.h5"))
    idx = back.index({"enabled": False, "resistivity": "multi_species"})
    assert idx == (1, 1)
    run = back.run(idx)
    assert run.config.model.stochastic.enabled is False
    assert run.config.model.resistivity == "multi_species"
    assert np.isfinite(summary(run)["t_last"])


def test_scan_of_shard_number_with_parks(fake_provider):
    c = cfg(**{"injection.source": "ablation", "injection.ablation.parks.n_atoms": 1e24,
               "injection.ablation.seed": 0, "time.n_output": 40,
               "scan.injection.ablation.n_shards": [50, 100]})
    res = run_scan(c, provider=fake_provider, progress=False)
    assert res.r_p_samples.shape == (2, 100)
    assert res.run((0,)).r_p_samples.size == 50


def test_scan_axis_on_undefined_impurity_is_rejected(fake_provider):
    from tqtoy.config import ConfigError
    with pytest.raises(ConfigError, match="must also be defined"):
        run_scan(cfg(**{"scan.impurities.argon.injected_atom_density": [1e18]}), provider=fake_provider)


def test_empty_run_is_reported(fake_provider):
    import dataclasses
    r = simulate(cfg(**{"time.n_output": 20}), provider=fake_provider)
    empty = dataclasses.replace(r, t=r.t[:0], fluid={k: v[:0] for k, v in r.fluid.items()}, nij=r.nij[:0],
                                energies=None)
    with pytest.raises(ValueError, match="valid output time"):
        summary(empty)
