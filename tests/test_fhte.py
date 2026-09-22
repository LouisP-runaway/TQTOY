"""Fast Hot Tail Estimator (no ADAS needed except where stated)."""

from pathlib import Path

import numpy as np
import pytest

from tqtoy import Config, simulate
from tqtoy.config import ConfigError, FHTEConfig
from tqtoy.fhte import (FHTERunConfig, build_traces, critical_momentum, hot_tail_estimate, hot_tail_from_run,
                        load_prescribed, run_prescribed, runaway_fraction, save_prescribed)

REF = np.load(Path(__file__).parent / "data" / "fhte_reference.npz")


@pytest.mark.parametrize("case", ["exponential", "chandrasekhar", "tq_run"])
def test_the_estimator_matches_its_reference(case):
    """Pinned reference values of the estimator, reproduced bit for bit on the same machine.

    Across machines, the backward integration of the momentum equation (rtol = 1e-5) amplifies
    round-off differences: on the tq_run case, whose temperature trace comes from a TQ run and
    drops by three decades, the runaway fraction of the last evaluation time was observed to
    move by 4e-4 in relative value after a change of CPU. The comparison is therefore tolerant
    on the fraction of that case.
    """
    g = {k.split("__")[1]: REF[k] for k in REF.files if k.startswith(case + "__")}
    r = hot_tail_estimate(g["t"], g["Te"], g["ne"], J=g["J"], n_output=int(g["n_output"]),
                          n_trajectory=int(g["n_trajectory"]), chandrasekhar=bool(g["chandrasekhar"]),
                          output_window="full")     # evaluation times over the whole run
    rtol = 1e-3 if case == "tq_run" else 1e-6
    np.testing.assert_array_equal(r.t_output, g["t_output"])
    np.testing.assert_allclose(r.fraction, g["fraction"], rtol=rtol, atol=0)
    np.testing.assert_allclose(r.p_c, g["p_c"], rtol=1e-9)
    np.testing.assert_allclose(r.p_trajectory, g["p_trajectory"], rtol=rtol)


def test_no_runaway_below_critical_field():
    t = np.linspace(0, 1e-3, 50)
    r = hot_tail_estimate(t, 5e3, 1e20, E=0.0, n_output=4)
    assert np.all(r.fraction == 0) and np.all(r.p_c == 100)
    assert critical_momentum(0.01, 1e20, 1e3, 2.0) == 100


def test_prescribed_field_and_monotonic_fraction():
    t = np.linspace(1e-8, 1e-3, 100)
    Te = 20 + 5e3 * np.exp(-t / 1e-4)
    weak = hot_tail_estimate(t, Te, 1e20, E=0.5, n_output=3).final_fraction
    strong = hot_tail_estimate(t, Te, 1e20, E=2.0, n_output=3).final_fraction
    assert 0 < weak < strong < 1
    # E >> E_c: the backward trajectory reaches p = 0, every electron runs away
    extreme = hot_tail_estimate(t, Te, 1e20, E=50.0, n_output=3)
    assert extreme.final_fraction == pytest.approx(1.0) and "p = 0" in extreme.message


def test_runaway_fraction_limits():
    assert runaway_fraction(0.0, 1e3) == pytest.approx(1.0)
    assert runaway_fraction(10.0, 1e3) == 0.0
    assert runaway_fraction(0.2, 1e3) < runaway_fraction(0.2, 5e3)


def test_invalid_options():
    t = np.linspace(0, 1e-3, 10)
    with pytest.raises(ValueError):
        hot_tail_estimate(t, 1e3, 1e20)                           # neither J nor E
    with pytest.raises(ConfigError):
        hot_tail_estimate(t, 1e3, 1e20, J=1e6, field="other")
    with pytest.raises(ConfigError):
        FHTEConfig(temperature="hot").validate(two_populations=False)


def test_prescribed_traces(tmp_path):
    csv = tmp_path / "tr.csv"
    t = np.linspace(0, 1e-3, 11)
    np.savetxt(csv, np.c_[t, 1e3 * np.ones_like(t), 2e19 + t * 1e21], delimiter=",", header="t,Te,ne", comments="")
    cfg = FHTERunConfig.from_dict({"prescribed": {"file": str(csv), "Te": {"column": "Te"}, "ne": {"column": "ne"},
                                                  "J": {"linear": {"initial": 1e6, "final": 5e5}}}})
    tr = build_traces(cfg.prescribed)
    np.testing.assert_allclose(tr["t"], t)
    np.testing.assert_allclose(tr["J"][[0, -1]], [1e6, 5e5])
    with pytest.raises(ConfigError, match="values for"):
        build_traces(FHTERunConfig.from_dict({"prescribed": {"Te": [1.0, 2.0]}}).prescribed)


def test_standalone_round_trip(tmp_path):
    cfg = FHTERunConfig.from_dict({"fhte": {"n_output": 5, "n_trajectory": 4}})
    tr, res = run_prescribed(cfg)
    path = save_prescribed(tmp_path / "f.h5", cfg, tr, res)
    cfg2, tr2, res2 = load_prescribed(path)
    np.testing.assert_array_equal(res2.fraction, res.fraction)
    assert cfg2.fhte.n_output == 5 and res2.T0 == res.T0


@pytest.mark.parametrize("over", [{}, {"fhte.field": "model"}, {"fhte.temperature": "cold"},
                                  {"fhte.bound_electron_weight": 0.5}, {"plasma.J_final": 1e5, "plasma.J_ramp_time": 1e-4},
                                  {"model.populations": 2, "fhte.temperature": "hot"}])
def test_from_run(fake_provider, over):
    cfg = Config.from_dict({"plasma": {"Te0": 3e3, "ne0": 5e19}, "impurities": {"neon": {"injected_atom_density": 1e19}},
                            "injection": {"n_D": 3e20}, "time": {"t_end": 2e-4, "n_output": 40}}).with_overrides(over)
    run = simulate(cfg, provider=fake_provider)
    from tqtoy.atomic import get_atomic_data
    r = hot_tail_from_run(run, atomic=get_atomic_data(["neon"], fake_provider))
    assert r.status == 0 and np.isfinite(r.final_fraction) and 0 <= r.final_fraction <= 1


def test_scan_integration(fake_provider, tmp_path):
    from tqtoy.diagnostics import scan_map, summary
    from tqtoy.io import load_results, save_fhte, save_results
    from tqtoy.scan import compute_fhte, run_scan
    base = {"plasma": {"Te0": 3e3, "ne0": 5e19}, "impurities": {"neon": {"injected_atom_density": 1e19}},
            "injection": {"n_D": 3e20}, "time": {"t_end": 2e-4, "n_output": 40},
            "scan": {"injection.n_D": [1e20, 1e21], "impurities.neon.injected_atom_density": [1e18, 1e19, 1e20]}}
    res = run_scan(Config.from_dict({**base, "fhte": {"enabled": True, "n_output": 3}}),
                   provider=fake_provider, progress=False)
    m = scan_map(res, "n_RE_hot_tail")
    assert m.shape == (2, 3) and np.all(np.isfinite(m))
    np.testing.assert_allclose(scan_map(res, "n_RE_hot_tail_density"), m * res.fhte["n0"])
    assert "n_RE_hot_tail_fraction" in summary(res.run((0, 0)))

    # post-hoc computation on results without FHTE, then stored in the file
    res2 = run_scan(Config.from_dict(base), provider=fake_provider, progress=False)
    assert res2.fhte is None
    path = save_results(res2, tmp_path / "s.h5")
    compute_fhte(res2, {"fhte.n_output": 3}, progress=False, provider=fake_provider)
    np.testing.assert_allclose(scan_map(res2, "n_RE_hot_tail"), m)      # same settings, same values
    save_fhte(res2, path)
    back = load_results(path)
    np.testing.assert_array_equal(back.fhte["fraction"], res2.fhte["fraction"])
    assert back.run((1, 2)).config.fhte.n_output == 3
    with pytest.raises(ConfigError):
        compute_fhte(back, {"plasma.Te0": 1.0}, progress=False)


def test_cli(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    from tqtoy.cli import main
    monkeypatch.chdir(tmp_path)
    main(["init", "-t", "fhte_prescribed"])
    assert (tmp_path / "fhte_prescribed_input.yaml").exists()      # input files are named *_input.yaml
    main(["fhte", "fhte_prescribed_input.yaml", "-o", "a.h5", "--set", "fhte.n_output=4", "prescribed.ne=5e19"])
    main(["summary", "a.h5"])
    main(["plot", "a.h5", "--no-show", "-d", "figs"])
    assert (tmp_path / "figs" / "hot_tail.png").exists()
    _, _, res = load_prescribed("a.h5")
    assert len(res.t_output) == 4 and res.n0 == 5e19


# --------------------------------------------------------------------------- evaluation window

def _crossing_traces():
    """A field pulse: E > E_c only between 20 and 40 us, below it before and after."""
    t = np.linspace(1e-8, 1e-3, 400)
    Te = 20.0 + 5e3 * np.exp(-t / 2e-5)              # fast quench, then a cold plasma
    E = np.where((t >= 2e-5) & (t <= 4e-5), 6.0, 1e-3)
    return t, Te, np.full(t.shape, 1e20), E


def test_evaluation_times_land_where_runaways_are_created():
    t, Te, ne, E = _crossing_traces()
    full = hot_tail_estimate(t, Te, ne, E=E, n_output=12, n_trajectory=8,
                             output_window="full", accumulate=False)
    active = hot_tail_estimate(t, Te, ne, E=E, n_output=12, n_trajectory=8)
    assert np.sum(full.p_c[1:] < 100) < np.sum(active.p_c[1:] < 100)
    # every interior evaluation time of the active window is a time where E > E_c
    assert np.all(active.p_c[1:-1] < 100)
    # the window is covered, and the first and last times of the run are kept
    assert active.t_output[0] == t[0] and active.t_output[-1] == t[-1]
    assert len(active.t_output) == 12


def test_no_integration_where_the_field_is_below_the_critical_field():
    t = np.linspace(1e-8, 1e-3, 200)
    res = hot_tail_estimate(t, 5.0, 1e20, J=1e3, n_output=8, n_trajectory=5)   # E << E_c everywhere
    assert np.all(res.fraction == 0)
    assert np.all(~np.isfinite(res.p_limit))         # no backward trajectory was integrated
    assert np.all(res.p_c == 100)


def test_the_runaway_population_is_kept_when_the_field_falls_back():
    t, Te, ne, E = _crossing_traces()
    res = hot_tail_estimate(t, Te, ne, E=E, n_output=14, n_trajectory=8)
    assert res.final_fraction > 0
    assert np.all(np.diff(res.fraction) >= 0)        # never decreases
    instant = hot_tail_estimate(t, Te, ne, E=E, n_output=14, n_trajectory=8, accumulate=False)
    assert instant.final_fraction <= res.final_fraction
    assert res.final_fraction == pytest.approx(np.nanmax(instant.fraction))


def test_accumulate_does_not_change_a_monotonic_case():
    t = np.linspace(1e-8, 1e-3, 200)
    Te = 20 + 5e3 * np.exp(-t / 1e-4)
    a = hot_tail_estimate(t, Te, 1e20, E=2.0, n_output=5, n_trajectory=5)
    b = hot_tail_estimate(t, Te, 1e20, E=2.0, n_output=5, n_trajectory=5, accumulate=False)
    np.testing.assert_array_equal(a.fraction, b.fraction)


def test_scalar_helpers_match_the_reference_implementation():
    """The cached Maxwell-Juttner tail must reproduce the direct integration."""
    from scipy.integrate import trapezoid
    from tqtoy.fhte import C, CHARGE, ME, runaway_fraction

    def direct(p_limit, T0, gamma_max=4.0, n_points=200000):
        gamma = np.linspace(1, gamma_max, n_points)
        theta = T0 * CHARGE / (ME * C**2)
        beta = np.sqrt(1.0 - 1.0/gamma**2)
        with np.errstate(divide="ignore"):
            ln_f = 2*np.log(gamma) + np.log(beta) - gamma/theta
        f = np.exp(ln_f - np.max(ln_f))
        above = gamma >= np.sqrt(1+p_limit**2)
        return trapezoid(y=f[above], x=gamma[above]) / trapezoid(f, gamma)

    for T0 in (100.0, 3.1e3, 2e4):
        for p in (0.0, 0.573, 1.5, 3.0):
            assert runaway_fraction(p, T0) == pytest.approx(direct(p, T0), rel=1e-12)


def test_every_runaway_window_receives_an_evaluation_time():
    """Two separate windows: the estimate must not miss one of them."""
    t = np.linspace(1e-6, 1e-3, 600)
    Te = np.where(t < 2e-4, 3e3, 30.0)
    E = np.where(((t > 1.5e-4) & (t < 2.0e-4)) | ((t > 5e-4) & (t < 9e-4)), 6.0, 1e-3)
    reference = hot_tail_estimate(t, Te, 1e20, E=E, n_output=60, n_trajectory=5, output_window="full")
    for n in (3, 4, 6, 12):
        r = hot_tail_estimate(t, Te, 1e20, E=E, n_output=n, n_trajectory=5)
        assert r.final_fraction == pytest.approx(reference.final_fraction, rel=1e-9), n
        assert not r.message
    # with a single evaluation time for two windows, the omission is reported
    poor = hot_tail_estimate(t, Te, 1e20, E=E, n_output=2, n_trajectory=5)
    assert "no evaluation time" in poor.message


def test_a_failed_integration_is_not_carried_forward():
    from scipy.integrate import solve_ivp as real_solve_ivp
    import tqtoy.fhte as fhte_module
    t = np.linspace(1e-6, 1e-3, 600)
    Te = np.where(t < 2e-4, 3e3, 30.0)
    E = np.where((t > 5e-4) & (t < 7e-4), 6.0, 1e-3)
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        sol = real_solve_ivp(*args, **kwargs)
        if calls["n"] == 5:
            sol.success, sol.message = False, "forced failure"
        return sol

    fhte_module.solve_ivp = flaky
    try:
        res = hot_tail_estimate(t, Te, 1e20, E=E, n_output=8, n_trajectory=5)
    finally:
        fhte_module.solve_ivp = real_solve_ivp
    assert res.status == -1 and "forced failure" in res.message
    assert np.isfinite(res.final_fraction) and res.final_fraction > 0
    assert np.sum(~np.isfinite(res.fraction)) == 1          # only the failed time is undefined


def test_summary_reports_the_last_time_with_runaway_creation():
    from tqtoy.fhte import summary_fhte
    t = np.linspace(1e-6, 1e-3, 600)
    Te = np.where(t < 2e-4, 3e3, 30.0)
    E = np.where((t > 5e-4) & (t < 7e-4), 6.0, 1e-3)
    res = hot_tail_estimate(t, Te, 1e20, E=E, n_output=10, n_trajectory=5)
    s = summary_fhte(res)
    assert s["t_last_runaway"] < res.t_output[-1]           # the run ends after the window
    assert np.isfinite(s["p_limit_last_runaway"]) and s["E_over_Ec_last_runaway"] > 1
    assert s["p_c_last_runaway"] < 100


def test_the_fraction_never_exceeds_one():
    t = np.linspace(1e-8, 1e-3, 100)
    res = hot_tail_estimate(t, 3020.0, 1e20, E=80.0, n_output=4, n_trajectory=4)
    assert np.nanmax(res.fraction) <= 1.0
