"""Analytic Jacobian of the model (solver.jacobian = analytic).

The impurity charge-state columns are differentiated analytically; the fluid columns are
finite differences. The reference here is a central finite difference of the same right-hand
side. Its step is relative and deliberately large (1e-3 of the component): the charge-state
balance is close to linear in those directions, so the truncation error stays small while a
smaller step would be dominated by round-off. With that step the reference is accurate to
about 1e-8 of the largest entry of the matrix.
"""

import numpy as np
import pytest

from tqtoy import Config, simulate
from tqtoy.config import ConfigError
from tqtoy.model import N_FLUID, TQModel

CASES = {
    "one population": {},
    "argon": {"impurities": {"argon": {"injected_atom_density": 5.0e19}}},
    "two elements": {"impurities": {"neon": {"injected_atom_density": 5.0e19},
                                    "argon": {"injected_atom_density": 1.0e19}}},
    "two populations": {"model": {"populations": 2}},
    "multi-species resistivity": {"model": {"populations": 2, "resistivity": "multi_species"}},
    "two populations, multi element": {"model": {"populations": 2},
                                      "impurities": {"neon": {"injected_atom_density": 5.0e19},
                                                     "argon": {"injected_atom_density": 1.0e19}}},
    "ablation source": {"injection": {"source": "ablation"}, "model": {"populations": 2}},
    "no radiation limit": {"model": {"radiation_limit": {"enabled": False}}},
    "no stochastic loss": {"model": {"stochastic": {"enabled": False}}},
    "rate multiplier": {"model": {"atomic_rate_multiplier": 30.0}},
    "linear loss": {"model": {"linear_loss_coefficient": 1.0e3}},
    "current ramp": {"plasma": {"J_final": 2.0e5, "J_ramp_time": 1.0e-3}},
    "blocked exchange": {"model": {"block_electron_ion_exchange": True}},
    "injected charge state": {"injection": {"charge_state": 1}},
}


def _model(overrides, provider):
    data = {"impurities": {"neon": {"injected_atom_density": 5.0e19}}}
    data.update(overrides)
    return TQModel(Config.from_dict(data), provider=provider)


def _central_jacobian(model, t, y, rel=1e-3):
    """Reference Jacobian by central differences, and its round-off floor entry by entry.

    A difference of two nearly equal numbers keeps only the digits the subtraction leaves.
    ``noise[r, c] = eps |f_r| / step_c`` is that floor: an entry of the reference below it
    carries no information, which happens whenever a state barely acts on a large derivative.
    """
    size = len(y)
    out = np.empty((size, size))
    noise = np.empty((size, size))
    eps = np.finfo(float).eps
    for c in range(size):
        step = rel * max(abs(y[c]), 1e-10)
        up, down = y.copy(), y.copy()
        up[c] += step
        down[c] -= step
        f_up = np.asarray(model.rhs(t, up), dtype=float)
        f_down = np.asarray(model.rhs(t, down), dtype=float)
        out[:, c] = (f_up - f_down) / (2 * step)
        noise[:, c] = eps * np.maximum(np.abs(f_up), np.abs(f_down)) / (2 * step)
    return out, noise


def _disagreement(model, t, y, rel=1e-3):
    """Worst disagreement of the analytic charge-state columns, in units of their own row.

    The finite difference is only as good as the agreement of its two steps: the analytic
    value must fall inside that uncertainty. The error is scaled by the largest entry of the
    same row, so a small row is not excused by a large one elsewhere in the matrix.
    """
    coarse, noise = _central_jacobian(model, t, y, rel=rel)
    fine, _ = _central_jacobian(model, t, y, rel=rel / 3)
    analytic = model.jacobian(t, y)
    block = slice(N_FLUID, None)
    uncertainty = 3 * np.abs(coarse - fine) + 20 * noise
    error = np.abs(analytic[:, block] - coarse[:, block]) - uncertainty[:, block]
    rows = np.maximum(np.abs(coarse).max(axis=1, keepdims=True), 1e-300)
    return float(np.max(error / rows))


def _states(model, seed, count=4):
    """Physically meaningful states: a finite difference is meaningless on a vanishing component."""
    rng = np.random.default_rng(seed)
    size = model.layout.size
    for _ in range(count):
        y = np.empty(size)
        y[:N_FLUID] = np.maximum(model.y0[:N_FLUID] * rng.uniform(0.3, 3.0, N_FLUID), 1e-6)
        y[N_FLUID:] = 10.0**rng.uniform(-3, 0, size - N_FLUID)      # 1e16 to 1e19 m^-3
        yield y


@pytest.mark.parametrize("name", list(CASES))
def test_charge_state_columns_are_exact(name, fake_provider):
    """The analytic part must match the finite differences to their own accuracy."""
    model = _model(CASES[name], fake_provider)
    worst = 0.0
    for t in (5e-8, 5e-6, 5e-4):
        for y in _states(model, seed=3):
            worst = max(worst, _disagreement(model, t, y))
    assert worst < 1e-9, f"{name}: {worst:.2e} beyond the accuracy of the finite differences"


@pytest.mark.parametrize("name", ["one population", "two populations", "ablation source"])
def test_the_whole_jacobian_is_consistent(name, fake_provider):
    """The fluid columns are forward differences: first-order accuracy is expected."""
    model = _model(CASES[name], fake_provider)
    for t in (5e-8, 5e-6):
        for y in _states(model, seed=5, count=3):
            reference, noise = _central_jacobian(model, t, y, rel=1e-4)
            analytic = model.jacobian(t, y)
            scale = max(float(np.abs(reference).max()), 1e-300)
            error = np.abs(analytic - reference) - 20 * noise
            assert float(np.max(error)) / scale < 1e-3


def test_inactive_rows_and_columns_are_zero(fake_provider):
    """With one population the hot slots are dummies: they cannot act on anything."""
    model = _model({}, fake_provider)
    jac = model.jacobian(5e-6, next(_states(model, seed=1)))
    for k in (0, 2, 4, 6):                       # Te_hot, ne_hot, Ti_hot, ni_hot
        assert np.all(jac[:, k] == 0.0)
        assert np.all(jac[k, :] == 0.0)


def test_the_charge_state_block_is_tridiagonal(fake_provider):
    """At fixed electron density and temperature the balance couples neighbouring states only."""
    model = _model({}, fake_provider)
    jac = model.jacobian(5e-6, next(_states(model, seed=2)))
    off, Z = model.layout.offsets[0], model.Z[0]
    block = jac[off:off + Z + 1, off:off + Z + 1]
    far = ~np.eye(Z + 1, dtype=bool) & ~np.eye(Z + 1, k=1, dtype=bool) & ~np.eye(Z + 1, k=-1, dtype=bool)
    assert np.all(block[far] == 0.0)


@pytest.mark.parametrize("rtol", [1e-2, 1e-4])
@pytest.mark.parametrize("overrides", [{}, {"model": {"populations": 2}},
                                       {"injection": {"source": "ablation"}}])
def test_a_run_gives_the_same_trajectory(overrides, rtol, fake_provider):
    """The Jacobian drives the Newton iteration, not the solution of the collocation equations.

    It changes the step sequence, so the two settings differ by the local error the tolerance
    allows. The test asserts that scaling rather than a fixed tolerance: an error in the
    Jacobian itself would not shrink with rtol.
    """
    data = {"impurities": {"neon": {"injected_atom_density": 5.0e19}}, "time": {"t_end": 1e-3, "n_output": 60}}
    data.update(overrides)
    base = Config.from_dict(data).with_overrides({"solver.rtol": rtol})
    numeric = simulate(base, provider=fake_provider)
    analytic = simulate(base.with_overrides({"solver.jacobian": "analytic"}), provider=fake_provider)
    assert analytic.ok and numeric.ok
    np.testing.assert_allclose(analytic.t, numeric.t)
    for name, reference in numeric.fluid.items():
        peak = max(float(np.max(np.abs(reference))), 1e-300)
        difference = float(np.max(np.abs(analytic.fluid[name] - reference))) / peak
        assert difference < 10 * rtol, f"{name}: {difference:.2e} against rtol = {rtol:.0e}"
    peak = float(np.max(np.abs(numeric.nij)))
    assert float(np.max(np.abs(analytic.nij - numeric.nij))) / peak < 10 * rtol


def test_analytic_needs_an_implicit_method():
    Config.from_dict({"solver": {"jacobian": "analytic", "method": "BDF"}}).validate()
    with pytest.raises(ConfigError, match="implicit"):
        Config.from_dict({"solver": {"jacobian": "analytic", "method": "RK45"}})
    with pytest.raises(ConfigError):
        Config.from_dict({"solver": {"jacobian": "nope"}})


def test_the_right_hand_side_is_reused(fake_provider):
    """The solver asks for the Jacobian at the point it has just evaluated."""
    model = _model({}, fake_provider)
    y = next(_states(model, seed=4))
    calls = {"n": 0}
    original = model.rhs

    def counting(t, state):
        calls["n"] += 1
        return original(t, state)

    model.rhs = counting
    model.jacobian(5e-6, y)                      # cold: one call for the base point
    first = calls["n"]
    original(5e-6, y)                            # as the solver does before asking for the Jacobian
    calls["n"] = 0
    model.jacobian(5e-6, y)
    assert calls["n"] == first - 1


def _limited_state(model):
    """A cold, impurity-rich state where the radiated power hits the limiter of the model."""
    y = np.empty(model.layout.size)
    y[:N_FLUID] = [0.02, 0.02, 1.0, 1.0, 0.02, 0.02, 1.0, 1.0]     # Te = 20 eV, ne = 1e20 m^-3
    y[N_FLUID:] = 10.0                                             # 1e20 m^-3 in every charge state
    return y


@pytest.mark.parametrize("populations", [1, 2])
def test_the_radiation_limiter_is_taken_into_account(populations, fake_provider):
    """Below model.radiation_limit.Te_below the radiated power is constant: its gradient is zero."""
    model = _model({"model": {"populations": populations}}, fake_provider)
    y = _limited_state(model)
    vals, _, nij = model._unpack(y)
    assert model._radiation_gradient(vals[3], vals[1], nij)[1], "the cold population must be limited"
    if populations == 2:
        assert model._radiation_gradient(vals[2], vals[0], nij)[1], "the hot population must be limited"
    assert _disagreement(model, 5e-6, y) < 1e-9

    # without the limiter the same state gives a different temperature row
    off, Z = model.layout.offsets[0], model.Z[0]
    free = _model({"model": {"populations": populations, "radiation_limit": {"enabled": False}}},
                  fake_provider)
    rows = (1,) if populations == 1 else (0, 1)
    for row in rows:
        assert not np.allclose(model.jacobian(5e-6, y)[row, off:off + Z + 1],
                               free.jacobian(5e-6, y)[row, off:off + Z + 1])


FLOORED = {"Te_cold": 1, "ne_cold": 3, "Ti_cold": 5, "ni_cold": 7, "Te_hot": 0, "ne_hot": 2}


def _below_the_floor(model, y, index):
    """The thermal and density slots of ``y`` with quantity ``index`` put below solver.floor.

    The state vector is built through the model, because with solver.state_variables = energy
    the thermal slots hold n T and not T.
    """
    fluid = model.fluid_from_state(y)
    fluid[index] = 0.5 * model.floor
    return model.state_from_fluid(fluid)


@pytest.mark.parametrize("state", ["energy", "temperature"])
@pytest.mark.parametrize("populations", [1, 2])
@pytest.mark.parametrize("name", list(FLOORED))
def test_a_quantity_at_the_floor_keeps_its_row(name, populations, state, fake_provider):
    """Only a row the floor is actually holding is zeroed.

    The analytic rows must follow the same rule as the right-hand side, otherwise the Jacobian
    announces a frozen equation where the solution still moves. The limited derivative reports
    an active clamp as an exact zero, which is what the Jacobian reads.
    """
    model = _model({"model": {"populations": populations},
                    "solver": {"state_variables": state}}, fake_provider)
    y = next(_states(model, seed=6))
    y[:N_FLUID] = _below_the_floor(model, y, FLOORED[name])
    index = FLOORED[name]
    _, flags, _ = model._unpack(y)
    assert flags[index], "the test state must reach the floor"
    f0 = np.asarray(model.rhs(5e-6, y))
    jac = model.jacobian(5e-6, y)
    for row in (0, 1, 3):                       # the rows carrying analytic charge-state entries
        if populations == 1 and row == 0:
            continue
        alive = np.any(jac[row, N_FLUID:] != 0.0)
        assert alive == (f0[row] != 0.0), \
            f"{name}: row {row} is {'alive' if alive else 'zero'} while d/dt = {f0[row]:.3e}"
    assert _disagreement(model, 5e-6, y) < 1e-9


@pytest.mark.parametrize("t", [5e-8, 1e-7, 5e-6, 5e-4])
@pytest.mark.parametrize("populations", [1, 2])
def test_every_injection_window(t, populations, fake_provider):
    """Before, at, during and after the injection the free-electron source changes branch."""
    model = _model({"model": {"populations": populations}}, fake_provider)
    for y in _states(model, seed=7, count=2):
        assert _disagreement(model, t, y) < 1e-9


@pytest.mark.parametrize("populations", [1, 2])
def test_the_ohmic_chain_is_exercised(populations, fake_provider):
    """A cold, weakly radiating, current-carrying plasma: the charge states act through Z_eff.

    In the usual states the radiated power dwarfs the ohmic power, so an error in the chain
    Z_eff -> resistivity -> current sharing -> ohmic power would hide behind it.
    """
    model = _model({"model": {"populations": populations}}, fake_provider)
    y = np.empty(model.layout.size)
    y[:N_FLUID] = model.state_from_fluid([2.0, 1.0, 1e19, 1e19, 2.0, 1.0, 1e19, 1e19])
    y[N_FLUID:] = 0.01                                             # 1e17 m^-3 per charge state
    vals, flags, nij = model._unpack(y)
    _, _, terms = model.evaluate(5e-6, vals, flags, nij, full=True)
    assert terms["P_ohm_cold"] > 1e3 * terms["P_rad_cold"], "the ohmic power must dominate"
    assert _disagreement(model, 5e-6, y) < 1e-9
