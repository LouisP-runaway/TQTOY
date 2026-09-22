import numpy as np
import pytest

from conftest import requires_adas
from tqtoy.model import StateLayout, N_FLUID, NIJ_SCALE
from tqtoy.physics.ablation import D2_fraction, parks_kappa, sample_shard_radii
from tqtoy.physics.collisions import collision_time, exchange_power
from tqtoy.physics.resistivity import critical_field, eta_spitzer, share_current, zeff


def test_zeff_pure_plasma_is_one():
    nij = np.zeros((11, 1))
    assert zeff(1e20, nij, [10]) == 1.0


def test_zeff_fully_stripped_neon():
    nij = np.zeros((11, 1))
    nij[10, 0] = 1e18
    ni = 1e20
    assert zeff(ni, nij, [10]) == pytest.approx((ni + 100e18) / (ni + 10e18))


def test_spitzer_scaling():
    # eta ~ Te^-1.5 at fixed Coulomb logarithm regime (Te >= 10 eV branch)
    e1 = eta_spitzer(1e3, 1.0, 1e20)
    e2 = eta_spitzer(4e3, 1.0, 1e20)
    assert e1 / e2 == pytest.approx(8.0, rel=0.1)
    assert eta_spitzer(1e3, 2.0, 1e20) > e1


def test_share_current_conserves_total():
    J_hot, J_cold = share_current(1e6, 1e-8, 1e-6)
    assert J_hot + J_cold == pytest.approx(1e6)
    assert J_hot > J_cold                     # the less resistive channel carries more current


def test_critical_field_order_of_magnitude():
    # E_c ~ 0.08 V/m at 1e20 m^-3 for lnL_c ~ 15-17
    Ec = critical_field(1e20, 1e3)
    assert 0.06 < Ec < 0.12


def test_collision_time_and_exchange_sign():
    assert collision_time(1e20, 1e3, 1e3, "ee", 1e3, 1e20) > 0
    assert exchange_power(1e20, 2e3, 1e20, 1e3, "ei", 2e3, 1e20) > 0
    assert exchange_power(1e20, 1e3, 1e20, 2e3, "ei", 1e3, 1e20) < 0
    with pytest.raises(ValueError):
        collision_time(1e20, 1e3, 1e3, "xx", 1e3, 1e20)


def test_d2_fraction():
    assert D2_fraction(2.0, 0.0) == 1.0
    assert D2_fraction(2.0, 1.0) == 0.5


def test_parks_sampling_preserves_pellet_volume():
    n_atoms, n_solid = 1.8e24, 4.95e28
    r = sample_shard_radii(300, n_atoms, n_solid, 1e-4, 1e-2, seed=1)
    assert r.size == 300
    assert np.sum(4 / 3 * np.pi * r**3) == pytest.approx(n_atoms / n_solid, rel=1e-10)
    np.testing.assert_array_equal(r, sample_shard_radii(300, n_atoms, n_solid, 1e-4, 1e-2, seed=1))
    assert parks_kappa(300, n_atoms, n_solid) > 0


def test_state_layout_multi_element():
    """Regression test for the charge-state indexing of several elements."""
    lay = StateLayout((10, 18))
    assert lay.offsets == [N_FLUID, N_FLUID + 11]
    assert lay.size == N_FLUID + 11 + 19
    y = np.zeros(lay.size)
    y[N_FLUID + 11 + 18] = 1.0                # fully stripped state of element 1
    nij = lay.nij_from_vector(y)
    assert nij.shape == (19, 2)
    assert nij[18, 1] == NIJ_SCALE and nij[:, 0].sum() == 0


@requires_adas
@pytest.mark.adas
def test_the_coronal_neon_cooling_curve_matches_the_published_one():
    """Coronal equilibrium of neon on the real atomic data (needs the OpenADAS data).

    Three properties that the published neon cooling curve has, and that no analytic
    surrogate reproduces: the mean charge saturates at 10 when the element is fully
    stripped at high temperature, it is below 1 at 2 eV, and the radiated power per
    electron and per ion peaks between 10 and 100 eV, above 1e-32 W m^3.
    """
    from tqtoy.atomic import AtomicData
    from tqtoy.physics.impurities import coronal_cooling_rate, coronal_fractions

    el = AtomicData(["neon"]).rates[0]
    charges = np.arange(el.Z + 1)

    def mean_charge(Te):
        return float(np.dot(charges, coronal_fractions(el, 1e20, Te)))

    assert mean_charge(2.0) < 1.0
    assert 9.9 < mean_charge(1e4) <= 10.0
    assert mean_charge(20.0) == pytest.approx(4.4, abs=0.5)

    Te = np.logspace(0, 4, 200)
    L = np.array([coronal_cooling_rate(el, 1e20, T) for T in Te])
    peak = Te[np.argmax(L)]
    assert 10.0 < peak < 100.0, f"the coronal peak of neon is at {peak:.1f} eV"
    assert L.max() > 1e-32
