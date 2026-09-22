"""Shared fixtures. ``FakeProvider`` gives smooth analytic atomic rates so that most
tests run without the OpenADAS data."""

import numpy as np
import pytest


class _Rate:
    def __init__(self, f):
        self.f = f

    def __call__(self, ne, Te):
        return float(self.f(ne, max(Te, 1e-3)))


class FakeProvider:
    """Analytic rates with the OpenADAS interface (orders of magnitude only)."""

    @staticmethod
    def _E(i):
        return 13.6 * (i + 1) ** 2   # ionisation energy of charge state i [eV]

    def ionisation_rate(self, element, i):
        E = self._E(i)
        return _Rate(lambda ne, T: 1e-13 * np.sqrt(T / E) * np.exp(-E / T) / (1 + T / E))

    def recombination_rate(self, element, i):
        return _Rate(lambda ne, T: 1e-19 * i**2 * (T / 10.0) ** -0.7)

    def line_radiated_power_rate(self, element, i):
        E = self._E(i)
        return _Rate(lambda ne, T: 1e-31 * np.exp(-E / (3 * T)) / np.sqrt(1 + T / E))

    def continuum_radiated_power_rate(self, element, i):
        return _Rate(lambda ne, T: 1e-35 * i**2 * np.sqrt(T))


@pytest.fixture
def fake_provider():
    return FakeProvider()


def adas_available():
    try:
        from cherab.core.atomic import neon
        from cherab.openadas import OpenADAS
        OpenADAS(permit_extrapolation=True).ionisation_rate(neon, 0)
        return True
    except Exception:
        return False


requires_adas = pytest.mark.skipif(not adas_available(), reason="OpenADAS data not installed (tqtoy install-adas)")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import sys
    if "matplotlib.pyplot" in sys.modules:
        sys.modules["matplotlib.pyplot"].close("all")
