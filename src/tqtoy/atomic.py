"""Atomic data (ionisation, recombination and radiation rates).

The default provider is OpenADAS through the CHERAB package. Any object
exposing the four methods below can be used instead (for tests or for another
database):

    ionisation_rate(element, charge)               -> callable(ne, Te) [m^3 s^-1]
    recombination_rate(element, charge)            -> callable(ne, Te) [m^3 s^-1]
    line_radiated_power_rate(element, charge)      -> callable(ne, Te) [W m^3]
    continuum_radiated_power_rate(element, charge) -> callable(ne, Te) [W m^3]

Densities are in m^-3 and temperatures in eV.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, List, Optional, Sequence

import numpy as np

ADAS_INSTALL_HINT = (
    "OpenADAS data are missing from the local CHERAB repository. Install them once with\n"
    "    tqtoy install-adas\n"
    "or, equivalently,\n"
    "    python -c \"from cherab.openadas.repository import populate; populate()\""
)


def resolve_element(element: Any):
    """Return a CHERAB-like element from a name ('neon'), a symbol ('Ne') or an element object.

    Objects that already carry an ``atomic_number`` attribute are returned unchanged.
    """
    if hasattr(element, "atomic_number"):
        return element
    try:
        from cherab.core.atomic import lookup_element
    except ImportError as exc:  # pragma: no cover
        raise ImportError("CHERAB is required to resolve element names (pip install cherab).") from exc
    try:
        return lookup_element(str(element))
    except ValueError as exc:
        raise ValueError(f"Unknown element '{element}'.") from exc


def default_provider(permit_extrapolation: bool = True):
    """OpenADAS provider with extrapolation allowed outside the tabulated range."""
    try:
        from cherab.openadas import OpenADAS
    except ImportError as exc:  # pragma: no cover
        raise ImportError("CHERAB is required for the default atomic data (pip install cherab).") from exc
    return OpenADAS(permit_extrapolation=permit_extrapolation)


@dataclass
class ElementRates:
    """Rate functions for all charge states of one element.

    Index ``i`` refers to the charge state i (0 = neutral, Z = fully stripped).
    Entries that do not exist physically are ``None``: ``ionisation[Z]``,
    ``recombination[0]``, ``line[Z]`` and ``continuum[0]``.
    """

    element: Any
    ionisation: List[Optional[Callable]]
    recombination: List[Optional[Callable]]
    line: List[Optional[Callable]]
    continuum: List[Optional[Callable]]
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def Z(self) -> int:
        return self.element.atomic_number

    def coefficients(self, ne: float, Te: float):
        """Rate vectors at (ne, Te): (ionisation 0..Z-1, recombination 1..Z, line 0..Z-1, continuum 1..Z).

        The values are cached on (ne, Te). The right-hand side asks for the charge-state
        balance and for the radiated power at the same point, and the solver perturbs the
        state variables one at a time when it estimates the Jacobian, so most calls repeat
        a point that was just evaluated. The rate functions themselves are unchanged.
        """
        key = (ne, Te)
        values = self._cache.get(key)
        if values is None:
            Z = self.Z
            values = (np.array([self.ionisation[i](ne, Te) for i in range(Z)]),
                      np.array([self.recombination[i](ne, Te) for i in range(1, Z + 1)]),
                      np.array([self.line[i](ne, Te) for i in range(Z)]),
                      np.array([self.continuum[i](ne, Te) for i in range(1, Z + 1)]))
            if len(self._cache) >= 8:
                self._cache.clear()
            self._cache[key] = values
        return values


class AtomicData:
    """Container of the rate functions of every impurity element of a simulation."""

    def __init__(self, elements: Sequence[Any], provider: Any = None):
        self.elements = [resolve_element(e) for e in elements]
        self.provider = provider if provider is not None else default_provider()
        try:
            self.rates = [self._load(el) for el in self.elements]
        except RuntimeError as exc:
            if "not available" in str(exc):
                raise RuntimeError(f"{exc}\n{ADAS_INSTALL_HINT}") from exc
            raise

    def _load(self, element) -> ElementRates:
        Z = element.atomic_number
        p = self.provider
        return ElementRates(
            element=element,
            ionisation=[p.ionisation_rate(element, i) for i in range(Z)] + [None],
            recombination=[None] + [p.recombination_rate(element, i) for i in range(1, Z + 1)],
            line=[p.line_radiated_power_rate(element, i) for i in range(Z)] + [None],
            continuum=[None] + [p.continuum_radiated_power_rate(element, i) for i in range(1, Z + 1)],
        )

    @property
    def atomic_numbers(self) -> List[int]:
        return [el.atomic_number for el in self.elements]

    @property
    def names(self) -> List[str]:
        return [el.name for el in self.elements]


@lru_cache(maxsize=16)
def _cached_openadas(names: tuple) -> AtomicData:
    return AtomicData(list(names))


def get_atomic_data(elements: Sequence[Any], provider: Any = None) -> AtomicData:
    """Return atomic data for ``elements``.

    With the default OpenADAS provider, the data are loaded once per process and
    reused by all subsequent simulations (useful for parameter scans).
    """
    if provider is not None:
        return AtomicData(elements, provider)
    names = tuple(resolve_element(e).name for e in elements)
    return _cached_openadas(names)


def install_openadas() -> None:
    """Download the OpenADAS rate files into the local CHERAB repository."""
    from cherab.openadas.repository import populate

    populate()
