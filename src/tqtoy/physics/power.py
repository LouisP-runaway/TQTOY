"""Ohmic heating and stochastic transport losses."""

import numpy as np

from ..constants import E_CHARGE, EPS_0, M_E


def ohmic_power(J, eta):
    """Ohmic power density eta J^2 [W m^-3]."""
    return eta * J**2


def stochastic_power(ne, Te, deltaB_over_B=1e-2, r=0.6, a=2.0, coulomb_log=15.0):
    """Electron heat loss by parallel transport along stochastic field lines [W m^-3].

    Rechester-Rosenbluth type estimate following Ward & Wesson (Nucl. Fusion 1992),
    as implemented here. It assumes Te(r) = Te0 (1 - r^2/a^2) and a
    flat density profile, and evaluates the local loss at minor radius r.

    Parameters
    ----------
    ne [m^-3], Te [eV] : electron density and temperature
    deltaB_over_B : relative magnetic perturbation
    r, a : evaluation radius and minor radius [m]
    coulomb_log : Coulomb logarithm (constant)
    """
    A = deltaB_over_B**2 * 6/M_E * 3 * (2*np.pi)**(3/2) * (EPS_0**2 * M_E**(1/2)) / (ne * E_CHARGE**4 * coulomb_log)
    return 2 * A * ne / (r * a**2) * (1 - 6*(r/a)**2) / (1 - (r/a)**2)**2 * (Te*E_CHARGE)**(7/2)
