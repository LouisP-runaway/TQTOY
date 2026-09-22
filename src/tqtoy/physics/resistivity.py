"""Effective charge, resistivity, current sharing and parallel electric fields."""

import numpy as np

from ..constants import C_LIGHT, E_CHARGE, EPS_0, M_E, M_ION
from .collisions import collision_time, log_c_ee


def zeff(ni_main, nij, atomic_numbers):
    """Effective charge Z_eff = (n_i + sum Z^2 n_Z) / (n_i + sum Z n_Z).

    ni_main : density of the singly charged main ions [m^-3]
    nij : array (Z_max+1, n_elements) of impurity charge-state densities [m^-3]
    atomic_numbers : atomic number of each impurity element
    """
    numerator = ni_main
    denominator = ni_main
    for j, Z in enumerate(atomic_numbers):
        column = nij[:Z + 1, j].tolist()             # scalar arithmetic: no per-element array indexing
        for i in range(0, Z + 1):
            numerator = numerator + float(i)**2 * column[i]
            denominator = denominator + float(i) * column[i]
    return numerator / denominator


def eta_spitzer(Te, Z_eff, ne):
    """Spitzer resistivity [Ohm m] with a Z_eff correction.

    eta = 1.65e-9 lnL Te[keV]^-1.5 f(Z_eff), with f(1) = 1. Te is floored at 0.01 eV.
    """
    if Te < 0.01:
        Te = 0.01
    coulomb_log = log_c_ee(ne, Te)
    coef_zeff = (Z_eff * (1. + 1.198 * Z_eff + 0.222 * Z_eff**2)) / ((1. + 2.966 * Z_eff + 0.753 * Z_eff**2) * ((1. + 1.198 + 0.222) / (1. + 2.966 + 0.753)))
    return 1.65e-9 * coulomb_log * (Te * 0.001)**(-1.5) * coef_zeff


def eta_multi_species(Te_hot, Te_cold, Ti_hot, Ti_cold, ne_hot, ne_cold, ni_hot, ni_cold, Z_eff, m_ion=M_ION):
    """Resistivities (eta_hot, eta_cold) with the hot-electron conductivity correction.

    The cold population keeps the Spitzer value. The hot conductivity accounts for
    collisions of hot electrons on cold electrons and on both ion populations.
    This expression is not validated, hence the warning on model.resistivity.
    """
    nu_ehot_ecold = 1.0 / collision_time(ne_cold, Te_hot, Te_cold, "ee", Te_hot, ne_hot, m_ion)
    nu_ehot_ihot = 1.0 / collision_time(ni_hot, Te_hot, Ti_hot, "ei", Te_hot, ne_hot, m_ion)
    nu_ehot_icold = 1.0 / collision_time(ni_cold, Te_hot, Ti_cold, "ei", Te_hot, ne_hot, m_ion)
    sum_nu_ehot = nu_ehot_ecold + nu_ehot_ihot + nu_ehot_icold
    sigma_c = 1 / eta_spitzer(Te_cold, Z_eff, ne_cold)
    sigma_h = ne_hot/ne_cold * nu_ehot_ecold/sum_nu_ehot*sigma_c + ne_hot * E_CHARGE**2 / M_E / sum_nu_ehot
    if sigma_h == 0 or sigma_c == 0:
        raise ValueError("Zero conductivity in the multi-species resistivity model.")
    return 1 / sigma_h, 1 / sigma_c


def share_current(J, eta_hot, eta_cold):
    """Split the current density between two parallel resistive channels (same E field).

    Returns (J_hot, J_cold) with J_hot = J eta_cold / (eta_cold + eta_hot).
    """
    alpha = eta_cold / (eta_cold + eta_hot)
    J_hot = J * alpha
    return J_hot, J - J_hot


def linear_current_ramp(J0, t, Jf, t_ramp):
    """Current density decreasing linearly from J0 (t = 0) to Jf (t >= t_ramp)."""
    if t >= t_ramp:
        return Jf
    return J0 + (Jf - J0) * t / t_ramp


def critical_field(ne, Te):
    """Connor-Hastie critical electric field [V m^-1].

    E_c = ne e^3 lnL_c / (4 pi eps0^2 me c^2), with the relativistic Coulomb
    logarithm lnL_c = 14.6 + 0.5 ln(Te[eV] / ne[1e20 m^-3]).
    Vectorised over ne and Te.
    """
    ne = np.asarray(ne, dtype=float)
    Te = np.maximum(np.asarray(Te, dtype=float), 1e-2)
    ln_lambda_c = 14.6 + 0.5 * np.log(Te / (ne * 1e-20))
    return ne * E_CHARGE**3 * ln_lambda_c / (4 * np.pi * EPS_0**2 * M_E * C_LIGHT**2)
