"""Coulomb logarithms, collision times and inter-species thermal exchange."""

import numpy as np

from ..constants import E_CHARGE, EPS_0, M_E, M_ION



def _masses(kind, m_ion):
    return {"ee": (M_E, M_E), "ei": (M_E, m_ion), "ie": (m_ion, M_E), "ii": (m_ion, m_ion)}[kind]


def log_c_ee(ne, Te):
    """Electron-electron Coulomb logarithm (NRL formulary).

    ne [m^-3], Te [eV]. Floors: Te -> 1 eV if negative, ne -> 1e15 m^-3, result >= 1.
    """
    if Te < 0:
        Te = 1
    if ne < 1e15:
        ne = 1e15
    if Te < 10:
        return max(23.0000 - np.log((ne * 1e-6)**0.5 * Te**(-1.5)), 1)
    return max(24.1513 - np.log((ne * 1e-6)**0.5 * Te**(-1.0)), 1)


def log_c_ei(ne, Te):
    """Electron-ion Coulomb logarithm (Wesson, Tokamaks). Floors: Te >= 1 eV, ne >= 1e15 m^-3."""
    if Te < 1:
        Te = 1
    if ne < 1e15:
        ne = 1e15
    return max(15.2 - 0.5 * np.log(ne * 1e-20) + np.log(Te * 1e-3), 1)


def log_c_ii(ni, Ti):
    """Ion-ion Coulomb logarithm (Wesson, Tokamaks). Floors: Ti -> 1 eV if negative, ni >= 1e15 m^-3."""
    if Ti < 0:
        Ti = 1
    if ni < 1e15:
        ni = 1e15
    return max(17.3 - 0.5 * np.log(ni * 1e-20) + 1.5 * np.log(Ti * 1e-3), 1)


def collision_time(n_beta, T_alpha, T_beta, kind, T_fast, n_fast, m_ion=M_ION):
    """Energy-exchange time between species alpha and beta [s] (Matsuyama 2017).

    Parameters
    ----------
    n_beta : density of the target species beta [m^-3]
    T_alpha, T_beta : temperatures of alpha and beta [eV]
    kind : 'ee', 'ei', 'ie' or 'ii' (first letter = alpha)
    T_fast, n_fast : temperature and density used in the Coulomb logarithm.
        The model evaluates the logarithm with the hottest population only.
    m_ion : main-ion mass [kg] (default: proton mass)
    """
    if kind == "ee":
        coulomb_log = log_c_ee(n_fast, T_fast)
    elif kind in ("ei", "ie"):
        coulomb_log = log_c_ei(n_fast, T_fast)
    elif kind == "ii":
        coulomb_log = log_c_ii(n_fast, T_fast)
    else:
        raise ValueError(f"Unknown interaction type '{kind}' (expected 'ee', 'ei', 'ie' or 'ii').")
    m_alpha, m_beta = _masses(kind, m_ion)
    pre_factor = (3 * np.sqrt(2) * np.pi**(3/2) * EPS_0 ** 2 * m_alpha * m_beta) / (n_beta * E_CHARGE ** 4 * coulomb_log)
    temp_factor = (T_alpha * E_CHARGE / m_alpha + T_beta * E_CHARGE / m_beta)**(3/2)
    return pre_factor * temp_factor


def exchange_power(n_a, T_a, n_b, T_b, kind, T_fast, n_fast, m_ion=M_ION):
    """Thermal power transferred from species a to species b [W m^-3].

    P = 3 n_a e (T_a - T_b) / (2 tau_ab), positive when a is hotter than b.
    ``kind`` and (T_fast, n_fast) are passed to :func:`collision_time`.
    """
    tau = collision_time(n_b, T_a, T_b, kind, T_fast, n_fast, m_ion)
    return 3 * n_a * E_CHARGE * (T_a - T_b) / (2 * tau)
