"""Impurity charge-state balance and radiated power."""

import numpy as np

from ..constants import E_CHARGE

# Low-temperature radiation limiter: below RAD_CAP_TE, the radiated power
# cannot exceed the electron thermal energy divided by RAD_CAP_TIME.
RAD_CAP_TE = 100.0      # [eV]
RAD_CAP_TIME = 1e-6     # [s]


def charge_state_derivative(rates, nij, ne, Te, source=None, source_state=0, rate_multiplier=1.0):
    """Time derivative of the impurity charge-state densities [m^-3 s^-1].

    Non-coronal balance between ionisation (ADAS SCD) and recombination (ADAS ACD):
        dn_i/dt = (S_{i-1} n_{i-1} ne - S_i n_i ne + a_{i+1} n_{i+1} ne - a_i n_i ne) * rate_multiplier

    Each term carries its own factor ne. Factoring it out of the
    bracket changes the last bits of the result.

    Parameters
    ----------
    rates : list of ElementRates, one per element
    nij : array (Z_max+1, n_elements) [m^-3]
    ne, Te : electron density [m^-3] and temperature [eV] seen by the impurities
    source : optional sequence (n_elements,) of particle sources [m^-3 s^-1]
        deposited in charge state ``source_state``
    rate_multiplier : factor applied to ionisation and recombination rates.
        Large values drive the distribution towards coronal equilibrium.
    """
    dnij = np.zeros_like(nij)
    for j, el in enumerate(rates):
        Z = el.Z
        S, a, _, _ = el.coefficients(ne, Te)        # S: states 0..Z-1, a: states 1..Z
        n = nij[:, j]
        # Sn[i] = S_i n_i ne (ionisation of state i), an[i] = a_{i+1} n_{i+1} ne (recombination
        # of state i+1). The expressions below are those of the loop over the states, in the
        # same order: dn_0 = -Sn_0 + an_0, dn_Z = Sn_{Z-1} - an_{Z-1} and, in between,
        # dn_i = Sn_{i-1} - Sn_i + an_i - an_{i-1}.
        Sn = S * n[:Z] * ne
        an = a * n[1:Z + 1] * ne
        column = dnij[:Z + 1, j]
        column[0] = -Sn[0] + an[0]
        column[Z] = Sn[Z - 1] - an[Z - 1]
        if Z > 1:
            column[1:Z] = Sn[:-1] - Sn[1:] + an[1:] - an[:-1]
        column *= rate_multiplier
        if source is not None:
            dnij[source_state, j] += source[j]
    return dnij


def radiated_power(rates, nij, ne, Te, cap_Te=RAD_CAP_TE, cap_time=RAD_CAP_TIME):
    """Impurity radiated power density [W m^-3].

    Line radiation (ADAS PLT) of states 0..Z-1 plus continuum and recombination
    radiation (ADAS PRB) of states 1..Z. Below ``cap_Te``, the power is limited to
    1.5 ne Te e / ``cap_time`` (low-temperature limiter). ``cap_Te = None``
    disables the limiter.
    """
    power = 0
    for j, el in enumerate(rates):
        Z = el.Z
        _, _, line, continuum = el.coefficients(ne, Te)   # line: 0..Z-1, continuum: 1..Z
        n = nij[:, j]
        for i in range(0, Z + 1):
            if i == 0:
                power += ne * n[i] * line[i]
            elif i == Z:
                power += ne * n[i] * continuum[i - 1]
            else:
                power += ne * n[i] * (line[i] + continuum[i - 1])
    if cap_Te is not None and Te < cap_Te:
        power = min(power, 3/2*ne*Te*E_CHARGE/cap_time)
    return power



def coronal_fractions(el, ne, Te):
    """Coronal charge-state fractions of one element at (ne, Te), summing to one.

    In coronal equilibrium every ionisation is balanced by the recombination of the state
    above it, so f_{k+1} / f_k = S_k / a_{k+1}. The ratios are accumulated in logarithms,
    because their product spans hundreds of decades at the ends of the temperature range,
    and the largest term is factored out before the exponential.

    Parameters
    ----------
    el : ElementRates of one element
    ne, Te : electron density [m^-3] and temperature [eV]
    """
    S, a, _, _ = el.coefficients(ne, Te)        # S: states 0..Z-1, a: states 1..Z
    tiny = 1e-300
    log_ratio = np.log(np.maximum(S, tiny)) - np.log(np.maximum(a, tiny))
    log_f = np.concatenate(([0.0], np.cumsum(log_ratio)))
    log_f = log_f - log_f.max()
    f = np.exp(log_f)
    return f / f.sum()


def coronal_cooling_rate(el, ne, Te):
    """Radiated power per electron and per impurity ion in coronal equilibrium [W m^3].

    Same radiation terms as :func:`radiated_power`, weighted by the coronal fractions
    instead of the charge-state densities of the run, and without the low-temperature
    limiter. The radiated power of an element in coronal equilibrium is ne n_imp L_Z.
    """
    Z = el.Z
    f = coronal_fractions(el, ne, Te)
    _, _, line, continuum = el.coefficients(ne, Te)     # line: 0..Z-1, continuum: 1..Z
    return float(np.dot(f[:Z], line) + np.dot(f[1:Z + 1], continuum))
