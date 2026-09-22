"""Shattered-pellet shard size distribution and ablation rate."""

import numpy as np
from scipy.special import k0


def D2_fraction(n_D, n_imp):
    """Molecular fraction X of D2 in a mixed D2/impurity pellet.

    X = (n_D/2) / (n_D/2 + n_imp), with n_D the number of D atoms and n_imp the
    number of impurity atoms (any consistent unit: counts or densities).
    """
    return (n_D / 2) / ((n_D / 2) + n_imp)


def parks_kappa(n_shards, n_atoms, solid_density):
    """Inverse characteristic shard size kappa_p of the Parks distribution [m^-1].

    kappa_p = (6 pi^2 N_s n_solid / N_atoms)^(1/3).
    """
    return (n_shards * solid_density * 6 * np.pi**2 / n_atoms)**(1/3)


def parks_pdf(r, kappa_p):
    """Parks shard-radius probability density r kappa_p^2 K0(r kappa_p)."""
    return r * kappa_p ** 2 * k0(r * kappa_p)


def sample_shard_radii(n_shards, n_atoms, solid_density, r_min, r_max, seed=None):
    """Draw ``n_shards`` radii from the Parks distribution by rejection sampling [m].

    The radii are rescaled so that the total shard volume equals the pellet volume
    n_atoms / solid_density.
    """
    rng = np.random.default_rng(seed)
    kappa_p = parks_kappa(n_shards, n_atoms, solid_density)
    pdf_max = np.max(parks_pdf(np.linspace(r_min, r_max, 10000), kappa_p))
    samples = np.empty(0)
    while samples.size < n_shards:
        x = rng.uniform(r_min, r_max, size=4 * n_shards)
        y = rng.uniform(0.0, pdf_max, size=4 * n_shards)
        samples = np.concatenate([samples, x[y < parks_pdf(x, kappa_p)]])
    samples = samples[:n_shards]
    return ((3 * n_atoms) / (4 * np.pi * solid_density))**(1/3) * (samples**3 / np.sum(samples**3))**(1/3)


def ablation_rate(n, T, X, volume, r_p, n_shards, time, r_p_samples=None, B=2.0, deuterium=False,
                  switch_on_time=1e-6):
    """Pellet ablation rate per unit plasma volume [atoms m^-3 s^-1].

    Scaling law for mixed D2/Ne pellets, with a magnetic field correction
    (2/max(B, 2))^0.843 and a tanh(t / switch_on_time) switch-on.

    Parameters
    ----------
    n, T : electron density [m^-3] and temperature [eV] of the ablating plasma
    X : molecular fraction of D2 in the pellet
    volume : plasma volume where the ablated material is deposited [m^3]
    r_p : shard radius used if ``r_p_samples`` is None [m]
    n_shards : number of shards used if ``r_p_samples`` is None
    time : current time [s]
    r_p_samples : optional array of individual shard radii [m]
    B : magnetic field [T]
    deuterium : use the deuterium coefficients instead of the hydrogen ones (default)
    switch_on_time : time constant of the tanh switch-on [s]
    """
    if T < 0:
        T = 1
    A = 27.0837
    B_coef = 1.48709
    if deuterium:
        C = 4.062 * 1e14
        fw = 20.183 / 4.0282
    else:
        C = 8.12 * 1e14
        fw = 20.183 / 2.016
    lamb = A + np.tan(B_coef * X)
    if r_p_samples is not None:
        g = 0
        for r_p_parks in r_p_samples:
            g += (C * lamb) / (fw * (1 - X) + X) * n ** (1 / 3) * T ** (5 / 3) * r_p_parks ** (4 / 3) / volume
    else:
        g = n_shards * (C * lamb) / (fw * (1 - X) + X) * n ** (1 / 3) * T ** (5 / 3) * r_p ** (4 / 3) / volume
    f_B = (2 / max(B, 2))**(0.843)
    smooth_function = np.tanh(time / switch_on_time)
    return g * f_B * smooth_function


def shard_equivalent_radius(pellet_length):
    """Radius of a sphere with the volume of a cylindrical pellet with L = D [m].

    r = (3 L^3 / 16)^(1/3), for a cylinder whose diameter equals its length.
    """
    return (3 * pellet_length**2 * pellet_length / 16)**(1/3)
