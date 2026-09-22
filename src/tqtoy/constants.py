"""Physical constants (SI units).

They are written here once so that every module shares the same values. Do not
reformat the expressions below: a literal and the product that produces it can
differ in the last bit, which the round-off tests of the suite detect.
"""

E_CHARGE = 1.602e-19        # Elementary charge [C]
EPS_0 = 8.85418782e-12      # Vacuum permittivity [F m^-1]
M_E = 9.109 * 1e-31         # Electron mass [kg]
M_P = 1.6726e-27            # Proton mass [kg]
C_LIGHT = 2.99792458e8      # Speed of light [m s^-1] (used by the critical field only)

# Main-ion mass used in the collision times. The proton mass is used for the
# main ions, including for a deuterium plasma. model.main_ion_mass scales it.
M_ION = 1.0 * M_P
