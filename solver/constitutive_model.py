"""
Constitutive models - linear / bilinear / trilinear / non-linear
"""

from numba import njit

@njit
def trilinear_constitutive_model(stretch, s0, s1, sc, bond_damage, beta):
    """
    Trilinear constitutive model
    """

    eta = s1 / s0
    bond_damage_temp = 0.0

    if (stretch > s0) and (stretch <= s1):
        bond_damage_temp = (1 - ((eta - beta) / (eta - 1) * (s0 / stretch))
                            + ((1 - beta) / (eta - 1)))
    elif (stretch > s1) and (stretch <= sc):
        bond_damage_temp = 1 - ((s0 * beta / stretch)
                                * ((sc - stretch) / (sc - s1)))
    elif stretch > sc:
        bond_damage_temp = 1.0

    # Bond softening factor can only increase (damage is irreversible)
    if bond_damage_temp > bond_damage:
        bond_damage = bond_damage_temp

    return bond_damage
