"""
Constitutive models - linear / bilinear / trilinear / non-linear
"""

from numba import njit


@njit
def linear(stretch, sc, d):
    """
    Linear constitutive model
    """

    d_temp = 0.0

    if stretch < sc:
        d_temp = 0.0

    elif stretch >= sc:
        d_temp = 1.0

    # Bond softening factor can only increase (damage is irreversible)
    if d_temp > d:
        d = d_temp

    return d


@njit
def trilinear(stretch, s0, s1, sc, d, beta):
    """
    Trilinear constitutive model
    """

    eta = s1 / s0
    d_temp = 0.0

    if (stretch > s0) and (stretch <= s1):
        d_temp = 1 - ((eta - beta) / (eta - 1) * (s0 / stretch)
                      ) + ((1 - beta) / (eta - 1))

    elif (stretch > s1) and (stretch <= sc):
        d_temp = 1 - (s0 * beta / stretch) * ((sc - stretch) / (sc - s1))

    elif stretch > sc:
        d_temp = 1.0

    # Bond softening factor can only increase (damage is irreversible)
    if d_temp > d:
        d = d_temp

    return d
