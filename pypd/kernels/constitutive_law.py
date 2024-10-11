"""
Constitutive models - linear / bilinear / trilinear / non-linear

Small, highly optimised computational units written using Numba

"""

from numba import njit
import numpy as np


@njit
def linear(i, s, d, sc):
    """
    Linear constitutive model
    """

    d_temp = 0.0

    if s < sc[i]:
        d_temp = 0.0

    elif s >= sc[i]:
        d_temp = 1.0

    # Bond softening factor can only increase (damage is irreversible)
    if d_temp > d:
        d = d_temp

    return d


@njit
def trilinear(i, s, d, s0, s1, sc, beta):
    """
    Trilinear constitutive model
    """

    eta = s1[i] / s0
    d_temp = 0.0

    if (s > s0) and (s <= s1[i]):
        d_temp = 1 - ((eta - beta) / (eta - 1) * (s0 / s)) + ((1 - beta) / (eta - 1))

    elif (s > s1[i]) and (s <= sc[i]):
        d_temp = 1 - (s0 * beta / s) * ((sc[i] - s) / (sc[i] - s1[i]))

    elif s > sc[i]:
        d_temp = 1.0

    # Bond softening factor can only increase (damage is irreversible)
    if d_temp > d:
        d = d_temp

    return d


@njit
def nonlinear(i, s, d, s0, sc, alpha, k):
    """
    Non-linear constitutive model
    """

    d_temp = 0.0

    if (s > s0) and (s <= sc[i]):
        numerator = 1 - np.exp(-k * (s - s0) / (sc[i] - s0))
        denominator = 1 - np.exp(-k)
        residual = alpha * (1 - (s - s0) / (sc[i] - s0))
        d_temp = 1 - (s0 / s) * ((1 - (numerator / denominator)) + residual) / (
            1 + alpha
        )
    elif s > sc[i]:
        d_temp = 1

    if d_temp > d:
        d = d_temp

    return d
