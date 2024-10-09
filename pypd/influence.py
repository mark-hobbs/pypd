"""
Influence function base class
-----------------------------

Notes
-----
* see pysph / solver / solver.py
    - kernel : pysph.base.kernels.Kernel
"""

import numpy as np


class InfluenceFunction:
    """
    Subclass this to define a new  influence function

    Attributes
    ----------
    c : float
        Constant

    omega :
        Influence function, also referred to as the weight function or kernel
        function, which describes how the interaction between particles
        diminishes with increasing distance.

    Methods
    -------
    """

    def __init__(self) -> None:
        self.c = c
        self.omega = omega

    def __call__(self, *args: np.Any, **kwds: np.Any) -> np.Any:
        pass

    def _return_c():
        pass

    def _return_omega():
        pass


class Constant(InfluenceFunction):

    def _return_c():
        return (9 * material.E) / (np.pi * t * particles.horizon**3)

    def _return_omega():
        return 1


class Triangular(InfluenceFunction):
    """
    Conical
    """

    def _return_omega(xi):
        """
        xi : ndarray (float)
            Reference bond length
        """
        return 1 - xi / horizon


class Quartic(InfluenceFunction):

    def _return_c():
        return (315 * E) / (8 * np.pi * horizon**3)

    def _return_omega(xi):
        """
        xi : ndarray (float)
            Reference bond length
        """
        return (xi / horizon) ** 4 - 2 * (xi / horizon) ** 2 + 1


class Normal(InfluenceFunction):
    pass
