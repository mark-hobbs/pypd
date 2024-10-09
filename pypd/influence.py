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

    def __call__(self):
        pass

    def _return_c():
        pass

    def _return_omega():
        pass


class Constant:

    def __init__(self, particles, xi):
        self.particles = particles
        self.xi = xi

    def __call__(self):
        return self._c() * self._omega()

    def _c(self):
        return (9 * self.particles.material.E) / (np.pi * self.particles.dx * self.particles.horizon**3)

    def _omega(self):
        return np.ones(len(self.xi))


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


class Quartic:

    def __init__(self, particles, xi):
        self.particles = particles
        self.xi = xi

    def __call__(self):
        return self._c() * self._omega()

    def _c(self):
        return (315 * self.particles.material.E) / (8 * np.pi * self.particles.dx * self.particles.horizon**3)

    def _omega(self):
        return (self.xi / self.particles.horizon) ** 4 - 2 * (self.xi / self.particles.horizon) ** 2 + 1


class Normal(InfluenceFunction):
    pass
