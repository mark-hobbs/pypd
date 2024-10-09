import numpy as np
from abc import ABC, abstractmethod


class InfluenceFunction(ABC):
    """
    Abstract base class for influence functions

    Attributes
    ----------
    c : float
        Stiffness constant, dependent on material and horizon properties.

    omega : ndarray (float)
        Influence function, describing how particle interactions diminish
        with increasing distance.

    Methods
    -------
    __call__():
        Compute the product of c and omega.

    _c():
        Compute the stiffness constant (to be implemented by subclasses).

    _omega():
        Compute the influence function (to be implemented by subclasses).
    """

    def __init__(self, particles, xi) -> None:
        self.particles = particles
        self.xi = xi

    def __call__(self):
        """
        Returns the product of the stiffness constant and the influence function.
        """
        return self._c() * self._omega()

    @abstractmethod
    def _c(self) -> float:
        """
        Compute the stiffness constant. This method must be implemented
        by the subclass.
        """
        pass

    @abstractmethod
    def _omega(self) -> np.ndarray:
        """
        Compute the influence function. This method must be implemented
        by the subclass.
        """
        pass


class Constant(InfluenceFunction):
    """
    Constant influence function where omega is 1 for all bonds.
    """

    def _c(self) -> float:
        """
        Compute the stiffness constant for the constant influence function.
        """
        return (9 * self.particles.material.E) / (
            np.pi * self.particles.dx * self.particles.horizon**3
        )

    def _omega(self) -> np.ndarray:
        """
        Constant influence function (omega), equal to 1 for all bonds.
        """
        return np.ones(len(self.xi))


class Quartic(InfluenceFunction):
    """
    Quartic influence function, which is a polynomial function of the bond
    length xi relative to the horizon.
    """

    def _c(self) -> float:
        """
        Compute the stiffness constant for the quartic influence function.
        """
        return (315 * self.particles.material.E) / (
            8 * np.pi * self.particles.dx * self.particles.horizon**3
        )

    def _omega(self) -> np.ndarray:
        """
        Quartic influence function (omega) as a function of bond length xi.
        """
        return (
            (self.xi / self.particles.horizon) ** 4
            - 2 * (self.xi / self.particles.horizon) ** 2
            + 1
        )


class Triangular(InfluenceFunction):
    """
    Conical
    """

    def _omega(self) -> np.ndarray:
        """
        xi : ndarray (float)
            Reference bond length
        """
        return 1 - xi / horizon


class Normal(InfluenceFunction):
    pass
