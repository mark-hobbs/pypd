from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class BoundaryConditions:
    """
    The main class for defining the boundary conditions

    Attributes
    ----------
    flag : ndarray(int, shape=(n_particles, 2))
        0 - no boundary condition
        1 - the node is subject to a boundary condition

    unit_vector : ndarray(float, shape=(n_particles, n_dim))
        Unit vector defining the direction of the boundary
        condition

    magnitude : float
        Magnitude of the applied force/displacement

    i_magnitude : float
        Magnitude of the applied force/displacement at time step i

    Methods
    -------

    Notes
    -----
    * Should this class inherit from the ParticleSet class (i.e. child class)?
    * applied displacement / applied force / constraint

    """

    def __init__(
        self,
        flag: NDArray[np.int_],
        unit_vector: NDArray[np.float64],
        magnitude: float,
    ) -> None:
        """
        BoundaryConditions class constructor

        Parameters
        ----------
        i_magnitude : float
            Magnitude at time step i

        Returns
        -------

        Notes
        -----
        * TODO: implement magnitude

        """
        self.flag: NDArray[np.int_] = flag
        self.unit_vector: NDArray[np.float64] = unit_vector
        self.magnitude: float = magnitude
        self.i_magnitude: float | None = None


class DisplacementBoundaryCondition(BoundaryConditions):
    def __init__(
        self,
        flag: NDArray[np.int_],
        unit_vector: NDArray[np.float64],
        magnitude: float,
    ) -> None:
        super().__init__(flag, unit_vector, magnitude)

    def _applied_displacement_magnitude(self) -> None:
        """
        self.magnitude = smooth_step_data()
        """
        pass


class ForceBoundaryCondition(BoundaryConditions):
    pass
