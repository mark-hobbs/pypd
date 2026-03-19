from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, ClassVar

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from numpy.typing import NDArray

from .kernels.penetrator import compute_contact_force
from .tools import smooth_step_data

if TYPE_CHECKING:
    from .particles import Particles
    from .simulation import Simulation


class Penetrator:
    """
    Represents a rigid penetrator that interacts with deformable bodies.

    Attributes
    ----------
    ID : int
        Unique identifier

    centre : ndarray(float, shape=(n_dim,))
        Centre position of the penetrator

    unit_vector : ndarray(float, shape=(n_dim,))
        Direction of movement

    magnitude : float
        Magnitude of the penetrator displacement

    radius : float
        Radius of the penetrator

    search_radius : float
        Radius within which to search for interacting particles

    family : ndarray(int, shape=(n_family_members,))
        Indices of particles that are within the search radius

    name : str
        Name of the penetrator

    penetrator_force_history : list
        History of forces applied by the penetrator

    Notes
    -----
    TODO: should Penetrator be a base class? Create a subclass for supports
    """

    ID_iter: ClassVar[Any] = itertools.count()
    _registry: ClassVar[list[Penetrator]] = []

    def __init__(
        self,
        centre: NDArray[np.float64],
        unit_vector: NDArray[np.float64],
        magnitude: float,
        radius: float,
        particles: Particles,
        name: str = "Penetrator",
        plot: bool = False,
    ) -> None:
        self._registry.append(self)
        self.ID: int = next(Penetrator.ID_iter)
        self.name: str = name
        self.centre: NDArray[np.float64] = centre
        self.unit_vector: NDArray[np.float64] = unit_vector
        self.magnitude: float = magnitude
        self.radius: float = radius
        self.search_radius: float = radius * 1.25
        self.family: NDArray[np.int_] = self._build_family(particles)
        self.k: float = self._compute_k(particles)
        if plot:
            self.plot_penetrator(particles)
        self.penetrator_force_history: list[NDArray[np.float64]] = []

        # self.compute_force = None

    def compile_cpu(self) -> None:
        pass

    def compile_gpu(self) -> None:
        pass

    def _build_family(self, particles: Particles) -> NDArray[np.int_]:
        family: list[int] = []
        for i in range(particles.n_nodes):
            distance = float(np.sqrt(np.sum((particles.x[i] - self.centre) ** 2)))
            if distance <= self.search_radius:
                family.append(i)

        return np.array(family, dtype=np.int_)

    def _compute_k(self, particles: Particles) -> float:
        """
        Compute contact stiffness K (N/m)

        Parameters
        ----------
        material : Material

        Notes
        -----
        Section 2.2.4 | https://arxiv.org/pdf/2408.06556
        """
        return float(particles.dx * particles.material.k)

    def update_position(
        self, i_time_step: int, n_time_steps: int
    ) -> NDArray[np.float64]:
        """
        Update the penetrator position
        """
        return self.centre + (
            self.unit_vector
            * smooth_step_data(
                i_time_step, 0, n_time_steps, np.array([0, 0]), self.magnitude
            )
        )

    def compute_force(self, particles: Particles, simulation: Simulation) -> None:
        """
        Compute the contact force between a rigid penetrator and deformable
        peridynamic body

        Parameters
        ----------
        particles : Particles

        Returns
        -------
        u : ndarray(float, shape=(n_nodes, n_dim))
            Updated displacement array

        v : ndarray(float, shape=(n_nodes, n_dim))
            Updated velocity array

        contact_force : ndarray(float, shape=(n_dim,))
            Resultant force components

        Notes
        -----
        TODO: write a decorator to save the force history
        """
        position = self.update_position(simulation.i_time_step, simulation.n_time_steps)
        force = compute_contact_force(
            self.family,
            self.radius,
            position,
            particles.x,
            particles.u,
            particles.f,
            particles.cell_volume,
            k=self.k,
        )
        self.penetrator_force_history.append(force)

    def plot(self, ax: Axes | None = None) -> Axes:
        """
        Plot the position of the penetrator at t=0
        """
        if ax is None:
            _, ax = plt.subplots()
        circle = plt.Circle(self.centre, self.radius, fill=False)
        ax.set_aspect(1)
        ax.add_patch(circle)
        return ax
