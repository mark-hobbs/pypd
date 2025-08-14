import numpy as np
import itertools
import matplotlib.pyplot as plt

from .tools import smooth_step_data
from .kernels.penetrator import calculate_contact_force


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
    ID_iter = itertools.count()
    _registry = []

    def __init__(
        self,
        centre,
        unit_vector,
        magnitude,
        radius,
        particles,
        name="Penetrator",
        plot=False,
    ):
        self._registry.append(self)
        self.ID = next(Penetrator.ID_iter)
        self.centre = centre
        self.unit_vector = unit_vector
        self.magnitude = magnitude
        self.radius = radius
        self.search_radius = radius * 1.25
        self.family = self._build_family(particles)
        self.name = name
        if plot:
            self.plot_penetrator(particles)
        self.penetrator_force_history = []

    def _build_family(self, particles):
        family = []
        for i in range(particles.n_nodes):
            distance = np.sqrt(np.sum((particles.x[i] - self.centre) ** 2))
            if distance <= self.search_radius:
                family.append(i)

        return np.array(family)

    def update_penetrator_position(self, i_time_step, n_time_steps):
        """
        Update the penetrator position
        """
        return self.centre + (
            self.unit_vector
            * smooth_step_data(
                i_time_step, 0, n_time_steps, np.array([0, 0]), self.magnitude
            )
        )

    def calculate_penetrator_force(self, particles, simulation):
        """
        Calculate the contact force between a rigid penetrator and deformable
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
        TODO: this function does not need to return u and v
        TODO: write a decorator to save the force history
        """
        position = self.update_penetrator_position(
            simulation.i_time_step, simulation.n_time_steps
        )
        force = calculate_contact_force(
            self.family,
            self.radius,
            position,
            particles.x,
            particles.u,
            particles.v,
            particles.material.density,
            particles.cell_volume,
            simulation.dt,
        )
        self.penetrator_force_history.append(force)

    def plot(self, ax=None):
        """
        Plot the position of the penetrator at t=0
        """
        if ax is None:
            _, ax = plt.subplots()
        circle = plt.Circle(self.centre, self.radius, fill=False)
        ax.set_aspect(1)
        ax.add_patch(circle)
