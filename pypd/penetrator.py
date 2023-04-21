
import numpy as np
import itertools
import matplotlib.pyplot as plt

from calculate import (smooth_step_data,
                       calculate_contact_force)

# TODO: should Penetrator be a base class? Create a subclass for supports


class Penetrator():

    ID_iter = itertools.count()
    _registry = []

    def __init__(self, centre, unit_vector, magnitude, radius, particles,
                name="Penetrator", plot=False):
        self._registry.append(self)
        self.ID = next(Penetrator.ID_iter)
        self.centre = centre
        self.unit_vector = unit_vector
        self.magnitude = magnitude
        self.radius = radius
        self.search_radius = radius * 1.25
        self.family = self._build_family(particles)
        self.name = name
        if plot == True:
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
        return self.centre + (self.unit_vector
                              * smooth_step_data(i_time_step,
                                                 0, n_time_steps,
                                                 np.array([0, 0]),
                                                 self.magnitude))

    def calculate_penetrator_force(self, particles, simulation, i_time_step):
        """
        Calculate the contact force between a rigid penetrator and deformable
        peridynamic body

        Parameters
        ----------
        particles : ParticleSet

        Returns
        -------
        u : ndarray
            Updated displacement array

        v : ndarray
            Updated velocity array

        contact_force : ndarray
            Resultant force components

        Notes
        -----
        TODO: this function does not need to return u and v
        TODO: write a decorator to save the force history
        """
        position = self.update_penetrator_position(i_time_step,
                                                   simulation.n_time_steps)
        force = calculate_contact_force(self.family, self.radius, position,
                                        particles.x, particles.u, particles.v,
                                        particles.material.density,
                                        particles.cell_volume, simulation.dt)
        self.penetrator_force_history.append(force)

    def plot_penetrator(self, particles):
        """
        Plot the position of the penetrator at t=0
        """
        _, ax = plt.subplots()
        circle = plt.Circle(self.centre,
                            self.radius,
                            fill = False)
        ax.set_aspect( 1 )
        ax.add_patch(circle)
        ax.scatter(particles.x[self.family, 0], particles.x[self.family, 1])
        plt.title(self.name)
        plt.show()