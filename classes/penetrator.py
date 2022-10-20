
import numpy as np
import itertools

from solver.calculate import (smooth_step_data,
                              calculate_contact_force)

# TODO: should Penetrator be a base class? Create a subclass for supports


class Penetrator():

    ID_iter = itertools.count()
    _registry = []

    def __init__(self, centre, radius, particles):

        self._registry.append(self)
        self.ID = next(Penetrator.ID_iter)
        self.centre = centre
        self.radius = radius
        self.search_radius = radius * 1.25
        self.family = self._build_family(particles)

    def _build_family(self, particles):
        family = []
        for i in range(particles.n_nodes):
            distance = np.sqrt(np.sum((particles.x[i] - self.centre)) ** 2)
            if distance <= self.search_radius:
                family.append(i)

        return np.array(family)

    def update_penetrator_position(self, i_time_step, n_time_steps):
        """
        Update the penetrator position

        TODO: the final displacement should not be hard coded
        TODO: this function clearly won't work properly
        """
        return self.centre + smooth_step_data(i_time_step, 0, n_time_steps,
                                              0, 0)

    def calculate_penetrator_force(self, particles, simulation, i_time_step):
        """
        Calculate the contact force between a rigid penetrator and deformable
        peridynamic body

        Parameters
        ----------
        particles : ParticleSet

        """
        position = self.update_penetrator_position(i_time_step,
                                                   simulation.n_time_steps)
        return calculate_contact_force(self.family, self.radius, position,
                                       particles.x, particles.u, particles.v,
                                       particles.material.density,
                                       particles.cell_volume, simulation.dt)
