
import numpy as np

from numba import int16, float64
from numba.experimental import jitclass

from solver.calculate import (smooth_step_data,
                              calculate_contact_force)

spec = [
    ('ID', int16),
    ('centre', float64[:]),
    ('radius', float64),
    ('search_radius', float64),
    ('family', int16[:])
]


# TODO: should Penetrator be a base class? Create a subclass for supports

@jitclass(spec)
class Penetrator():

    # Numba doesn't support class members (class parameters). Therefore it is
    # not currently possible to assign a unique ID to every class instance by
    # incrementing a counter

    # Use a structured numpy array to avoid issues with passing an instance of
    # a class to a jit compiled function

    def __init__(self, ID, centre, radius, search_radius, particles):

        self.ID = ID
        self.centre = centre
        self.radius = radius
        self.search_radius = search_radius
        self.family = self._build_family(particles)

    def _build_family(self, particles):
        family = []
        for i in range(particles.n_nodes):
            distance = np.sqrt(np.sum((particles.x[i, :]
                                      - self.centre[:, :])) ** 2)
            if distance <= self.search_radius:
                family.append(i)

        return np.array(family)

    def update_penetrator_position(self, i_time_step, n_time_steps):
        """
        Update the penetrator position

        TODO: the final displacement should not be hard coded
        """
        return self.centre + smooth_step_data(i_time_step, 0, n_time_steps,
                                              0, 1e-4)

    def calculate_penetrator_force(self, particles, simulation, i_time_step):
        """
        Calculate the contact force between a rigid penetrator and deformable
        peridynamic body

        Parameters
        ----------
        particles : ParticleSet

        """
        penetrator_position = self.update_penetrator_position(i_time_step,
                                                              simulation.n_time_steps)
        return calculate_contact_force(penetrator_position)

    def update_penetrator_position():
        """
        Update the penetrator position
        """
        pass
