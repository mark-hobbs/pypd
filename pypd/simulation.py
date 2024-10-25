import numpy as np
from tqdm import trange

from .tools import calculate_stable_time_step


class Simulation:

    def __init__(self, n_time_steps, damping, dt=None, integrator=None):
        self.n_time_steps = n_time_steps
        self.damping = damping
        self.dt = dt
        self.integrator = integrator

    def run(self, model):
        if self.dt is None:
            self.dt = self._calculate_stable_dt(model.particles, np.max(model.bonds.c))

        for i in trange(self.n_time_steps, desc="Simulation progress", unit="steps"):
            self._single_time_step(i, model)

    def _single_time_step(self, i, model):
        model.particles.compute_forces(model.bonds)
        model.particles.update_positions(i, self)

    @staticmethod
    def _calculate_stable_dt(particles, c, sf=0.8):
        """
        Calculate stable time step

        Parameters
        ----------
        particles : ParticleSet

        c : float
            Bond stiffness

        sf : float
            Safety factor
        """
        return sf * calculate_stable_time_step(
            particles.material.density, particles.dx, particles.horizon, c
        )
