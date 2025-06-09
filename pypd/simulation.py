import numpy as np
from numba import cuda
from tqdm import trange

from .integrator import EulerCromer
from .tools import calculate_stable_time_step, get_cuda_device_info


class Simulation:

    def __init__(self, n_time_steps, damping, dt=None, integrator=None, animation=None):
        """
        Initialise the Simulation class

        Parameters
        ----------
        n_time_steps : int
            Number of time steps

        damping : float
            Local damping coefficient (Kg/m^3s)

        dt : float, optional
            Time step size. If None, it will be calculated based on stability
            conditions

        integrator : Integrator, optional
            Numerical integrator. Defaults to EulerCromer()

        animation : Animation, optional
            Animation object for visualising the simulation (default is None)
        """
        self.n_time_steps = n_time_steps
        self.damping = damping
        self.dt = dt
        self.integrator = integrator if integrator is not None else EulerCromer()
        self.animation = animation
        self.i_time_step = 0

        self.cuda_available = self._is_cuda_available()
        print(f"Is CUDA available: {self.cuda_available}")
        if self.cuda_available:
            get_cuda_device_info()

    def run(self, model):
        """
        Run the simulation
        """
        if self.cuda_available:
            model._host_to_device()

        if self.dt is None:
            self.dt = self._calculate_stable_dt(model.particles, np.max(model.bonds.c))

        for self.i_time_step in trange(
            self.n_time_steps, desc="Simulation progress", unit="steps"
        ):
            self._single_time_step(model)

            if model.observations:
                for observation in model.observations:
                    observation.record_history(self.i_time_step, model.particles.u)

        if self.cuda_available:
            model._device_to_host()

        if self.animation:
            self.animation.generate_animation()

    def _single_time_step(self, model):
        """
        Single time step
        """
        model.particles.compute_forces(model.bonds, self.cuda_available)
        model.particles.update_positions(self)

        if model.penetrators:
            for penetrator in model.penetrators:
                penetrator.calculate_penetrator_force(model.particles, self)

        if self.animation and self.i_time_step % self.animation.frequency == 0:
            self.animation.save_frame(model.particles, model.bonds)

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

    def _is_cuda_available(self):
        """
        Check if CUDA is available
        """
        return cuda.is_available()
