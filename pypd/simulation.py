from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from tqdm import trange

from .integrator import EulerCromer
from .tools import calculate_stable_time_step, smooth_step_data, get_cuda_device_info
from .backend import Backend

if TYPE_CHECKING:
    from .integrator import Integrator
    from .animation import Animation
    from .model import Model
    from .particles import Particles


class Simulation:

    def __init__(
        self,
        n_time_steps: int,
        damping: float,
        dt: float | None = None,
        integrator: Integrator | None = None,
        animation: Animation | None = None,
    ) -> None:
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
        self.n_time_steps: int = n_time_steps
        self.damping: float = damping
        self.dt: float = dt
        self.integrator: Integrator = integrator if integrator is not None else EulerCromer()
        self.animation: Animation = animation
        self.i_time_step: int = 0
        self.backend: Backend | None = None

    def run(self, model: Model) -> None:
        """
        Run the simulation
        """
        self._initialise_backend(model)
        self.backend.host_to_device()

        if self.dt is None:
            self.dt = self._calculate_stable_dt(model.particles, np.max(model.bonds.c))

        iterator = trange(self.n_time_steps, unit=" steps")

        for self.i_time_step in iterator:
            self._single_time_step(model)

            if model.observations:
                for observation in model.observations:
                    observation.record_history(self.i_time_step, model.particles.u)

        self.backend.device_to_host()

        if self.animation:
            self.animation.generate_animation()

    def _single_time_step(self, model: Model) -> None:
        """
        Single time step

        TODO:
        - compute forces
        - external contact forces
        - time integration
        """
        model.particles.bc.i_magnitude = smooth_step_data(
            self.i_time_step, 0, self.n_time_steps, 0, model.particles.bc.magnitude
        )
        self.backend.compute_forces()

        if model.penetrators:
            for penetrator in model.penetrators:
                penetrator.compute_force(model.particles, self)

        self.integrator(self, model.particles)

        if self.animation and self.i_time_step % self.animation.frequency == 0:
            self.animation.save_frame(model.particles, model.bonds)

    @staticmethod
    def _calculate_stable_dt(particles: Particles, c: float, sf: float = 0.8) -> float:
        """
        Calculate stable time step

        Parameters
        ----------
        particles : Particles

        c : float
            Bond stiffness

        sf : float
            Safety factor
        """
        return sf * calculate_stable_time_step(
            particles.material.density, particles.dx, particles.horizon, c
        )

    def _initialise_backend(self, model: Model) -> None:
        """
        Initialise backend to handle device logic (GPU/CPU)
        """
        self.backend = Backend(model)
        if self.backend.cuda_available:
            get_cuda_device_info()
