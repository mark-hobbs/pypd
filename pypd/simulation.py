from .tools import calculate_stable_time_step


class Simulation:

    def __init__(self, n_time_steps, damping, dt=None):
        self.n_time_steps = n_time_steps
        self.damping = damping
        self.dt = dt

    @staticmethod
    def calculate_stable_dt(particles, c, sf=0.5):
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


class Simulation_2D(Simulation):
    pass


class Simulation_3D(Simulation):
    pass
