from .tools import calculate_stable_time_step


class Simulation:
    # Define a config file (yaml file)

    def __init__(self, n_time_steps, damping, dt=None):
        self.n_time_steps = n_time_steps
        self.damping = damping
        self.dt = dt

    @staticmethod
    def calculate_stable_dt(particles, constitutive_law, sf=0.8):
        """
        Calculate stable time step

        Parameters
        ----------
        particles : ParticleSet

        constitutive_law : ConstitutiveLaw

        sf : float
            Safety factor

        Notes
        -----
        - TODO: dx should be an attribute of a Mesh class
        - TODO: c should be an attribute of a bond
        """
        return sf * calculate_stable_time_step(
            particles.material.density,
            particles.dx,
            particles.horizon,
            constitutive_law.c,
        )

    def initiate_arrays():
        pass

    def calculate_nodal_forces():
        """
        Execute code - pure python, OpenCL etc
        """
        pass

    def update_nodal_positions():
        """
        Execute code - pure python, OpenCL etc
        (Euler, Euler-Cromer, Velocity-Verlet time integration scheme)
        """
        pass

    def time_stepping_scheme():
        """
        Run the simulation - step through time
        """
        # for t in range(n_time_steps):
        #     self.calculate_nodal_force()
        #     self.update_nodal_positions()
        pass


class Simulation_2D(Simulation):
    pass


class Simulation_3D(Simulation):
    pass
