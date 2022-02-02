
class Simulation():

    # Define a config file (yaml file)

    def __init__(self, dt, n_time_steps, damping):
        self.dt = dt
        self.n_time_steps = n_time_steps
        self.damping = damping
        # self.model = model
        # self.config = config

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
