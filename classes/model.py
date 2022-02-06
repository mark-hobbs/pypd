"""
Model class
-----------

TODO: rename classes as base or baseclasses?
"""

from tqdm import trange
import matplotlib.pyplot as plt


class Model():
    """
    Model class

    Attributes
    ----------

    Methods
    -------
    
    Notes
    -----
    * see pysph / solver / solver.py

    """

    def __init__(self, particles, bonds, simulation):
        """
        Model class constructor

        Parameters
        ----------
        particles : ParticleSet
        bonds : BondSet
        simulation : Simulation class
            Define simulation parameters

        Returns
        -------

        Notes
        -----
        """
        self.particles = particles
        self.bonds = bonds
        self.simulation = simulation

    def _single_time_step(self, i_time_step):
        """
        Single time step

        Parameters
        ----------

        Returns
        -------

        Notes
        -----
        * rename as _integration_step?
        """

        self.particles.calculate_particle_forces(self.bonds)
        self.particles.update_particle_positions(self.simulation, i_time_step)

    def run_simulation(self):
        """
        Run the simulation

        Parameters
        ----------

        Returns
        -------

        Notes
        -----

        """

        for i_time_step in trange(self.simulation.n_time_steps,
                                  desc="Simulation Progress",
                                  unit="steps"):
            self._single_time_step(i_time_step)

        self.plot_deformed_particles()

    def plot_deformed_particles(self, dsf=10):
        """
        Plot the deformed particles

        Parameters
        ----------
        dsf : int
            Displacement scale factor (default = 10)

        Returns
        -------

        Notes
        -----
        """
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        x_coords = self.particles.x[:, 0] + (self.particles.u[:, 0] * dsf)
        y_coords = self.particles.x[:, 1] + (self.particles.u[:, 1] * dsf)
        ax.scatter(x_coords, y_coords, s=2, cmap='jet')
        plt.axis('equal')
        plt.savefig('Plate', dpi=1000)
