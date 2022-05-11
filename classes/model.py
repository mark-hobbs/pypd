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

    def __init__(self, particles, bonds, simulation, integrator):
        """
        Model class constructor

        Parameters
        ----------
        particles : ParticleSet

        bonds : BondSet
        
        simulation : Simulation class
            Define simulation parameters

        integrator : Integrator class

        Returns
        -------

        Notes
        -----
        """
        self.particles = particles
        self.bonds = bonds
        self.simulation = simulation
        self.integrator = integrator

    def _single_time_step(self, i_time_step):
        """
        Single time step

        Parameters
        ----------

        Returns
        -------

        Notes
        -----
        """

        nf, _ = self.particles.calculate_particle_forces(self.bonds)
        self.particles.update_particle_positions(nf, self.simulation,
                                                self.integrator, i_time_step)

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

        self.particles.calculate_particle_damage(self.bonds)
        self.plot_deformed_particles(sz=1, data=self.particles.damage)

    def plot_deformed_particles(self, sz=2, dsf=10, data=None,
                                fig_title="deformed_particles"):
        """
        Plot the deformed particles

        Parameters
        ----------
        sz : int
            The marker size (particle size) in points (default = 2)

        dsf : int
            Displacement scale factor (default = 10)

        data : ndarray
            Array-like list to be mapped to colours. For example:
            particle.damage, particle.stress etc

        fig_title : str
            The figure is saved as fig_title

        Returns
        -------

        Notes
        -----
        """
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        x_coords = self.particles.x[:, 0] + (self.particles.u[:, 0] * dsf)
        y_coords = self.particles.x[:, 1] + (self.particles.u[:, 1] * dsf)
        ax.scatter(x_coords, y_coords, s=sz, c=data, cmap='jet')
        plt.axis('equal')
        plt.savefig(fig_title, dpi=300)
