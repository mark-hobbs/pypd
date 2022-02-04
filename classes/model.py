"""
Model class
-----------

TODO: rename classes as base or baseclasses?
"""

from tqdm import trange
import numpy as np

from .simulation import Simulation
from .particles import ParticleSet
from .bonds import BondSet


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
