"""
Model class
-----------

TODO: rename classes as base or baseclasses?
"""

from classes.simulation import Simulation
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

    def run_simulation(self):
        """
        Run the simulation

        Parameters
        ----------

        Returns
        -------

        Notes
        -----
        * Should I have a seperate method _single_time_step()?
        
        """

        self.particles.calculate_particle_forces(self.bonds)
        self.particles.update_particle_positions(self.simulation)


