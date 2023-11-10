"""
Model class
-----------

TODO: rename classes as base or baseclasses?
"""

from tqdm import trange
import matplotlib.pyplot as plt


class Model:
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

    def __init__(
        self,
        particles,
        bonds,
        simulation,
        integrator,
        constitutive_law,
        penetrators=None,
        observations=None,
        animation=None,
    ):
        """
        Model class constructor

        Parameters
        ----------
        particles : ParticleSet

        bonds : BondSet

        simulation : Simulation class
            Define simulation parameters

        integrator : Integrator class

        constitutive_law : ConstitutiveLaw class

        penetrators: list
            List of penetrator objects/instances

        Returns
        -------

        Notes
        -----
        """
        self.particles = particles
        self.bonds = bonds
        self.simulation = simulation
        self.integrator = integrator
        self.constitutive_law = constitutive_law
        self.penetrators = penetrators
        self.observations = observations
        self.animation = animation

        if self.simulation.dt is None:
            self.simulation.dt = self.simulation.calculate_stable_dt(
                self.particles, self.constitutive_law
            )
            print(self.simulation.dt)

    def _single_time_step(self, i_time_step):
        """
        Single time step

        Parameters
        ----------

        Returns
        -------
        simulation_data : SimulationData class

        Notes
        -----
        """

        nf, _ = self.particles.calculate_particle_forces(
            self.bonds, self.constitutive_law.calculate_bond_damage
        )
        self.particles.update_particle_positions(
            nf, self.simulation, self.integrator, i_time_step
        )

        if self.penetrators:
            for penetrator in self.penetrators:
                penetrator.calculate_penetrator_force(
                    self.particles, self.simulation, i_time_step
                )
        
        if self.animation:
            pass

    def run_simulation(self):
        """
        Run the simulation

        Parameters
        ----------

        Returns
        -------
        history : SimulationData class
            History of the simulation run

        Notes
        -----
        TODO: clean up observation.record_history()

        """
        for i_time_step in trange(
            self.simulation.n_time_steps, desc="Simulation Progress", unit="steps"
        ):
            self._single_time_step(i_time_step)

            if self.observations:
                for observation in self.observations:
                    observation.record_history(i_time_step, self.particles.u)

    def plot_damage(self, sz=1, dsf=0, fig_title="damage"):
        """
        Plot the damaged particles

        Parameters
        ----------
        sz : int
            The marker size (particle size) in points (default = 1)

        dsf : int
            Displacement scale factor (default = 0)

        fig_title : str
            The figure is saved as fig_title

        Returns
        -------

        Notes
        -----
        TODO: rename as model.save_final_state_fig() and save the figure in 
        this method
        """
        self.particles.calculate_particle_damage(self.bonds)
        self.particles.plot_particles(
            sz=sz, dsf=dsf, data=self.particles.damage, fig_title=fig_title
        )
