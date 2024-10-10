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
        self.constitutive_law = self.bonds.constitutive_law
        self.simulation = simulation
        self.integrator = integrator
        self.penetrators = penetrators
        self.observations = observations
        self.animation = animation

        if self.simulation.dt is None:
            self.simulation.dt = self.simulation.calculate_stable_dt(
                self.particles, self.constitutive_law
            )

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

        self.particles.compute_forces(
            self.bonds, self.constitutive_law.calculate_bond_damage
        )
        self.particles.update_positions(self.simulation, self.integrator, i_time_step)

        if self.penetrators:
            for penetrator in self.penetrators:
                penetrator.calculate_penetrator_force(
                    self.particles, self.simulation, i_time_step
                )

        if self.animation:
            if i_time_step % self.animation.frequency == 0:
                self.animation.save_frame(self.particles, self.bonds)

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

        if self.animation:
            self.animation.generate_animation()

    def save_final_state_fig(self, sz=1, dsf=0, fig_title="damage", show_axis=True):
        """
        Save a figure of the final state of the simulation

        Parameters
        ----------
        sz : int
            The marker size (particle size) in points (default = 1)

        dsf : int
            Displacement scale factor (default = 0)

        fig_title : str
            The figure is saved as fig_title

        show_axis : bool
            Display the axis (default = True)

        Returns
        -------

        Notes
        -----
        """
        fig = plt.figure(figsize=(12, 6))
        self.particles.compute_damage(self.bonds)
        self.particles.plot(fig, sz=sz, dsf=dsf, data=self.particles.damage)

        ax = fig.gca()
        ax.set_aspect("equal", "box")
        if not show_axis:
            ax.axis("off")

        fig.tight_layout()
        fig.savefig(fig_title, dpi=300)
