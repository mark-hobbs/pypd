import matplotlib.pyplot as plt

from .kernels.particles import make_compute_nodal_forces, compute_nodal_forces_gpu


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

    def __init__(self, particles, bonds, penetrators=None, observations=None):
        """
        Model class constructor

        Attributes
        ----------
        particles : Particles
            The particle set, including properties such as positions,
            velocities, boundary conditions and material type

        bonds : Bonds
            The set of bonds that define the interactions between particles,
            including stiffness and damage properties

        penetrators : list, optional
            A list of penetrator objects representing external bodies
            that can interact with the particles. Default is None.

        observations : list, optional
            A list of observation objects for tracking quantities or events
            during the simulation. Default is None.

        Methods
        -------
        save_final_state_fig(...)
            Save a figure representing the final state of the simulation.
        """
        self.particles = particles
        self.bonds = bonds

        self.penetrators = penetrators
        self.observations = observations

        self.compute_particle_forces_cpu = make_compute_nodal_forces(
            bonds.constitutive_law.calculate_bond_damage
        )

    def compute_particle_forces(self, cuda_available):
        """
        Compute particle forces

        Parameters
        ----------
        cuda_available : bool
            Flag indicating if CUDA is available

        Returns
        -------
        particles.f: ndarray (float)
            Particle forces

        Notes
        -----
        * Particle forces are modified in place
        """
        if cuda_available:
            compute_nodal_forces_gpu(
                self.particles.d_f,
                self.particles.d_x,
                self.particles.d_u,
                self.particles.cell_volume,
                self.bonds.d_bondlist,
                self.bonds.d_d,
                self.bonds.d_c,
                self.bonds.d_f_x,
                self.bonds.d_f_y,
                self.bonds.d_surface_correction_factors,
            )
        else:
            self.compute_particle_forces_cpu(
                self.particles.f,
                self.particles.x,
                self.particles.u,
                self.particles.cell_volume,
                self.bonds.bondlist,
                self.bonds.d,
                self.bonds.c,
                self.bonds.f_x,
                self.bonds.f_y,
                self.bonds.surface_correction_factors,
            )

    def save_state_fig(self, sz=1, dsf=0, fig_title="damage", show_axis=True):
        """
        Save a figure of the current state of the simulation

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
