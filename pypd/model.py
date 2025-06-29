
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

    def _host_to_device(self):
        """
        Move arrays from host to device (GPU)
        """
        from numba import cuda

        cuda.to_device(self.particles.x)
        cuda.to_device(self.particles.u)
        cuda.to_device(self.particles.v)
        cuda.to_device(self.particles.a)
        cuda.to_device(self.particles.f)
        cuda.to_device(self.particles.bc.flag)
        cuda.to_device(self.particles.bc.unit_vector)
        cuda.to_device(self.bonds.c)
        cuda.to_device(self.bonds.d)
        cuda.to_device(self.bonds.f_x)
        cuda.to_device(self.bonds.f_y)

    def _device_to_host(self):
        """
        Move arrays from device (GPU) to host
        """
        from numba import cuda

        cuda.to_host(self.particles.x)
        cuda.to_host(self.particles.u)
