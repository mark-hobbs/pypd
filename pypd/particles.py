import numpy as np
from numba import cuda

from .tools import smooth_step_data
from .kernels.particles import (
    build_particle_families,
    compute_node_damage,
    compute_strain_energy_density,
)


class Particles:
    """
    The main class for storing and managing particles (nodes).

    Attributes
    ----------
    x : ndarray(float, shape=(n_nodes, n_dim))
        Material point coordinates in the reference configuration

    n_nodes : int
        Number of particles

    n_dim : int
        Number of dimensions (2 or 3-dimensional system)

    bc : BoundaryConditions
        Boundary conditions

    dx : float
        Mesh resolution (only valid for regular meshes)

    cell_area : float
        Cell area. If a regular mesh is employed, this value will be a
        constant for all nodes

    cell_volume : float
        Cell volume. If a regular mesh is employed, this value will be a
        constant for all nodes

    horizon : float
        Horizon radius

    material : Material
        Material properties

    nlist : ndarray(int, shape=(n_nodes, max_n_family_members))
        Neighbour list for each particle, where each entry stores the indices
        of particles interacting with the corresponding particle. Padding
        entries are indicated by -1.

    n_family_members: ndarray(int, shape=(n_nodes,))
        Array specifying the number of family members for each particle

    f : ndarray(float, shape=(n_nodes, n_dim))
        Force array

    u : ndarray(float, shape=(n_nodes, n_dim))
        Displacement array

    v : ndarray(float, shape=(n_nodes, n_dim))
        Velocity array

    a : ndarray(float, shape=(n_nodes, n_dim))
        Acceleration array

    damage : ndarray(float, shape=(n_nodes,))
        The value of damage will range from 0 to 1, where 0 indicates that
        all bonds connected to the node are in the elastic range, and 1
        indicates that all bonds connected to the node have failed

    W : ndarray(float, shape=(n_nodes,))
        Strain energy density (J/m^3) at every node

    Methods
    -------

    Notes
    -----
    * Class should accept both regular and irregular meshes
    * Should dx be an attribute? Or is a Mesh class needed?
        - particles.dx
        - mesh.dx
    """

    def __init__(self, x, dx, bc, material, m=np.pi, nlist=None):
        """
        Particles class constructor

        Parameters
        ----------
        x : ndarray(float, shape=(n_nodes, n_dim))
            Material point coordinates in the reference configuration

        dx : float
            Mesh resolution (only valid for regular meshes)

        bc : BoundaryConditions

        material : Material

        m : float
            Ratio between the horizon radius and grid resolution (default
            value is pi)

        nlist : ndarray(int, shape=(n_nodes, n_family_members)), optional
            Neighbour list for each particle, where each entry stores the
            indices of particles interacting with the corresponding particle
            (n_nodes, n_family_members)

        Returns
        -------

        Notes
        -----
        """

        self.x = x
        self.n_nodes = np.shape(self.x)[0]
        self.n_dim = np.shape(self.x)[1]

        self.bc = bc

        self.dx = dx  # TODO: this should not be an attribute of the particle set. Perhaps a Mesh class is required?
        self.cell_area = dx**2
        self.cell_volume = dx**3

        self.horizon = m * dx

        self.material = material
        # compute_nodal_forces_cpu = make_compute_nodal_forces()

        self.nlist = nlist
        if self.nlist is None:
            self.nlist, self.n_family_members = self._build_particle_families()

        # TODO: move the following to an initialise method in Model or Simulation?
        self.f = np.zeros((self.n_nodes, self.n_dim))
        self.u = np.zeros((self.n_nodes, self.n_dim))
        self.v = np.zeros((self.n_nodes, self.n_dim))
        self.a = np.zeros((self.n_nodes, self.n_dim))

        self.damage = np.zeros(self.n_nodes)
        self.W = np.zeros(self.n_nodes)

        self.d_x = None
        self.d_u = None
        self.d_v = None
        self.d_a = None
        self.d_f = None
        self.d_bc_flag = None
        self.d_bc_unit_vector = None

    @property
    def nlist(self):
        return self._nlist

    @nlist.setter
    def nlist(self, value):
        self._nlist = value

    @property
    def n_family_members(self):
        return self._n_family_members

    @n_family_members.setter
    def n_family_members(self, value):
        self._n_family_members = value

    def _build_particle_families(self):
        """
        Build particle families

        Returns
        -------
        nlist : ndarray(int, shape=(n_nodes, n_family_members))
            Neighbour list for each particle, where each entry stores the
            indices of particles interacting with the corresponding particle

        n_family_members: ndarray(int, shape=(n_nodes,))
            Array specifying the number of family members for each particle

        Notes
        -----
        """
        return build_particle_families(self.x, self.horizon)

    def _host_to_device(self):
        """
        Move arrays from host to device (GPU)
        """
        self.d_nlist = cuda.to_device(self.nlist)
        self.d_x = cuda.to_device(self.x)
        self.d_u = cuda.to_device(self.u)
        self.d_v = cuda.to_device(self.v)
        self.d_a = cuda.to_device(self.a)
        self.d_f = cuda.to_device(self.f)
        self.d_bc_flag = cuda.to_device(self.bc.flag)
        self.d_bc_unit_vector = cuda.to_device(self.bc.unit_vector)

    def _device_to_host(self):
        """
        Move arrays from device (GPU) to host
        """
        self.d_u.copy_to_host(self.u)

    def compute_damage(self, bonds):
        """
        Compute particle damage

        Parameters
        ----------
        bonds : Bonds

        Returns
        -------
        damage : ndarray(float, shape=(n_nodes,))
            The value of damage will range from 0 to 1, where 0 indicates that
            all bonds connected to the node are in the elastic range, and 1
            indicates that all bonds connected to the node have failed
        """
        self.damage = compute_node_damage(
            self.x, bonds.bondlist, bonds.d, self.n_family_members
        )

    def compute_strain_energy_density(self, bonds):
        """
        Compute the strain energy density (J/m^3) at every node

        Parameters
        ----------
        bonds : Bonds

        Returns
        -------
        W : ndarray(float, shape=(n_nodes,))
            Strain energy density
        """
        self.W = compute_strain_energy_density(
            self.x,
            self.u,
            self.cell_volume,
            bonds.bondlist,
            bonds.d,
            bonds.c,
        )

    def plot(self, fig, sz=1, dsf=10, data=None):
        """
        Scatter plot of displaced particle positions

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The top-level container that holds all elements of a Matplotlib
            plot

        sz : int
            The marker size (particle size) in points (default = 2)

        dsf : int
            Displacement scale factor (default = 10)

        data : ndarray
            Array-like list to be mapped to colours. For example:
            particle.damage, particle.stress etc

        Returns
        -------
        The ax.scatter() function in Matplotlib returns a PathCollection
        object. This object represents a collection of scatter points or
        markers on a plot. It contains information about the plotted markers,
        including their positions, sizes, colours, and other properties.

        Notes
        -----
        """
        x_coords = self.x[:, 0] + (self.u[:, 0] * dsf)
        y_coords = self.x[:, 1] + (self.u[:, 1] * dsf)

        ax = fig.add_subplot(1, 1, 1)
        return ax.scatter(x_coords, y_coords, s=sz, c=data, cmap="jet")
