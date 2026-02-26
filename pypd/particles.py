from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from numba import cuda

from .kernels.particles import (
    build_particle_families,
    compute_node_damage,
    compute_strain_energy_density,
)

if TYPE_CHECKING:
    import matplotlib.figure
    import matplotlib.collections
    from numpy.typing import NDArray
    from numba.cuda.cudadrv.devicearray import DeviceNDArray
    from .boundary_conditions import BoundaryConditions
    from .material import Material
    from .bonds import Bonds


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

    def __init__(
        self,
        x: NDArray[np.float64],
        dx: float,
        bc: BoundaryConditions,
        material: Material,
        m: float = np.pi,
        nlist: NDArray[np.int32] | None = None,
    ) -> None:
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

        self.x: NDArray[np.float64] = x
        self.n_nodes: int = np.shape(self.x)[0]
        self.n_dim: int = np.shape(self.x)[1]

        self.bc: BoundaryConditions = bc

        # TODO: this should not be an attribute of the particle set. A Mesh class is required
        self.dx: float = dx
        self.cell_area: float = dx**2
        self.cell_volume: float = dx**3

        self.horizon: float = m * dx

        self.material: Material = material

        self.nlist = nlist
        if self.nlist is None:
            self.nlist, self.n_family_members = self._build_particle_families()

        # TODO: move the following to an initialise method in Model or Simulation?
        self.f: NDArray[np.float64] = np.zeros((self.n_nodes, self.n_dim))
        self.u: NDArray[np.float64] = np.zeros((self.n_nodes, self.n_dim))
        self.v: NDArray[np.float64] = np.zeros((self.n_nodes, self.n_dim))
        self.a: NDArray[np.float64] = np.zeros((self.n_nodes, self.n_dim))

        self.damage: NDArray[np.float64] = np.zeros(self.n_nodes)
        self.W: NDArray[np.float64] = np.zeros(self.n_nodes)

        self.d_x: DeviceNDArray | None = None
        self.d_u: DeviceNDArray | None = None
        self.d_v: DeviceNDArray | None = None
        self.d_a: DeviceNDArray | None = None
        self.d_f: DeviceNDArray | None = None
        self.d_bc_flag: DeviceNDArray | None = None
        self.d_bc_unit_vector: DeviceNDArray | None = None
        self.d_nlist: DeviceNDArray | None = None

    @property
    def nlist(self) -> NDArray[np.int32]:
        return self._nlist

    @nlist.setter
    def nlist(self, value: NDArray[np.int32] | None) -> None:
        self._nlist = value

    @property
    def n_family_members(self) -> NDArray[np.int32]:
        return self._n_family_members

    @n_family_members.setter
    def n_family_members(self, value: NDArray[np.int32]) -> None:
        self._n_family_members = value

    def _build_particle_families(self) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
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

    def _host_to_device(self) -> None:
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

    def _device_to_host(self) -> None:
        """
        Move arrays from device (GPU) to host
        """
        self.d_u.copy_to_host(self.u)

    def compute_damage(self, bonds: Bonds) -> None:
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

    def compute_strain_energy_density(self, bonds: Bonds) -> None:
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

    def plot(
        self,
        fig: matplotlib.figure.Figure,
        sz: int = 1,
        dsf: int = 10,
        data: NDArray[np.float64] | None = None,
    ) -> matplotlib.collections.PatchCollection:
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
