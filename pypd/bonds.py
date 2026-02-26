from __future__ import annotations
from typing import Any, TYPE_CHECKING

import numpy as np
from numba import cuda

from .kernels.bonds import build_bond_list, build_bond_length, map_to_neighbour_list
from .influence import Constant
from .constitutive_law import Linear
from .tools import determine_intersection, rebuild_neighbour_list

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from numba.cuda.cudadrv.devicearray import DeviceNDArray
    from .particles import Particles
    from .influence import InfluenceFunction
    from .constitutive_law import ConstitutiveLaw


class Bonds:
    """
    The main class for storing bonds.

    Attributes
    ----------
    bondlist : ndarray(int, shape=(n_bonds, 2))
        Array of pairwise interactions (bond list) of size (n_bonds, 2)

    n_bonds : int
        Number of bonds

    xi : ndarray(float, shape=(n_bonds,))
        Reference bond length

    influence : InfluenceFunction
        Influence function

    c : ndarray(float, shape=(n_bonds,))
        Bond stiffness

    d : ndarray(float, shape=(n_bonds,))
        Bond damage (softening parameter). The value of d will range from 0
        to 1, where 0 indicates that the bond is still in the elastic range,
        and 1 represents a bond that has failed

    f_x : ndarray(float, shape=(n_bonds,))
        Bond force in the x-direction

    f_y : ndarray(float, shape=(n_bonds,))
        Bond force in the y-direction

    surface_correction_factors : ndarray(float, shape=(n_bonds,))
        Array of surface correction factors (to correct the peridynamic
        surface effect). Also known as stiffness correction factors.

    constitutive_law : ConstitutiveLaw

    Methods
    -------

    Notes
    -----
    * Code design
        - assign the same properties to all bonds
        - uniquely assign properties to individual bonds
    """

    def __init__(
        self,
        particles: Particles,
        constitutive_law: type[ConstitutiveLaw] | None = None,
        constitutive_law_params: dict[str, Any] | None = None,
        influence: type[InfluenceFunction] | None = None,
        bondlist: NDArray[np.int32] | None = None,
        surface_correction: bool = False,
        notch: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None,
        damage_on: bool = True,
    ) -> None:
        """
        Bonds class constructor

        Parameters
        ----------
        particles : Particles

        constitutive_law : ConstitutiveLaw

        constitutive_law_params : dict, optional
            Parameters for the constitutive law. If not provided, default
            parameters will be used.

        influence : InfluenceFunction

        bondlist : ndarray(int, shape=(n_bonds, 2)), optional
            Array of pairwise interactions (bond list) of size (n_bonds, 2).
            If not provided, it will be built using the neighbour list.

        surface_correction : bool, optional
            Flag indicating if surface correction factors should be applied.
            Default is False.

        notch : tuple of points defining the notch (optional)
            A tuple containing two points (P1, P2) that define the line of the notch

        damage_on : bool, optional
            Flag indicating if damage should be considered. Default is True.
        """
        self.bondlist: NDArray[np.int32] = bondlist or self._build_bond_list(
            particles.nlist
        )

        if notch is not None:
            self.bondlist, particles.nlist, particles.n_family_members = (
                self._build_notch(particles, notch)
            )

        self.n_bonds: int = len(self.bondlist)
        self.xi: NDArray[np.float64] = self._calculate_bond_length(particles.x)

        if influence is None:
            self.influence: InfluenceFunction = Constant(particles, self.xi)
        elif isinstance(influence, type):
            self.influence = influence(particles, self.xi)

        self.c: NDArray[np.float64] = self._compute_bond_stiffness()
        self.d: NDArray[np.float64] = np.zeros(self.n_bonds)
        self.f_x: NDArray[np.float64] = np.zeros(self.n_bonds)
        self.f_y: NDArray[np.float64] = np.zeros(self.n_bonds)

        if surface_correction:
            self.surface_correction_factors: NDArray[np.float64] = (
                self._calculate_surface_correction_factors(particles)
            )
        else:
            self.surface_correction_factors = np.ones(self.n_bonds)

        if constitutive_law is None:
            self.constitutive_law: ConstitutiveLaw = Linear(
                particles, c=self.c, t=particles.dx, damage_on=damage_on
            )
        elif isinstance(constitutive_law, type):
            constitutive_law_params = constitutive_law_params or {}
            self.constitutive_law = constitutive_law(
                particles, c=self.c, t=particles.dx, **constitutive_law_params
            )

        self.d_c: DeviceNDArray | None = None
        self.d_d: DeviceNDArray | None = None
        self.d_surface_correction_factors: DeviceNDArray | None = None
        self.d_s0: DeviceNDArray | None = None
        self.d_s1: DeviceNDArray | None = None
        self.d_sc: DeviceNDArray | None = None

    def _build_bond_list(self, nlist: NDArray[np.int32]) -> NDArray[np.int32]:
        """
        Build bond list

        Parameters
        ----------
        nlist : ndarray(int, shape=(n_particles, n_neighbours))
            Neighbour list

        Returns
        -------
        bondlist : ndarray(int, shape=(n_bonds, 2))
            Array of pairwise interactions (bond list)
        """
        return build_bond_list(nlist)

    def _calculate_bond_length(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the length of all bonds in the reference configuration

        Parameters
        ----------
        x : ndarray(float, shape=(n_particles, n_dim))
            Particle positions in the reference configuration

        Returns
        -------
        xi : ndarray(float, shape=(n_bonds,))
            Reference bond length
        """
        return build_bond_length(x, self.bondlist)

    def _compute_bond_stiffness(self) -> NDArray[np.float64]:
        """
        Compute the stiffness of all bonds

        Returns
        -------
        c : ndarray(float, shape=(n_bonds,))
            Bond stiffness
        """
        return self.influence()

    def _calculate_surface_correction_factors(
        self, particles: Particles
    ) -> NDArray[np.float64]:
        """
        Compute surface correction factors (lambda) using the volume
        correction method, first proposed in Chapter 2 of Ref. [1]

        Bobaru, F., Foster, J., Geubelle, P., and Silling, S. (2017). Handbook
        of Peridynamic Modeling. Chapman and Hall/CRC, New York, 1st edition.
        """
        surface_correction_factors = np.ones(self.n_bonds)
        v0 = np.pi * particles.horizon**2

        for k_bond in range(self.n_bonds):
            node_i = self.bondlist[k_bond, 0]
            node_j = self.bondlist[k_bond, 1]
            v_i = particles.n_family_members[node_i] * particles.cell_area
            v_j = particles.n_family_members[node_j] * particles.cell_area
            surface_correction_factors[k_bond] = (2 * v0) / (v_i + v_j)

        return surface_correction_factors

    def _build_notch(
        self,
        particles: Particles,
        notch: tuple[NDArray[np.float64], NDArray[np.float64]]
    ) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]]:
        n_nodes = np.shape(particles.x)[0]
        n_bonds = np.shape(self.bondlist)[0]

        P1 = notch[0]
        P2 = notch[1]

        mask = []

        for k_bond in range(n_bonds):
            node_i = self.bondlist[k_bond, 0]
            node_j = self.bondlist[k_bond, 1]

            P3 = particles.x[node_i]
            P4 = particles.x[node_j]

            intersect = determine_intersection(P1, P2, P3, P4)

            if intersect:
                mask.append(k_bond)

        filtered_bondlist = np.delete(self.bondlist, mask, axis=0)
        filtered_nlist, filtered_n_family_members = rebuild_neighbour_list(
            n_nodes, filtered_bondlist
        )

        return filtered_bondlist, filtered_nlist, filtered_n_family_members

    def _map_to_neighbour_list(
        self, property: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Map a per-bond property (n_bonds,) to neighbour arrays
        (n_nodes, max_n_neighbours)
        """
        return map_to_neighbour_list(self.bondlist, property)

    def _host_to_device(self) -> None:
        """
        Move arrays from host to device (GPU)
        """
        c: NDArray[np.float64] = self._map_to_neighbour_list(self.c)
        d: NDArray[np.float64] = self._map_to_neighbour_list(self.d)
        surface_correction_factors: NDArray[np.float64] = self._map_to_neighbour_list(
            self.surface_correction_factors
        )
        s0: NDArray[np.float64] = self._map_to_neighbour_list(self.constitutive_law.s0)
        s1: NDArray[np.float64] = self._map_to_neighbour_list(self.constitutive_law.s1)
        sc: NDArray[np.float64] = self._map_to_neighbour_list(self.constitutive_law.sc)

        self.d_c = cuda.to_device(c)
        self.d_d = cuda.to_device(d)
        self.d_surface_correction_factors = cuda.to_device(surface_correction_factors)
        self.d_s0 = cuda.to_device(s0)
        self.d_s1 = cuda.to_device(s1)
        self.d_sc = cuda.to_device(sc)

    def _device_to_host(self):
        """
        Move arrays from device (GPU) to host

        TODO:
        - Map device arrays to bondlist-shaped host arrays
        - TypeError: incompatible dtype: float32 vs. float64
        """
        # self.d_d.copy_to_host(self.d)
        return 0
