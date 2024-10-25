"""
Bond array class
----------------
"""

import numpy as np

from .kernels.bonds import build_bond_list, build_bond_length
from .influence import Constant
from .constitutive_law import Linear
from .tools import determine_intersection, rebuild_node_families


class BondSet:
    """
    The main class for storing the bond set.

    Attributes
    ----------
    bondlist : ndarray (int)
        Array of pairwise interactions (bond list)

    nlist : ndarray (int)
        TODO: define a new name and description
        TODO: ndarray or list of numpy arrays?

    xi : ndarray (float)
        Reference bond length

    material: ndarray (int)
        Array defining the material type of every bond
        TODO: name - bonds.material or bonds.type?

    c : ndarray (float)
        Bond stiffness

    d : ndarray (float)
        Bond damage (softening parameter). The value of d will range from 0
        to 1, where 0 indicates that the bond is still in the elastic range,
        and 1 represents a bond that has failed

    volume_correction_factors : ndarray (float)
        Array of volume correction factors (to improve spatial integration
        accuracy)

    lambda : ndarray (float)
        Array of surface correction factors (to correct the peridynamic
        surface effect). Also known as stiffness correction factors.

    stretch : ndarray (float)
        Bond stretch (dimensionless)

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
        particles,
        constitutive_law=None,
        influence=None,
        bondlist=None,
        surface_correction=False,
        notch=None,
        damage_on=True,
    ):
        """
        BondSet class constructor

        Parameters
        ----------
        particles : Particle class

        constitutive_law : ConstitutiveLaw class

        influence_function : InfluenceFunction class

        notch : tuple of points defining the notch (optional)
            A tuple containing two points (P1, P2) that define the line of the notch

        """
        self.bondlist = bondlist or self._build_bond_list(particles.nlist)

        if notch is not None:
            self.bondlist, particles.n_family_members = self._build_notch(
                particles, notch
            )

        self.n_bonds = len(self.bondlist)
        self.xi = self._calculate_bond_length(particles.x)

        if influence is None:
            self.influence = Constant(particles, self.xi)
        elif isinstance(influence, type):
            self.influence = influence(particles, self.xi)

        self.c = self._compute_bond_stiffness()
        self.d = np.zeros(self.n_bonds)
        self.f_x = np.zeros(self.n_bonds)
        self.f_y = np.zeros(self.n_bonds)

        if surface_correction:
            self.surface_correction_factors = (
                self._calculate_surface_correction_factors(particles)
            )
        else:
            self.surface_correction_factors = np.ones(self.n_bonds)

        if constitutive_law is None:
            self.constitutive_law = Linear(
                particles, c=self.c, t=particles.dx, damage_on=damage_on
            )
        elif isinstance(constitutive_law, type):
            self.constitutive_law = constitutive_law(
                particles, c=self.c, t=particles.dx
            )

    def _build_bond_list(self, nlist):
        """
        Build bond list

        Parameters
        ----------

        Returns
        -------
        bondlist : ndarray (int)
            Array of pairwise interactions (bond list)
        """
        return build_bond_list(nlist)

    def _calculate_bond_length(self, x):
        """
        Compute the length of all bonds in the reference configuration

        Returns
        -------
        xi : ndarray (float)
            Reference bond length
        """
        return build_bond_length(x, self.bondlist)

    def _compute_bond_stiffness(self):
        """
        Compute the stiffness of all bonds

        Returns
        -------
        c : ndarray (float)
            Bond stiffness
        """
        return self.influence()

    def _calculate_surface_correction_factors(self, particles):
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

    def _build_notch(self, particles, notch):
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

        reduced_bondlist = np.delete(self.bondlist, mask, axis=0)
        n_family_members = rebuild_node_families(n_nodes, reduced_bondlist)

        return reduced_bondlist, n_family_members
