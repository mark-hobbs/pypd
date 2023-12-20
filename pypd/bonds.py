"""
Bond array class
----------------
"""

import numpy as np

from .kernels.bonds import build_bond_list, build_bond_length


# Bonds, BondArray, or BondSet?
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
        self, particles, constitutive_law, bondlist=None, surface_correction=False
    ):
        """
        BondSet class constructor

        Parameters
        ----------
        nlist : ndarray
            TODO: write description

        particles : Particle class

        material : Material class

        constitutive_law : ConstitutiveLaw class

        Returns
        -------

        Notes
        -----
        """

        self.bondlist = bondlist or self._build_bond_list(particles.nlist)
        self.n_bonds = len(self.bondlist)
        self.xi = self._calculate_bond_length(particles.x)
        self.c = np.zeros(self.n_bonds)
        self.d = np.zeros(self.n_bonds)
        self.f_x = np.zeros(self.n_bonds)
        self.f_y = np.zeros(self.n_bonds)

        if surface_correction:
            self.surface_correction_factors = (
                self._calculate_surface_correction_factors(particles)
            )
        else:
            self.surface_correction_factors = np.ones(self.n_bonds)

        self.constitutive_law = (
            constitutive_law  # Constitutive model (material_model / material_law?)
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

        Notes
        -----
        TODO: is this programming pattern good practice?

        """
        return build_bond_list(nlist)

    def calculate_bond_stiffness():
        """
        * Should this be part of the ConstitutiveModel class?
        """
        pass

    def _calculate_bond_length(self, x):
        """
        Compute the length of all bonds in the reference configuration

        Returns
        -------
        xi : ndarray (float)
            Reference bond length
        """
        return build_bond_length(x, self.bondlist)

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

    def calculate_bond_stretch():
        pass

    def calculate_bond_force(self):
        """
        Calculate bond forces

        Parameters
        ----------
        x : ndarray (float)
            Material point coordinates in the reference configuration

        Returns
        -------
        d : ndarray (float)
            Bond damage (softening parameter). The value of d will range from 0
            to 1, where 0 indicates that the bond is still in the elastic range,
            and 1 represents a bond that has failed

        Notes
        -----
        TODO: it would be inefficient to define this as a class method. A more
        efficient approach would be to implement the following methods as a
        single method in the particles class.

        1. bonds.calculate_bond_stretch(particles)
        2. bonds.calculate_bond_damage(particles)
        3. bonds.calculate_bond_force(particles)

        1. particles.calculate_particle_forces(bonds)
        """
        pass

    def calculate_bond_damage(self):
        """
        Calculate bond damage (softening parameter)

        Parameters
        ----------
        d : ndarray (float)
            Bond damage (softening parameter). The value of d will range from 0
            to 1, where 0 indicates that the bond is still in the elastic range,
            and 1 represents a bond that has failed


        Returns
        -------
        d : ndarray (float)
            Bond damage (softening parameter). The value of d will range from 0
            to 1, where 0 indicates that the bond is still in the elastic range,
            and 1 represents a bond that has failed

        Notes
        -----
        * User must select or define a constitutive law (linear / bilinear /
        trilinear / non-linear)
        * from solver.constitutive_model import trilinear
        """
        pass
