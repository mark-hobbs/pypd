"""
Bond array class
----------------
"""

# Bonds, BondArray, or BondSet?


class Bonds():
    """
    The main class for storing the bond set.

    Attributes
    ----------
    bondlist : ndarray (int)
        Array of pairwise interactions (bond list)
    nlist : ndarray (int)
        TODO: define a new name and description
    bond_type: ndarray (int)
        Array defining the material type of every bond
    c : ndarray (float)
        Bond stiffness.
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
    """

    def __init__(self):
        pass

    def build_particle_families():
        # TODO: which class does this method belong to?
        pass

    def calculate_bond_stiffness():
        pass

    def calculate_bond_length():
        pass
    
    def calculate_bond_stretch():
        pass

    def calculate_bond_force():
        """
        bonds.calculate_bond_force(particles)
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
        """
        pass
