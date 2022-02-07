"""
Constitutive law class
----------------------

Notes
-----

"""
import numpy as np


class ConstitutiveLaw():
    """
    Subclass this to define a new constitutive law. This class ensures that
    all constitutive models follow the correct format.
    """

    def __init__():
        pass

    def required_parameters():
        """
        Define the required parameters
        """
        pass

    def calculate_parameter_values():
        """
        Determine the parameter values for the implemented constitutive law
        """
        raise NotImplementedError("This method must be implemented!")

    def calculate_damage_parameter():
        """
        Calculate bond damage (softening parameter). The value of d will range
        from 0 to 1, where 0 indicates that the bond is still in the elastic
        range, and 1 represents a bond that has failed
        """
        pass


class Linear(ConstitutiveLaw):
    """
    Linear constitutive model

    Attributes
    ----------

    Methods
    -------

    Notes
    -----
    """

    def __init__(self, material):
        """
        Linear constitutive model class constructor

        Parameters
        ----------
        material : Material class

        Returns
        -------
        c : float
            Bond stiffness
        
        sc : float
            Critical stretch

        Notes
        -----
        """
        self.c = self._calculate_bond_stiffness(material)
        self.sc = self._calculate_critical_stretch(material)

    def _calculate_bond_stiffness(self, material):
        """
        Bond stiffness
            - linear elastic model
            - 2D
            - plane stress
            
        Parameters
        ----------

        Returns
        -------

        Notes
        -----
        """
        c = (9 * material.E) / (np.pi * t * horizon**3)
        return c

    def _calculate_critical_stretch(self, material):
        """
        Critical stretch
            - linear elastic model
            - 2D
            
        Parameters
        ----------

        Returns
        -------

        Notes
        -----

        """
        sc = np.sqrt((4 * np.pi * material.Gf)
                     / (9 * material.E * horizon))
        return sc


class Bilinear(ConstitutiveLaw):
    pass


class Trilinear(ConstitutiveLaw):
    pass


class NonLinear(ConstitutiveLaw):
    pass
