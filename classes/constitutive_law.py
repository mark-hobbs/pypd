"""
Constitutive law class
----------------------

Notes
-----

"""
import numpy as np
from numba import njit

from solver.constitutive_model import linear, trilinear


class ConstitutiveLaw():
    """
    Subclass this to define a new constitutive law. This class ensures that
    all constitutive models follow the correct format.
    """

    def __init__():
        pass

    def _calculate_bond_stiffness(self, material, particles):
        """
        Bond stiffness
            - linear elastic model
            - 2D
            - plane stress

        Parameters
        ----------
        material :
            Instance of material class

        Returns
        -------
        c : float
            Bond stiffness

        Notes
        -----
        TODO: this function is generic to all material models
        """
        t = 2.5E-3  # TODO: do not hardcode values
        return (9 * material.E) / (np.pi * t * particles.horizon**3)

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

    def calculate_bond_damage():
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
    * Examine compiling classes with @jitclass
    * How do we employ a material model?
        - bond.material_model.calculate_bond_damage()
    * Should the class inherit from ConstitutiveLaw?
        - class Linear(ConstitutiveLaw):
    """

    def __init__(self, material, particles, c=None, sc=None):
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
        * TODO: passing an instance of particles is probably bad design and
        should be improved
        """
        self.c = c
        self.sc = sc

        if self.c is None:
            self.c = self._calculate_bond_stiffness(material, particles)

        if self.sc is None:
            self.sc = self._calculate_critical_stretch(material, particles)

    def _calculate_critical_stretch(self, material, particles):
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
                     / (9 * material.E * particles.horizon))
        return sc

    @staticmethod
    def calculate_bond_damage(sc):
        """
        Calculate bond damage

        Parameters
        ----------
        sc : float

        Returns
        -------
        wrapper : function
            Return a function with the call statement:
                - calculate_bond_damage(stretch, d)
            The parameters specific to the material model are wrapped...

        Notes
        -----
        """
        @njit
        def wrapper(stretch, d):
            """
            Calculate bond damage

            Parameters
            ----------
            stretch : float
                Bond stretch

            d : float
                Bond damage (softening parameter) at time t. The value of d
                will range from 0 to 1, where 0 indicates that the bond is
                still in the elastic range, and 1 represents a bond that has
                failed

            Returns
            -------
            d : float
                Bond damage (softening parameter) at time t+1. The value of d
                will range from 0 to 1, where 0 indicates that the bond is
                still in the elastic range, and 1 represents a bond that has
                failed

            Notes
            -----
            * Examine closures and factory functions
            """
            return linear(stretch, d, sc)

        return wrapper

    def calculate_nodal_forces(self):
        """
        Calculate nodal force

        Parameters
        ----------

        Returns
        -------

        Notes
        -----
        * Ideally this method would not be required.
        * Methods in this class should be concerned with the behaviour of a
        single bond
        * Called by particles.calculate_particle_forces()
        """
        pass


class Bilinear(ConstitutiveLaw):
    pass


class Trilinear(ConstitutiveLaw):

    def __init__(self, material, particles, c=None, s0=None, s1=None, sc=None,
                 beta=0.25):
        """
        Trilinear constitutive model class constructor

        Parameters
        ----------
        material : Material class

        Returns
        -------
        c : float
            Bond stiffness

        s0 : float
            Linear elastic limit

        s1 : float

        sc : float
            Critical stretch

        beta : float
            Kink point in the trilinear model (default = 0.25)

        Notes
        -----
        """
        self.c = c
        self.s0 = s0
        self.s1 = s1
        self.sc = sc
        self.beta = beta

        if self.c is None:
            self.c = self._calculate_bond_stiffness(material, particles)

        if self.s0 is None:
            self.s0 = self._calculate_linear_elastic_limit(material)

        if self.sc is None:
            self.sc = self._calculate_critical_stretch()

    def _calculate_linear_elastic_limit(self, material):
        return material.ft / material.E

    def _calculate_critical_stretch(self):
        pass

    @staticmethod
    def calculate_bond_damage(s0, s1, sc, beta):
        """
        Calculate bond damage

        Parameters
        ----------
        sc : float

        Returns
        -------
        wrapper : function
            Return a function with the call statement:
                - calculate_bond_damage(stretch, d)
            The parameters specific to the material model are wrapped...

        Notes
        -----
        """
        @njit
        def wrapper(stretch, d):
            """
            Calculate bond damage

            Parameters
            ----------
            stretch : float
                Bond stretch

            d : float
                Bond damage (softening parameter) at time t. The value of d
                will range from 0 to 1, where 0 indicates that the bond is
                still in the elastic range, and 1 represents a bond that has
                failed

            Returns
            -------
            d : float
                Bond damage (softening parameter) at time t+1. The value of d
                will range from 0 to 1, where 0 indicates that the bond is
                still in the elastic range, and 1 represents a bond that has
                failed

            Notes
            -----
            * Examine closures and factory functions
            """
            return trilinear(stretch, d, s0, s1, sc, beta)

        return wrapper


class NonLinear(ConstitutiveLaw):
    pass
