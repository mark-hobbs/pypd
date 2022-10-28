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

    TODO: rename ConstitutiveLaw2D?
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
        return (9 * material.E) / (np.pi * self.t * particles.horizon**3)

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

    def __init__(self, material, particles, t, c=None, sc=None):
        """
        Linear constitutive model class constructor

        Parameters
        ----------
        material : Material class

        particles: ParticleSet class

        thickness : float
            Discretisation dx...

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
        self.t = t
        self.c = c
        self.sc = sc

        if self.c is None:
            self.c = self._calculate_bond_stiffness(material, particles)

        if self.sc is None:
            self.sc = self._calculate_sc(material, particles)

    def _calculate_sc(self, material, particles):
        """
        Calculate the critical stretch for a linear elastic material in
        two-dimensions

        Parameters
        ----------

        Returns
        -------

        Notes
        -----

        """
        return np.sqrt((4 * np.pi * material.Gf)
                       / (9 * material.E * particles.horizon))

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

    def __init__(self, material, particles, t, c=None, s0=None, sc=None,
                 beta=0.25):
        """
        Trilinear constitutive model class constructor

        Parameters
        ----------
        material : Material class

        particles: ParticleSet class

        thickness : float
            Discretisation dx...

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
        self.t = t
        self.c = c
        self.s0 = s0
        self.sc = sc
        self.beta = beta
        self.gamma = self._calculate_gamma()
        
        if self.c is None:
            self.c = self._calculate_bond_stiffness(material, particles)

        if self.s0 is None:
            self.s0 = self._calculate_s0(material)

        if self.sc is None:
            self.sc = self._calculate_sc(material, particles)

        self.s1 = self._calculate_s1()

    def _calculate_s0(self, material):
        """
        Calculate the linear elastic limit
        """
        return material.ft / material.E

    def _calculate_sc(self, material, particles):
        """
        Trilinear model - calculate the critical stretch
        """
        numerator = 10 * self.gamma * material.Gf
        denominator = (np.pi * particles.horizon**5 * self.c * self.s0
                       * (1 + (self.gamma * self.beta)))
        return (numerator / denominator) + self.s0

    def _calculate_gamma(self):
        return (3 + (2 * self.beta)) / (2 * self.beta * (1 - self.beta))

    def _calculate_s1(self):
        return self.s0 + ((self.sc - self.s0) / self.gamma)

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

    def print_parameters(self):
        """
        Print constitutive model parameters
        """
        print('{0:>10s} : {1:>12,.5E}'.format('c', self.c))
        print('{0:>10s} : {1:>12,.5E}'.format('s0', self.s0))
        print('{0:>10s} : {1:>12,.5E}'.format('sc', self.sc))
        print('{0:>10s} : {1:>12,.2f}'.format('beta', self.beta))


class NonLinear(ConstitutiveLaw):

    def __init__(self, material, particles, t, c=None, s0=None, sc=None,
                 alpha=0.25, k=25):
        """
        Non-linear constitutive model class constructor

        Parameters
        ----------
        material : Material class

        particles: ParticleSet class

        thickness : float
            Discretisation dx...

        Returns
        -------
        c : float
            Bond stiffness

        s0 : float
            Linear elastic limit

        sc : float
            Critical stretch

        alpha : float
            alpha controls the position of the transition from exponential to
            linear decay (default = 0.25)

        k : float
            k controls the rate of exponential decay (default = 25)

        Notes
        -----
        """
        self.t = t
        self.c = c
        self.s0 = s0
        self.sc = sc
        self.alpha = alpha
        self.k = k

        if self.c is None:
            self.c = self._calculate_bond_stiffness(material, particles)

        if self.s0 is None:
            self.s0 = self._calculate_s0(material)

        if self.sc is None:
            self.sc = self._calculate_sc(material, particles)

    def _calculate_s0(self, material):
        """
        Calculate the linear elastic limit
        """
        return material.ft / material.E

    def _calculate_sc(self, material, particles):
        """
        Nonlinear model - calculate the critical stretch
        """
        numerator_a = 4 * self.k * (1 - np.exp(self.k)) * (1 + self.alpha)
        numerator_b = ((self.t * self.c * particles.horizon**4 * self.s0**2
                       * ((2 * self.k)
                          - (2 * np.exp(self.k))
                          + (self.alpha * self.k)
                          - (self.alpha * self.k * np.exp(self.k) + 2)))
                       / ((4 * self.k)
                          + (np.exp(self.k) - 1)
                          * (1 + self.alpha)))
        numerator = numerator_a * (material.Gf - numerator_b)
        denominator_a = self.t * self.c * particles.horizon**4 * self.s0
        denominator_b = ((2 * self.k) - (2 * np.exp(self.k))
                         + (self.alpha * self.k)
                         - (self.alpha * self.k * np.exp(self.k)) + 2)
        denominator = denominator_a * denominator_b
        return numerator / denominator

    def print_parameters(self):
        """
        Print constitutive model parameters
        """
        print('{0:>10s} : {1:>12,.5E}'.format('c', self.c))
        print('{0:>10s} : {1:>12,.5E}'.format('s0', self.s0))
        print('{0:>10s} : {1:>12,.5E}'.format('sc', self.sc))
        print('{0:>10s} : {1:>12,.2f}'.format('alpha', self.alpha))
        print('{0:>10s} : {1:>12,.2f}'.format('k', self.k))
