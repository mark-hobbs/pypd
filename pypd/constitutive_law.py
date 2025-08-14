
import numpy as np
from numba import njit

from .kernels.constitutive_law import linear, trilinear, nonlinear


class ConstitutiveLaw:
    """
    Subclass this to define a new constitutive law. This class ensures that
    all constitutive models follow the correct format.

    Attributes
    ----------
    material : Material
        An instance of the Material class representing the material properties

    c : ndarray(float, shape=(n_bonds,))
        Bond stiffness (micromodulus)

    influence : InfluenceFunction
        An instance of the InfluenceFunction class. The influence function,
        also referred to as the weight function or kernel function, describes
        how the interaction between particles diminishes with increasing distance.

    Methods
    -------
    """

    def __init__(self):
        pass

    def _calculate_sc():
        """
        Calculate the critical stretch
        """
        raise NotImplementedError("This method must be implemented!")

    def _calculate_bond_damage():
        """
        Calculate bond damage (softening parameter). The value of d will range
        from 0 to 1, where 0 indicates that the bond is still in the elastic
        range, and 1 represents a bond that has failed
        """
        raise NotImplementedError("This method must be implemented!")


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

    def __init__(self, particles, c, t, sc=None, damage_on=True):
        """
        Linear constitutive model class constructor

        Parameters
        ----------
        particles: ParticleSet class

        thickness : float
            In a 2D problem, the thickness is equivalent to the discretisation
            resolution, denoted as dx.

        Returns
        -------
        c : ndarray(float, shape=(n_bonds,))
            Bond stiffness

        sc : ndarray(float, shape=(n_bonds,))
            Critical stretch

        Notes
        -----
        * TODO: passing an instance of particles is probably bad design and
        should be improved
        """
        self.c = c
        self.t = t
        self.sc = self._calculate_sc(particles)
        self.damage_on = damage_on
        self.calculate_bond_damage = self._calculate_bond_damage(
            self.sc, self.damage_on
        )

    def _calculate_sc(self, particles):
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
        return np.full(
            len(self.c),
            np.sqrt(
                (4 * np.pi * particles.material.Gf)
                / (9 * particles.material.E * particles.horizon)
            ),
        )

    @staticmethod
    def _calculate_bond_damage(sc, damage_on):
        """
        Calculate bond damage

        Parameters
        ----------
        sc : ndarray(float, shape=(n_bonds,))
            Critical stretch
    
        damage_on : bool

        Returns
        -------
        wrapper : function
            Return a function with the call statement:
                - calculate_bond_damage(stretch, d)
            The parameters specific to the material model are wrapped...

        Notes
        -----
        """
        if damage_on:

            @njit
            def wrapper(i, stretch, d):
                """
                Calculate bond damage

                Parameters
                ----------
                stretch : ndarray(float, shape=(n_bonds,))
                    Bond stretch

                d : ndarray(float, shape=(n_bonds,))
                    Bond damage (softening parameter) at time t. The value of d
                    will range from 0 to 1, where 0 indicates that the bond is
                    still in the elastic range, and 1 represents a bond that has
                    failed

                Returns
                -------
                d : ndarray(float, shape=(n_bonds,))
                    Bond damage (softening parameter) at time t+1. The value of d
                    will range from 0 to 1, where 0 indicates that the bond is
                    still in the elastic range, and 1 represents a bond that has
                    failed

                Notes
                -----
                * Examine closures and factory functions
                """
                return linear(i, stretch, d, sc)

        else:

            @njit
            def wrapper(i, stretch, d):
                """
                Returns
                -------
                d: ndarray(float, shape=(n_bonds,))
                    An array of zeros with the same size as the input array `d`,
                    indicating no bond damage.
                """
                return 0

        return wrapper


class Bilinear(ConstitutiveLaw):
    pass


class Trilinear(ConstitutiveLaw):

    def __init__(self, particles, c, t, s0=None, sc=None, beta=0.25, **kwargs):
        """
        Trilinear constitutive model class constructor

        Parameters
        ----------
        particles: ParticleSet class

        thickness : float
            For 2D problems, the thickness is equivalent to dx
        
        Returns
        -------
        c : ndarray(float, shape=(n_bonds,))
            Bond stiffness

        s0 : ndarray(float, shape=(n_bonds,))
            Linear elastic limit

        s1 : ndarray(float, shape=(n_bonds,))

        sc : ndarray(float, shape=(n_bonds,))
            Critical stretch

        beta : float
            Kink point in the trilinear model (default = 0.25)

        Notes
        -----
        """
        self.c = c
        self.t = t
        self.beta = beta
        self.gamma = self._calculate_gamma()
        self.s0 = s0 or self._calculate_s0(particles)
        self.sc = sc or self._calculate_sc(particles)
        self.s1 = self._calculate_s1()
        self.calculate_bond_damage = self._calculate_bond_damage(
            self.s0, self.s1, self.sc, self.beta
        )

        for key, value in kwargs.items():
            setattr(self, key, value)

    def _calculate_s0(self, particles):
        """
        Calculate the linear elastic limit
        """
        return particles.material.ft / particles.material.E

    def _calculate_sc(self, particles):
        """
        Trilinear model (2D case) - calculate the critical stretch
        """
        numerator = 4 * self.gamma * particles.material.Gf
        denominator = (
            self.t
            * particles.horizon**4
            * self.c
            * self.s0
            * (1 + (self.gamma * self.beta))
        )
        return (numerator / denominator) + self.s0

    def _calculate_gamma(self):
        return (3 + (2 * self.beta)) / (2 * self.beta * (1 - self.beta))

    def _calculate_s1(self):
        return self.s0 + ((self.sc - self.s0) / self.gamma)

    @staticmethod
    def _calculate_bond_damage(s0, s1, sc, beta):
        """
        Calculate bond damage

        Parameters
        ----------
        s0 : ndarray(float, shape=(n_bonds,))

        s1 : ndarray(float, shape=(n_bonds,))

        sc : ndarray(float, shape=(n_bonds,))

        beta : float
            Kink point in the trilinear model (default = 0.25)

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
        def wrapper(i, stretch, d):
            """
            Calculate bond damage

            Parameters
            ----------
            stretch : ndarray(float, shape=(n_bonds,))
                Bond stretch

            d : ndarray(float, shape=(n_bonds,))
                Bond damage (softening parameter) at time t. The value of d
                will range from 0 to 1, where 0 indicates that the bond is
                still in the elastic range, and 1 represents a bond that has
                failed

            Returns
            -------
            d : ndarray(float, shape=(n_bonds,))
                Bond damage (softening parameter) at time t+1. The value of d
                will range from 0 to 1, where 0 indicates that the bond is
                still in the elastic range, and 1 represents a bond that has
                failed

            Notes
            -----
            * Examine closures and factory functions
            """
            return trilinear(i, stretch, d, s0, s1, sc, beta)

        return wrapper

    def print_parameters(self):
        """
        Print constitutive model parameters
        """
        print("{0:>10s} : {1:>12,.5E}".format("c", self.c))
        print("{0:>10s} : {1:>12,.5E}".format("s0", self.s0))
        print("{0:>10s} : {1:>12,.5E}".format("sc", self.sc))
        print("{0:>10s} : {1:>12,.2f}".format("beta", self.beta))


class NonLinear(ConstitutiveLaw):

    def __init__(self, particles, c, t, s0=None, sc=None, alpha=0.25, k=25, **kwargs):
        """
        Non-linear constitutive model class constructor

        Parameters
        ----------
        particles: ParticleSet class

        thickness : float

        Returns
        -------
        c : ndarray(float, shape=(n_bonds,))
            Bond stiffness

        s0 : ndarray(float, shape=(n_bonds,))
            Linear elastic limit

        sc : ndarray(float, shape=(n_bonds,))
            Critical stretch

        alpha : float
            alpha controls the position of the transition from exponential to
            linear decay (default = 0.25)

        k : float
            k controls the rate of exponential decay (default = 25)

        Notes
        -----
        """
        self.c = c
        self.t = t
        self.alpha = alpha
        self.k = k
        self.s0 = s0 or self._calculate_s0(particles)
        self.sc = sc or self._calculate_sc(particles)
        self.calculate_bond_damage = self._calculate_bond_damage(
            self.s0, self.sc, self.alpha, self.k
        )

        for key, value in kwargs.items():
            setattr(self, key, value)

    def _calculate_s0(self, particles):
        """
        Calculate the linear elastic limit
        """
        return particles.material.ft / particles.material.E

    def _calculate_sc(self, particles):
        """
        Nonlinear model (2D case) - calculate the critical stretch
        """
        numerator_a = 4 * self.k * (1 - np.exp(self.k)) * (1 + self.alpha)
        numerator_b = (
            self.t
            * self.c
            * particles.horizon**4
            * self.s0**2
            * (
                (2 * self.k)
                - (2 * np.exp(self.k))
                + (self.alpha * self.k)
                - (self.alpha * self.k * np.exp(self.k) + 2)
            )
        ) / ((4 * self.k) + (np.exp(self.k) - 1) * (1 + self.alpha))
        numerator = numerator_a * (particles.material.Gf - numerator_b)
        denominator_a = self.t * self.c * particles.horizon**4 * self.s0
        denominator_b = (
            (2 * self.k)
            - (2 * np.exp(self.k))
            + (self.alpha * self.k)
            - (self.alpha * self.k * np.exp(self.k))
            + 2
        )
        denominator = denominator_a * denominator_b
        return numerator / denominator

    @staticmethod
    def _calculate_bond_damage(s0, sc, alpha, k):
        """
        Calculate bond damage

        Parameters
        ----------
        s0 : ndarray(float, shape=(n_bonds,))

        sc : ndarray(float, shape=(n_bonds,))

        alpha : float
            alpha controls the position of the transition from exponential to
            linear decay (default = 0.25)

        k : float
            k controls the rate of exponential decay (default = 25)

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
        def wrapper(i, stretch, d):
            """
            Calculate bond damage

            Parameters
            ----------
            stretch : ndarray(float, shape=(n_bonds,))
                Bond stretch

            d : ndarray(float, shape=(n_bonds,))
                Bond damage (softening parameter) at time t. The value of d
                will range from 0 to 1, where 0 indicates that the bond is
                still in the elastic range, and 1 represents a bond that has
                failed

            Returns
            -------
            d : ndarray(float, shape=(n_bonds,))
                Bond damage (softening parameter) at time t+1. The value of d
                will range from 0 to 1, where 0 indicates that the bond is
                still in the elastic range, and 1 represents a bond that has
                failed

            Notes
            -----
            * Examine closures and factory functions
            """
            return nonlinear(i, stretch, d, s0, sc, alpha, k)

        return wrapper

    def print_parameters(self):
        """
        Print constitutive model parameters
        """
        print("{0:>10s} : {1:>12,.5E}".format("c", self.c))
        print("{0:>10s} : {1:>12,.5E}".format("s0", self.s0))
        print("{0:>10s} : {1:>12,.5E}".format("sc", self.sc))
        print("{0:>10s} : {1:>12,.2f}".format("alpha", self.alpha))
        print("{0:>10s} : {1:>12,.2f}".format("k", self.k))
