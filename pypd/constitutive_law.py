from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
from numba import njit, cuda
from numpy.typing import NDArray

from .kernels.constitutive_law import linear, linear_gpu, trilinear, nonlinear

if TYPE_CHECKING:
    from .particles import Particles


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

    def __init__(self) -> None:
        pass

    def _calculate_sc(self, *args: Any, **kwargs: Any) -> NDArray[np.float64]:
        """
        Calculate the critical stretch
        """
        raise NotImplementedError("This method must be implemented!")

    @staticmethod
    def _make_material_law(*args: Any, **kwargs: Any) -> Any:
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

    def __init__(
        self,
        particles: Particles,
        c: NDArray[np.float64],
        t: float,
        sc: NDArray[np.float64] | None = None,
        damage_on: bool = True,
    ) -> None:
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
        self.c: NDArray[np.float64] = c
        self.t: float = t
        self.damage_on: bool = damage_on
        self.calculate_bond_damage: Any = None

        self.s0: NDArray[np.float64] | None = None
        self.s1: NDArray[np.float64] | None = None
        self.sc: NDArray[np.float64] = self._calculate_sc(particles)

    def compile_cpu(self) -> None:
        self.calculate_bond_damage = self._make_material_law(self.sc, self.damage_on)

    def compile_gpu(self) -> None:
        self.s0 = np.full_like(self.sc, np.inf)
        self.s1 = np.full_like(self.sc, np.inf)
        self.calculate_bond_damage = linear_gpu

    def _calculate_sc(self, particles: Particles) -> NDArray[np.float64]:
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
    def _make_material_law(
        sc: NDArray[np.float64], damage_on: bool
    ) -> Any:
        """
        Make material law

        Factory function that encapsulates model parameters and provides a
        consistent call interface for computing bond damage

        Parameters
        ----------
        sc : ndarray(float, shape=(n_bonds,))
            Critical stretch

        damage_on : bool

        Returns
        -------
        material_law : function
            Return a function with the call statement:
                - material_law(stretch, d)

        Notes
        -----
        """
        if damage_on:

            @njit
            def material_law(i, stretch, d):
                """
                Material law (calculate bond damage)

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
            def material_law(i, stretch, d):
                """
                Returns
                -------
                d: ndarray(float, shape=(n_bonds,))
                    An array of zeros with the same size as the input array `d`,
                    indicating no bond damage.
                """
                return 0

        return material_law


class Bilinear(ConstitutiveLaw):
    pass


class Trilinear(ConstitutiveLaw):

    def __init__(
        self,
        particles: Particles,
        c: NDArray[np.float64],
        t: float,
        s0: NDArray[np.float64] | None = None,
        sc: NDArray[np.float64] | None = None,
        beta: float = 0.25,
        **kwargs: Any,
    ) -> None:
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
        self.c: NDArray[np.float64] = c
        self.t: float = t
        self.beta: float = beta
        self.gamma: float = self._calculate_gamma()
        self.s0: NDArray[np.float64] = s0 or self._calculate_s0(particles)
        self.sc: NDArray[np.float64] = sc or self._calculate_sc(particles)
        self.s1: NDArray[np.float64] = self._calculate_s1()
        self.calculate_bond_damage: Any = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    def compile_cpu(self) -> None:
        self.calculate_bond_damage = self._make_material_law(
            self.s0, self.s1, self.sc, self.beta
        )

    def _calculate_s0(self, particles: Particles) -> NDArray[np.float64]:
        """
        Calculate the linear elastic limit
        """
        return particles.material.ft / particles.material.E

    def _calculate_sc(self, particles: Particles) -> NDArray[np.float64]:
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

    def _calculate_gamma(self) -> float:
        return (3 + (2 * self.beta)) / (2 * self.beta * (1 - self.beta))

    def _calculate_s1(self) -> NDArray[np.float64]:
        return self.s0 + ((self.sc - self.s0) / self.gamma)

    @staticmethod
    def _make_material_law(
        s0: NDArray[np.float64],
        s1: NDArray[np.float64],
        sc: NDArray[np.float64],
        beta: float,
    ) -> Any:
        """
        Make material law

        Factory function that encapsulates model parameters and provides a
        consistent call interface for computing bond damage

        Parameters
        ----------
        s0 : ndarray(float, shape=(n_bonds,))

        s1 : ndarray(float, shape=(n_bonds,))

        sc : ndarray(float, shape=(n_bonds,))

        beta : float
            Kink point in the trilinear model (default = 0.25)

        Returns
        -------
        material_law : function
            Return a function with the call statement:
                - material_law(stretch, d)

        Notes
        -----
        """

        @njit
        def material_law(i, stretch, d):
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

        return material_law

    @staticmethod
    def _make_material_law_gpu(beta: float) -> Any:
        """
        Create device function and setup arrays
        """

        @cuda.jit(device=True)
        def material_law(s, d, s0, s1, sc):
            """
            Material law (calculate bond damage) device function
            """
            return trilinear_gpu(s, d, s0, s1, sc, beta)

        return material_law

    def print_parameters(self) -> None:
        """
        Print constitutive model parameters
        """
        print("{0:>10s} : {1:>12,.5E}".format("c", self.c))
        print("{0:>10s} : {1:>12,.5E}".format("s0", self.s0))
        print("{0:>10s} : {1:>12,.5E}".format("sc", self.sc))
        print("{0:>10s} : {1:>12,.2f}".format("beta", self.beta))


class NonLinear(ConstitutiveLaw):

    def __init__(
        self,
        particles: Particles,
        c: NDArray[np.float64],
        t: float,
        s0: NDArray[np.float64] | None = None,
        sc: NDArray[np.float64] | None = None,
        alpha: float = 0.25,
        k: float = 25,
        **kwargs: Any,
    ) -> None:
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
        self.c: NDArray[np.float64] = c
        self.t: float = t
        self.alpha: float = alpha
        self.k: float = k
        self.s0: NDArray[np.float64] = s0 or self._calculate_s0(particles)
        self.sc: NDArray[np.float64] = sc or self._calculate_sc(particles)
        self.calculate_bond_damage: Any = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    def compile_cpu(self) -> None:
        self.calculate_bond_damage = self._make_material_law(
            self.s0, self.sc, self.alpha, self.k
        )

    def _calculate_s0(self, particles: Particles) -> NDArray[np.float64]:
        """
        Calculate the linear elastic limit
        """
        return particles.material.ft / particles.material.E

    def _calculate_sc(self, particles: Particles) -> NDArray[np.float64]:
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
    def _make_material_law(
        s0: NDArray[np.float64],
        sc: NDArray[np.float64],
        alpha: float,
        k: float,
    ) -> Any:
        """
        Make material law

        Factory function that encapsulates model parameters and provides a
        consistent call interface for computing bond damage

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
        material_law : function
            Return a function with the call statement:
                - material_law(stretch, d)

        Notes
        -----
        """

        @njit
        def material_law(i, stretch, d):
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

        return material_law

    def print_parameters(self) -> None:
        """
        Print constitutive model parameters
        """
        print("{0:>10s} : {1:>12,.5E}".format("c", self.c))
        print("{0:>10s} : {1:>12,.5E}".format("s0", self.s0))
        print("{0:>10s} : {1:>12,.5E}".format("sc", self.sc))
        print("{0:>10s} : {1:>12,.2f}".format("alpha", self.alpha))
        print("{0:>10s} : {1:>12,.2f}".format("k", self.k))
