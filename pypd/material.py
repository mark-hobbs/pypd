from __future__ import annotations


class Material:
    """
    The main class for defining material properties

    Attributes
    ----------

    Methods
    -------

    Notes
    -----

    """

    def __init__(
        self,
        name: str,
        E: float,
        Gf: float,
        density: float,
        ft: float,
        nu: float | None = None,
    ) -> None:
        """
        Material class constructor

        Parameters
        ----------
        name : str
            Material name (steel etc)

        E : float
            Young's modulus (or modulus of elasticity) (units)

        Gf : float
            Fracture energy (N/m)

        density : float
            Material density (kg/m^3)

        ft : float
            Tensile strength (units)

        nu : float
            Poisson's ratio (default = None)

        Returns
        -------

        Notes
        -----
        * define a constitutive model?
            - glass.constitutive_law = linear

        * material_flag - flag the material type of every particle

        """

        self.name: str = name
        self.E: float = E
        self.Gf: float = Gf
        self.density: float = density
        self.ft: float = ft
        self.nu: float | None = nu

        if self.nu is not None:
            self.k: float | None = self._compute_bulk_modulus()
        else:
            self.k = None

    def _compute_bulk_modulus(self) -> float:
        """
        Compute Bulk Modulus (K): resistance to uniform compression
        """
        if self.nu is None:
            raise ValueError(
                "Poisson's ratio (nu) must be defined to calculate Bulk Modulus."
            )

        if self.nu >= 0.5:
            return float("inf")

        return self.E / (3 * (1 - 2 * self.nu))
