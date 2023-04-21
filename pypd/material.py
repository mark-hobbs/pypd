"""
Material class
--------------

"""


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

    def __init__(self, name, E, Gf, density, ft, nu=None):
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

        self.name = name
        self.E = E
        self.Gf = Gf
        self.density = density
        self.ft = ft
        self.nu = nu
