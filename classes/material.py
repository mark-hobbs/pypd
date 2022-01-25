"""
Material class
--------------

"""

class Material():
    """
    The main class for defining material properties

    Attributes
    ----------

    Methods
    -------
    
    Notes
    -----

    """

    def __init__(self, name, youngs_modulus, fracture_energy,
                 density, poissons_ratio, tensile_strength):
        """
        Material class constructor

        Parameters
        ----------

        Returns
        -------

        Notes
        -----
        * define a constitutive model?
            - glass.constitutive_law = linear

        """

        self.name = name
        self.youngs_modulus = youngs_modulus
        self.fracture_energy = fracture_energy
        self.density = density
        self.poissons_ratio = poissons_ratio
        self.tensile_strength = tensile_strength
