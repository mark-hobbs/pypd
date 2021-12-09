
class Material():

    def __init__(self, youngs_modulus, fracture_energy,
                 density, poissons_ratio, tensile_strength):

        self.youngs_modulus = youngs_modulus
        self.fracture_energy = fracture_energy
        self.density = density
        self.poissons_ratio = poissons_ratio
        self.tensile_strength = tensile_strength
