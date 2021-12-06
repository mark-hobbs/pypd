
class Material():

    def __init__(self, youngs_modulus, fracture_energy,
                 density, poissons_ratio):

        self.youngs_modulus = youngs_modulus
        self.fracture_energy = fracture_energy
        self.density = density
        self.poissons_ratio = poissons_ratio
