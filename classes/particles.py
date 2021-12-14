"""
Particle array class
--------------------
"""

# Particles or ParticleArray?
class Particles():

    def __init__(self, coordinates):
        self.coordinates = coordinates
        # self.u = np.zeros([n_nodes, n_dim])
        # self.v = np.zeros([n_nodes, n_dim])
        # self.a = np.zeros([n_nodes, n_dim])

    # TODO: see PySPH
    def add_property():
        pass

    def add_constant():
        pass

    # TODO: this would require passing in a instance of the bonds class. Would
    # this lead to circular references?
    def calculate_particle_forces():
        pass

    def update_particle_positions():
        pass
