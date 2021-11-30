import numpy as np

from input import tools

particles = tools.build_particle_coordinates(5e-3, 10, 10, 10)
print("Number of particles:", np.shape(particles)[0])

neighbour_list, n_family_members = tools.build_particle_families(particles,
                                                                 5e-3 * 3)
print(type(neighbour_list))
