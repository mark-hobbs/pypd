
import numpy as np
import sklearn.neighbors as neighbors

# TODO: should this be a class (Particles)?

def build_particle_coordinates(dx, n_div_x, n_div_y, n_div_z):

    particle_coordinates = np.zeros([n_div_x * n_div_y * n_div_z, 3])
    counter = 0

    for i_z in range(n_div_z):          # Width
        for i_y in range(n_div_y):      # Depth
            for i_x in range(n_div_x):  # Length
                coord_x = dx * i_x
                coord_y = dx * i_y
                coord_z = dx * i_z
                particle_coordinates[counter, 0] = coord_x
                particle_coordinates[counter, 1] = coord_y
                particle_coordinates[counter, 1] = coord_z
                counter += 1

    return particle_coordinates


def build_particle_families(particle_coordinates, horizon):

    # neighbour_list is a list of numpy arrays

    nnodes = np.shape(particle_coordinates)[0]

    tree = neighbors.KDTree(particle_coordinates, leaf_size=160)
    neighbour_list = tree.query_radius(particle_coordinates, r = horizon)
    print(np.shape(neighbour_list))

    # Remove identity values, as there is no bond between a node and itself
    neighbour_list = [neighbour_list[i][neighbour_list[i] != i]
                      for i in range(nnodes)]  # list comprehension

    n_family_members = [len(neighbour_list[i]) for i in range(nnodes)]
    n_family_members = np.array(n_family_members, dtype = np.intc)

    # TODO: Is this safe because 0 is used as a node ID?
    nlist = np.zeros((nnodes, n_family_members.max()), dtype=np.intc)
    
    for i in range(nnodes):
        nlist[i, :n_family_members[i]] = neighbour_list[i]

    nlist = nlist.astype(np.intc)

    return nlist


def build_penetrator():
    """Build rigid penetrator"""
    pass


def build_volume_correction_factors():
    pass


def calculate_bond_stiffness(E, delta):
    c = (12 * E) / (np.pi * delta**4)
    return c

def calculate_stable_time_step():
    pass
