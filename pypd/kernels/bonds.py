"""
Small, highly optimised computational units written using Numba
"""

import numpy as np
import sklearn.neighbors as neighbors
from numba import njit, prange


def build_particle_families(x, horizon):
    """
    Build particle families

    Parameters
    ----------
    x : ndarray (float)
        Material point coordinates in the reference configuration

    horizon : float
        Material point horizon (non-local length scale)

    Returns
    -------
    nlist : list of numpy arrays (int)
        TODO: define a new name and description

    Notes
    -----
    TODO: include a discussion of the algorithm

    """
    n_nodes = np.shape(x)[0]

    tree = neighbors.KDTree(x, leaf_size=160)
    neighbour_list = tree.query_radius(x, r=horizon)

    # Remove identity values, as there is no bond between a node and itself
    neighbour_list = [neighbour_list[i][neighbour_list[i] != i] for i in range(n_nodes)]

    n_family_members = [len(neighbour_list[i]) for i in range(n_nodes)]
    n_family_members = np.array(n_family_members, dtype=np.intc)

    nlist = np.ones((n_nodes, n_family_members.max()), dtype=np.intc) * -1

    for i in range(n_nodes):
        nlist[i, : n_family_members[i]] = neighbour_list[i]

    nlist = nlist.astype(np.intc)

    return nlist, n_family_members


def build_bond_list(nlist):
    """
    Build bond list
    """
    bondlist = [
        [i, j] for i, neighbours in enumerate(nlist) for j in neighbours if i < j
    ]
    bondlist = np.array(bondlist, dtype=np.intc)

    return bondlist


@njit(parallel=True)
def build_bond_length(x, bondlist):
    """Build the bond length array"""
    n_bonds = np.shape(bondlist)[0]
    xi = np.zeros(n_bonds)

    for k_bond in prange(n_bonds):
        node_i = bondlist[k_bond, 0]
        node_j = bondlist[k_bond, 1]

        xi_x = x[node_j, 0] - x[node_i, 0]
        xi_y = x[node_j, 1] - x[node_i, 1]

        xi[k_bond] = np.sqrt(xi_x**2 + xi_y**2)

    return xi


def calculate_bond_stiffness(E, delta):
    c = (12 * E) / (np.pi * delta**4)
    return c
