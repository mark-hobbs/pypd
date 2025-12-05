"""
Small, highly optimised computational units written using Numba
"""

import numpy as np
from numba import njit, prange


def build_bond_list(nlist):
    """
    Build bond list
    """
    bondlist = [
        [i, j] for i, neighbours in enumerate(nlist) for j in neighbours if i < j
    ]
    bondlist = np.array(bondlist, dtype=np.intc)

    return bondlist


def map_to_neighbour_list(bondlist, property):
    """
    Map a per-bond property (n_bonds,) to neighbour arrays
    (n_nodes, max_n_neighbours)
    """
    n_nodes = bondlist.max() + 1
    property_lists = [[] for _ in range(n_nodes)]

    for k, (i, j) in enumerate(bondlist):
        p = property[k]
        property_lists[i].append(p)
        property_lists[j].append(p)

    max_n_neighbours = max(len(lst) for lst in property_lists)
    property_array = np.zeros((n_nodes, max_n_neighbours), dtype=np.float32)

    for i, lst in enumerate(property_lists):
        property_array[i, : len(lst)] = lst

    return property_array


@njit(parallel=True)
def build_bond_length(x, bondlist):
    """
    Build the bond length array
    """
    n_bonds = np.shape(bondlist)[0]
    xi = np.zeros(n_bonds)

    for k_bond in prange(n_bonds):
        node_i = bondlist[k_bond, 0]
        node_j = bondlist[k_bond, 1]

        xi_x = x[node_j, 0] - x[node_i, 0]
        xi_y = x[node_j, 1] - x[node_i, 1]

        xi[k_bond] = np.sqrt(xi_x**2 + xi_y**2)

    return xi
