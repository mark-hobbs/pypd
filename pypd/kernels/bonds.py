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
