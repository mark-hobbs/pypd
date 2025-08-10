"""
Small, highly optimised computational units written using Numba
"""

import numpy as np
import sklearn.neighbors as neighbors
from numba import njit, prange, cuda


@njit(parallel=True, fastmath=True)
def compute_nodal_forces_cpu(
    x,
    u,
    cell_volume,
    bondlist,
    d,
    c,
    f_x,
    f_y,
    material_law,
    surface_correction_factors,
):
    """
    Compute particle forces - employs bondlist (cpu optimised)

    Parameters
    ----------
    x : ndarray (float)
        Material point coordinates in the reference configuration

    u : ndarray (float)
        Nodal displacement

    cell_volume : float

    bondlist : ndarray (int)
        Array of pairwise interactions (bond list)

    d : ndarray (float)
        Bond damage (softening parameter). The value of d will range from 0
        to 1, where 0 indicates that the bond is still in the elastic range,
        and 1 represents a bond that has failed

    c : float
        Bond stiffness

    material_law : function

    surface_correction_factors : ndarray (float)
    
    Returns
    -------
    node_force : ndarray (float)
        Nodal force array

    d : ndarray (float)
        Bond damage (softening parameter). The value of d will range from 0
        to 1, where 0 indicates that the bond is still in the elastic range,
        and 1 represents a bond that has failed
    """

    n_nodes = np.shape(x)[0]
    n_dimensions = np.shape(x)[1]
    n_bonds = np.shape(bondlist)[0]
    node_force = np.zeros((n_nodes, n_dimensions))

    for k_bond in prange(n_bonds):
        node_i = bondlist[k_bond, 0]
        node_j = bondlist[k_bond, 1]

        xi_x = x[node_j, 0] - x[node_i, 0]
        xi_y = x[node_j, 1] - x[node_i, 1]

        xi_eta_x = xi_x + (u[node_j, 0] - u[node_i, 0])
        xi_eta_y = xi_y + (u[node_j, 1] - u[node_i, 1])

        xi = np.sqrt(xi_x**2 + xi_y**2)
        y = np.sqrt(xi_eta_x**2 + xi_eta_y**2)
        stretch = (y - xi) / xi

        d[k_bond] = material_law(k_bond, stretch, d[k_bond])

        f = (
            stretch
            * c[k_bond]
            * (1 - d[k_bond])
            * cell_volume
            * surface_correction_factors[k_bond]
        )
        f_x[k_bond] = f * xi_eta_x / y
        f_y[k_bond] = f * xi_eta_y / y

    # Reduce bond forces to particle forces
    for k_bond in range(n_bonds):
        node_i = bondlist[k_bond, 0]
        node_j = bondlist[k_bond, 1]

        node_force[node_i, 0] += f_x[k_bond]
        node_force[node_j, 0] -= f_x[k_bond]
        node_force[node_i, 1] += f_y[k_bond]
        node_force[node_j, 1] -= f_y[k_bond]

    return node_force, d


def compute_nodal_forces_gpu():
    """
    Compute particle forces (gpu optimised)
    """
    compute_nodal_forces_kernel[grid_size, block_size]()


@cuda.jit
def compute_nodal_forces_kernel():
    """
    CUDA kernel
    """
    pass


@njit
def compute_node_damage(x, bondlist, d, n_family_members):
    """
    Compute the nodal damage

    Parameters
    ----------
    x : ndarray (float)
        Material point coordinates in the reference configuration

    bondlist : ndarray (int)
        Array of pairwise interactions (bond list)

    d : ndarray (float)
        Bond damage (softening parameter). The value of d will range from 0
        to 1, where 0 indicates that the bond is still in the elastic range,
        and 1 represents a bond that has failed

    n_family_members : ndarray (int)

    Returns
    -------
    node_damage : ndarray (float)
        The value of node_damage will range from 0 to 1, where 0 indicates that
        all bonds connected to the node are in the elastic range, and 1
        indicates that all bonds connected to the node have failed

    Notes
    -----
    """
    n_nodes = np.shape(x)[0]
    n_bonds = np.shape(bondlist)[0]
    node_damage = np.zeros((n_nodes,))

    for k_bond in range(n_bonds):
        node_i = bondlist[k_bond, 0]
        node_j = bondlist[k_bond, 1]

        node_damage[node_i] += d[k_bond]
        node_damage[node_j] += d[k_bond]

    node_damage = node_damage / n_family_members

    return node_damage


@njit(parallel=True, fastmath=True)
def compute_strain_energy_density(x, u, cell_volume, bondlist, d, c):
    """
    Compute strain energy density - employs bondlist

    Parameters
    ----------
    bondlist : ndarray (int)
        Array of pairwise interactions (bond list)

    x : ndarray (float)
        Material point coordinates in the reference configuration

    u : ndarray (float)
        Nodal displacement

    d : ndarray (float)
        Bond damage (softening parameter). The value of d will range from 0
        to 1, where 0 indicates that the bond is still in the elastic range,
        and 1 represents a bond that has failed

    c : float
        Bond stiffness

    material_law : function

    Returns
    -------
    W : ndarray (float)
        Strain energy density at each node
    """

    n_nodes = np.shape(x)[0]
    n_bonds = np.shape(bondlist)[0]
    w = np.zeros(n_bonds)
    W = np.zeros(n_nodes)

    for k_bond in prange(n_bonds):
        node_i = bondlist[k_bond, 0]
        node_j = bondlist[k_bond, 1]

        xi_x = x[node_j, 0] - x[node_i, 0]
        xi_y = x[node_j, 1] - x[node_i, 1]

        xi_eta_x = xi_x + (u[node_j, 0] - u[node_i, 0])
        xi_eta_y = xi_y + (u[node_j, 1] - u[node_i, 1])

        xi = np.sqrt(xi_x**2 + xi_y**2)
        y = np.sqrt(xi_eta_x**2 + xi_eta_y**2)
        stretch = (y - xi) / xi

        w[k_bond] = (0.5 * c[k_bond] * stretch**2 * xi) * (1 - d[k_bond]) * cell_volume

    # Reduce the micropotential (energy stored in a bond) to strain energy density
    for k_bond in range(n_bonds):
        node_i = bondlist[k_bond, 0]
        node_j = bondlist[k_bond, 1]

        W[node_i] += w[k_bond]
        W[node_j] += w[k_bond]

    return W


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
