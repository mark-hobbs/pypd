"""
Small, highly optimised computational units written using Numba
"""
import math

import numpy as np
import sklearn.neighbors as neighbors
from numba import njit, prange, cuda


def make_compute_nodal_forces(material_law):
    """
    Factory function that returns a JIT compiled compute_nodal_forces()
    with the given material law baked in

    Parameters
    ----------
    material_law : function
        A function that defines the material behaviour

    Returns
    -------
    compute_nodal_forces : function
        A function that computes nodal forces

    TODO: while the cuda_available flag could be injected into this function,
    hardware controls belongs to Simulation not Model
    """

    @njit(parallel=True, fastmath=True)
    def compute_nodal_forces_cpu(
        node_force,
        x,
        u,
        cell_volume,
        bondlist,
        d,
        c,
        f_x,
        f_y,
        surface_correction_factors,
    ):
        """
        Compute particle forces - employs bondlist (cpu optimised)

        Parameters
        ----------
        node_force : np.ndarray(float, shape=(n_nodes, n_dim))
            Nodal force array

        x : np.ndarray(float, shape=(n_nodes, n_dim))
            Material point coordinates in the reference configuration

        u : np.ndarray(float, shape=(n_nodes, n_dim))
            Nodal displacement

        cell_volume : float

        bondlist : np.ndarray(int, shape=(n_bonds, 2))
            Array of pairwise interactions (bond list)

        d : np.ndarray(float, shape=(n_bonds,))
            Bond damage (softening parameter). The value of d will range from 0
            to 1, where 0 indicates that the bond is still in the elastic range,
            and 1 represents a bond that has failed

        c : np.ndarray(float, shape=(n_bonds,))
            Bond stiffness

        surface_correction_factors : np.ndarray(float, shape=(n_bonds,))

        Returns
        -------
        node_force : np.ndarray(float, shape=(n_nodes, n_dimensions))
            Nodal force array

        d : np.ndarray(float, shape=(n_bonds,))
            Bond damage (softening parameter). The value of d will range from 0
            to 1, where 0 indicates that the bond is still in the elastic range,
            and 1 represents a bond that has failed

        Notes
        -----
        * node_force and d are modified in place and returned for clarity
        """
        n_bonds = np.shape(bondlist)[0]
        node_force[:] = 0.0

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

    return compute_nodal_forces_cpu


def compute_nodal_forces_gpu(
    node_force, x, u, cell_volume, bondlist, d, c, f_x, f_y, surface_correction_factors
):
    """
    Compute particle forces (gpu optimised)
    """
    BLOCKS_PER_GRID = bondlist.shape[0]
    THREADS_PER_BLOCK = 256
    compute_nodal_forces_kernel[BLOCKS_PER_GRID, THREADS_PER_BLOCK](
        node_force,
        x,
        u,
        cell_volume,
        bondlist,
        d,
        c,
        f_x,
        f_y,
        surface_correction_factors,
    )


@cuda.jit
def compute_nodal_forces_kernel(
    node_force, x, u, cell_volume, nlist, d, c, f_x, f_y, surface_correction_factors
):
    """
    TODO: 
     - How do I reset node_forces to 0 after every time step?
     - bondlist data structure is not suitable for GPU

    PLACEHOLDER
    ------------
    n_nodes = node_force.shape[0]
    n_dimensions = node_force.shape[1]

    idx = cuda.grid(1)
    total = n_nodes * n_dimensions

    if idx < total:
        node_i = idx // n_dimensions
        dof = idx % n_dimensions
        node_force[node_i, dof] = 1.0
    """

    shared_x = cuda.shared.array(THREADS_PER_BLOCK, dtype=node_force.dtype)
    shared_y = cuda.shared.array(THREADS_PER_BLOCK, dtype=node_force.dtype)

    node_i = cuda.blockIdx.x
    thread_id = cuda.threadIdx.x
    n_family = nlist.shape[1]

    val_x = 0.0
    val_y = 0.0

    if thread_id < n_family:
        node_j = nlist[node_i, thread_id]

        xi_x = x[node_j, 0] - x[node_i, 0]
        xi_y = x[node_j, 1] - x[node_i, 1]

        xi_eta_x = xi_x + (u[node_j, 0] - u[node_i, 0])
        xi_eta_y = xi_y + (u[node_j, 1] - u[node_i, 1])

        xi = math.sqrt(xi_x**2 + xi_y**2)
        y = math.sqrt(xi_eta_x**2 + xi_eta_y**2)
        stretch = (y - xi) / xi

        d[node_i, thread_id] = 1.0  # placeholder

        f = (
            stretch
            * c[node_i, thread_id]
            * (1 - d[node_i, thread_id])
            * cell_volume
            * surface_correction_factors[node_i, thread_id]
        )

        val_x = f * xi_eta_x / y
        val_y = f * xi_eta_y / y

    shared_x[thread_id] = val_x
    shared_y[thread_id] = val_y

    cuda.syncthreads()

    # Reduction
    stride = THREADS_PER_BLOCK // 2
    while stride > 0:
        if thread_id < stride:
            shared_x[thread_id] += shared_x[thread_id + stride]
            shared_y[thread_id] += shared_y[thread_id + stride]
        cuda.syncthreads()
        stride //= 2

    if thread_id == 0:
        node_force[node_i, 0] = shared_x[0]
        node_force[node_i, 1] = shared_y[0]


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
