"""
Small, highly optimised computational units written using Numba
"""

import numpy as np
from numba import njit, prange, cuda


@njit(parallel=True)
def euler_cromer_cpu(
    f, u, v, a, density, bc_flag, bc_magnitude, bc_unit_vector, damping, dt
):
    """
    Update particle positions using an Euler-Cromer time integration scheme

    Parameters
    ----------
    f : np.ndarray(float, shape=(n_nodes, n_dimensions))
        Particle forces

    u : np.ndarray(float, shape=(n_nodes, n_dimensions))
        Particle displacement

    v : np.ndarray(float, shape=(n_nodes, n_dimensions))
        Particle velocity

    a : np.ndarray(float, shape=(n_nodes, n_dimensions))
        Particle acceleration

    density : float
        Material density

    bc_flag : np.ndarray(int, shape=(n_nodes, n_dimensions))
        0 - no boundary condition
        1 - the node is subject to a boundary condition

    bc_magnitude : float
        Magnitude of the applied force/displacement

    bc_unit_vector : np.ndarray(float, shape=(n_nodes, n_dimensions))
        Unit vector defining the direction of the boundary condition

    damping : float
        Damping coefficient

    dt : float
        Time step size

    Returns
    -------
    u : np.ndarray(float, shape=(n_nodes, n_dimensions))
        Updated particle displacement

    v : np.ndarray(float, shape=(n_nodes, n_dimensions))
        Updated particle velocity

    Notes
    -----
    - u and v are modified in place and returned for clarity
    """

    n_nodes = np.shape(f)[0]
    n_dimensions = np.shape(f)[1]

    for node_i in prange(n_nodes):
        for dof in range(n_dimensions):
            a[node_i, dof] = (f[node_i, dof] - damping * v[node_i, dof]) / density
            v[node_i, dof] = v[node_i, dof] + (a[node_i, dof] * dt)
            u[node_i, dof] = u[node_i, dof] + (v[node_i, dof] * dt)

            if bc_flag[node_i, dof] != 0:
                u[node_i, dof] = bc_magnitude * bc_unit_vector[node_i, dof]

    return u, v


def euler_cromer_gpu(
    f, u, v, a, density, bc_flag, bc_magnitude, bc_unit_vector, damping, dt
):
    euler_cromer_kernel[grid_size, block_size]()


@cuda.jit
def euler_cromer_kernel():
    """
    CUDA kernel for Euler-Cromer time integration scheme
    """
    pass
