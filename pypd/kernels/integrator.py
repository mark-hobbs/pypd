"""
Small, highly optimised computational units written using Numba
"""


import numpy as np
from numba import njit, prange


@njit(parallel=True)
def euler_cromer(
    node_force,
    u,
    v,
    a,
    damping,
    node_density,
    dt,
    bc_flag,
    bc_magnitude,
    bc_unit_vector,
):
    """
    Update particle positions using an Euler-Cromer time integration scheme

    Parameters
    ----------
    u : ndarray (float)
        Particle displacement

    v : ndarray (float)
        Particle velocity

    a : ndarray (float)
        Particle acceleration

    Returns
    -------

    Notes
    -----
    * add random noise to particle displacement
        -  * np.random.uniform(0.98, 1.0)
    * We need a generic method for employing different time integration schemes
        - Euler
        - Euler-Cromer
        - Velocity-Verlet
    """

    n_nodes = np.shape(node_force)[0]
    n_dimensions = np.shape(node_force)[1]

    for node_i in prange(n_nodes):
        for dof in range(n_dimensions):
            a[node_i, dof] = (
                node_force[node_i, dof] - damping * v[node_i, dof]
            ) / node_density
            v[node_i, dof] = v[node_i, dof] + (a[node_i, dof] * dt)
            u[node_i, dof] = u[node_i, dof] + (v[node_i, dof] * dt)

            if bc_flag[node_i, dof] != 0:
                u[node_i, dof] = bc_magnitude * bc_unit_vector[node_i, dof]

    return u, v
