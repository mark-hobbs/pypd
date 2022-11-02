"""
Solver calculate functions
--------------------------

This module contains the core functions that are employed during a simulation.

"""


import numpy as np
from numba import njit, prange


@njit(parallel=True)
def calculate_nodal_forces(x, u, cell_volume, bondlist, d, c, f_x, f_y,
                           material_law):
    """
    Calculate particle forces - employs bondlist

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
    node_force : ndarray (float)
        Nodal force array

    d : ndarray (float)
        Bond damage (softening parameter). The value of d will range from 0
        to 1, where 0 indicates that the bond is still in the elastic range,
        and 1 represents a bond that has failed

    Notes
    -----
    * Can the constitutive model function be passed in as an argument?
        - See factory functions and closures
    * If the bondlist is loaded from a .mat file:
          node_i = bondlist[k_bond, 0] - 1
          node_j = bondlist[k_bond, 1] - 1

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

        d[k_bond] = material_law(stretch, d[k_bond])

        f = stretch * c * (1 - d[k_bond]) * cell_volume
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


# @njit(parallel=True)
# def calculate_nodal_forces(x, u, cell_volume, bondlist, d, c, f, material_law):
#     """
#     Calculate particle forces - employs bondlist

#     Parameters
#     ----------
#     bondlist : ndarray (int)
#         Array of pairwise interactions (bond list)

#     x : ndarray (float)
#         Material point coordinates in the reference configuration

#     u : ndarray (float)
#         Nodal displacement

#     d : ndarray (float)
#         Bond damage (softening parameter). The value of d will range from 0
#         to 1, where 0 indicates that the bond is still in the elastic range,
#         and 1 represents a bond that has failed

#     c : float
#         Bond stiffness

#     material_law : function

#     Returns
#     -------
#     node_force : ndarray (float)
#         Nodal force array

#     d : ndarray (float)
#         Bond damage (softening parameter). The value of d will range from 0
#         to 1, where 0 indicates that the bond is still in the elastic range,
#         and 1 represents a bond that has failed

#     Notes
#     -----
#     * TODO: why does this function not work? And why is it so slow?
#     """

#     n_nodes = np.shape(x)[0]
#     n_dimensions = np.shape(x)[1]
#     n_bonds = np.shape(bondlist)[0]

#     xi_component = np.zeros(n_dimensions, np.float64)
#     xi_eta_component = np.zeros(n_dimensions, np.float64)
#     node_force = np.zeros((n_nodes, n_dimensions))

#     for k_bond in prange(n_bonds):
#         node_i = bondlist[k_bond, 0]
#         node_j = bondlist[k_bond, 1]

#         for dof in range(n_dimensions):
#             xi_component[dof] = x[node_j, dof] - x[node_i, dof]
#             xi_eta_component[dof] = (xi_component[dof]
#                                      + (u[node_j, dof] - u[node_i, dof]))

#         xi = np.sqrt(np.sum(xi_component ** 2))
#         y = np.sqrt(np.sum(xi_eta_component ** 2))
#         stretch = (y - xi) / xi

#         d[k_bond] = material_law(stretch, d[k_bond])
#         f_scalar = stretch * c * (1 - d[k_bond]) * cell_volume

#         for dof in range(n_dimensions):
#             f[k_bond, dof] = f_scalar * (xi_eta_component[dof] / y)

#     # Reduce bond forces to particle forces
#     for k_bond in range(n_bonds):
#         node_i = bondlist[k_bond, 0]
#         node_j = bondlist[k_bond, 1]

#         for dof in range(n_dimensions):
#             node_force[node_i, dof] += f[k_bond, dof]
#             node_force[node_j, dof] -= f[k_bond, dof]

#     return node_force, d


@njit(parallel=True)
def euler_cromer(node_force, u, v, a, damping, node_density, dt,
                 bc_flag, bc_magnitude, bc_unit_vector):
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
            a[node_i, dof] = (node_force[node_i, dof] -
                              damping * v[node_i, dof]) / node_density
            v[node_i, dof] = v[node_i, dof] + (a[node_i, dof] * dt)
            u[node_i, dof] = u[node_i, dof] + (v[node_i, dof] * dt)

            if bc_flag[node_i, dof] != 0:
                u[node_i, dof] = bc_magnitude * bc_unit_vector[node_i, dof]

    return u, v


@njit
def smooth_step_data(current_time_step, start_time_step, final_time_step,
                     start_value, final_value):
    """
    Smooth 5th order polynomial
    """
    xi = ((current_time_step - start_time_step)
          / (final_time_step - start_time_step))
    alpha = (start_value + (final_value - start_value)
             * xi**3 * (10 - 15 * xi + 6 * xi**2))

    return alpha


@njit
def calculate_node_damage(x, bondlist, d, n_family_members):
    """
    Calculate the nodal damage

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
    node_damage = np.zeros((n_nodes, ))

    for k_bond in range(n_bonds):

        node_i = bondlist[k_bond, 0]
        node_j = bondlist[k_bond, 1]

        node_damage[node_i] += d[k_bond]
        node_damage[node_j] += d[k_bond]

    node_damage = node_damage / n_family_members

    return node_damage


@njit
def calculate_contact_force(penetrator_family, penetrator_radius,
                            penetrator_position, x, u, v,
                            density, cell_volume, dt):
    """
    Calculate contact force - calculate the contact force between a rigid
    penetrator and a deformable peridynamic body.

    Parameters
    ----------

    Returns
    -------
    u : ndarray
        Updated displacement array

    v : ndarray
        Updated velocity array

    contact_force : ndarray
        Resultant force components

    Notes
    -----
    - 'C style' code
    - Based on code from rigid_impactor.f90 in Chapter 10 - Peridynamic Theory
    & its Applications by Madenci & Oterkus
    """

    n_nodes = len(penetrator_family)
    n_dimensions = np.shape(x)[1]
    u_previous = u.copy()
    v_previous = v.copy()

    distance_component = np.zeros(n_dimensions, np.float64)
    contact_force = np.zeros(n_dimensions, np.float64)
    unit_vector = np.zeros(n_dimensions, np.float64)
    unit_vector_scaled = np.zeros(n_dimensions, np.float64)
    a = np.zeros(n_dimensions, np.float64)

    for i in range(n_nodes):
        node = penetrator_family[i]
        for j in range(n_dimensions):
            distance_component[j] = ((x[node, j] + u[node, j])
                                     - penetrator_position[j])

        distance = np.sqrt(np.sum(distance_component ** 2))

        if distance < penetrator_radius:
            for j in range(n_dimensions):
                unit_vector[j] = distance_component[j] / distance
                unit_vector_scaled[j] = unit_vector[j] * penetrator_radius

                u[node, j] = ((penetrator_position[j] + unit_vector_scaled[j])
                              - x[node, j])
                v[node, j] = (u[node, j] - u_previous[node, j]) / dt
                a[j] = (v[node, j] - v_previous[node, j]) / dt

                contact_force[j] = contact_force[j] + (density * cell_volume
                                                       * a[j])

    return contact_force


def calculate_contact_force_vectorised(penetrator_family, penetrator_radius,
                                       penetrator_position, x, u, v,
                                       density, cell_volume, dt):
    """
    Calculate contact force - calculate the contact force between a rigid
    penetrator and a deformable peridynamic body.

    Parameters
    ----------

    Returns
    -------
    u : ndarray
        Updated displacement array

    v : ndarray
        Updated velocity array

    contact_force : ndarray
        Resultant force components

    Notes
    -----
    - Vectorised code
    - Based on code from rigid_impactor.f90 in Chapter 10 - Peridynamic Theory &
    its Applications by Madenci & Oterkus
    """

    n_nodes = len(penetrator_family)
    n_dimensions = np.shape(x)[1]
    u_previous = u.copy()
    v_previous = v.copy()
    contact_force = np.zeros(n_dimensions, np.float64)

    for i in range(n_nodes):

        node = penetrator_family[i]

        distance_component = (x[node] + u[node]) - penetrator_position
        distance = np.sqrt(np.sum(distance_component ** 2))

        if distance < penetrator_radius:

            unit_vector = distance_component / distance
            unit_vector_scaled = unit_vector * penetrator_radius

            u[node] = (penetrator_position + unit_vector_scaled) - x[node]
            v[node] = (u[node] - u_previous[node]) / dt
            a = (v[node] - v_previous[node]) / dt

            contact_force = contact_force + (density * cell_volume * a)

    return u, v, contact_force
