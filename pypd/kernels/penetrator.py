"""
Small, highly optimised computational units written using Numba
"""

import numpy as np
from numba import njit


@njit
def compute_contact_force(
    penetrator_family,
    penetrator_radius,
    penetrator_position,
    x,
    u,
    f,
    cell_volume,
    k,
):
    """
    Compute Short-Range Contact Force (Penalty Method)

    Parameters
    ----------
    k : float
        The penalty stiffness. Usually set high (e.g., 10-100x the
        material bulk modulus).
    """
    n_nodes = len(penetrator_family)
    n_dimensions = x.shape[1]

    contact_force = np.zeros(n_dimensions, np.float64)

    for i in range(n_nodes):
        node = penetrator_family[i]

        # Calculate the relative distance vector between the centre of the penetrator and the node
        dist_sq = 0.0
        diff = np.zeros(n_dimensions)
        for j in range(n_dimensions):
            diff[j] = (x[node, j] + u[node, j]) - penetrator_position[j]
            dist_sq += diff[j] ** 2

        distance = np.sqrt(dist_sq)

        if distance < penetrator_radius:
            overlap = penetrator_radius - distance
            for j in range(n_dimensions):
                unit_vector = diff[j] / distance
                force = k * overlap * unit_vector
                f[node, j] += force / cell_volume
                contact_force[j] -= force

    return contact_force


@njit
def compute_contact_force_legacy(
    penetrator_family,
    penetrator_radius,
    penetrator_position,
    x,
    u,
    v,
    density,
    cell_volume,
    dt,
):
    """
    Compute contact force - calculate the contact force between a rigid
    penetrator and a deformable peridynamic body (Kinematic Constraint Method)

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
    - The penalty method replaces the kinematic constraint method. This 
    function is preserved for reference.
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
            distance_component[j] = (x[node, j] + u[node, j]) - penetrator_position[j]

        distance = np.sqrt(np.sum(distance_component**2))

        if distance < penetrator_radius:
            for j in range(n_dimensions):
                unit_vector[j] = distance_component[j] / distance
                unit_vector_scaled[j] = unit_vector[j] * penetrator_radius

                u[node, j] = (penetrator_position[j] + unit_vector_scaled[j]) - x[
                    node, j
                ]
                v[node, j] = (u[node, j] - u_previous[node, j]) / dt
                a[j] = (v[node, j] - v_previous[node, j]) / dt

                contact_force[j] = contact_force[j] + (density * cell_volume * a[j])

    return contact_force


def compute_contact_force_vectorised_legacy(
    penetrator_family,
    penetrator_radius,
    penetrator_position,
    x,
    u,
    v,
    density,
    cell_volume,
    dt,
):
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
    - The penalty method replaces the kinematic constraint method. This 
    function is preserved for reference.
    """

    n_nodes = len(penetrator_family)
    n_dimensions = np.shape(x)[1]
    u_previous = u.copy()
    v_previous = v.copy()
    contact_force = np.zeros(n_dimensions, np.float64)

    for i in range(n_nodes):
        node = penetrator_family[i]

        distance_component = (x[node] + u[node]) - penetrator_position
        distance = np.sqrt(np.sum(distance_component**2))

        if distance < penetrator_radius:
            unit_vector = distance_component / distance
            unit_vector_scaled = unit_vector * penetrator_radius

            u[node] = (penetrator_position + unit_vector_scaled) - x[node]
            v[node] = (u[node] - u_previous[node]) / dt
            a = (v[node] - v_previous[node]) / dt

            contact_force = contact_force + (density * cell_volume * a)

    return u, v, contact_force
