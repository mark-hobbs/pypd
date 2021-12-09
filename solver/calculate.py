
import numpy as np
from numba import njit, prange

from solver.constitutive_model import trilinear_constitutive_model


@njit(parallel=True)
def calculate_particle_forces(bondlist, x, u, d, c, cell_volume,
                              s0, s1, sc, beta, f_x, f_y, f_z,
                              particle_force):
    """
    Calculate particle forces
    """
    n_bonds = np.shape(bondlist)[0]

    for k_bond in prange(n_bonds):

        node_i = bondlist[k_bond, 0] - 1
        node_j = bondlist[k_bond, 1] - 1

        xi_x = x[node_j, 0] - x[node_i, 0]
        xi_y = x[node_j, 1] - x[node_i, 1]
        xi_z = x[node_j, 2] - x[node_i, 2]

        xi_eta_x = xi_x + (u[node_j, 0] - u[node_i, 0])
        xi_eta_y = xi_y + (u[node_j, 1] - u[node_i, 1])
        xi_eta_z = xi_z + (u[node_j, 2] - u[node_i, 2])

        xi = np.sqrt(xi_x**2 + xi_y**2 + xi_z**2)
        y = np.sqrt(xi_eta_x**2 + xi_eta_y**2 + xi_eta_z**2)
        stretch = (y - xi) / xi

        d[k_bond] = trilinear_constitutive_model(stretch, s0, s1, sc,
                                                 d[k_bond], beta)

        f = stretch * c * (1 - d[k_bond]) * cell_volume
        f_x[k_bond] = f * xi_eta_x / y
        f_y[k_bond] = f * xi_eta_y / y
        f_z[k_bond] = f * xi_eta_z / y

    # Reduce bond forces to particle forces
    for k_bond in range(n_bonds):

        node_i = bondlist[k_bond, 0] - 1
        node_j = bondlist[k_bond, 1] - 1

        particle_force[node_i, 0] += f_x[k_bond]
        particle_force[node_j, 0] -= f_x[k_bond]
        particle_force[node_i, 1] += f_y[k_bond]
        particle_force[node_j, 1] -= f_y[k_bond]
        particle_force[node_i, 2] += f_z[k_bond]
        particle_force[node_j, 2] -= f_z[k_bond]

    return particle_force, d


# @njit(parallel=True)
# def calculate_particle_forces(nlist, particle_coordinates, u, bond_damage,
#                               bond_stiffness, cell_volume, f_x, f_y, f_z,
#                               particle_force):
#     """
#     Calculate particle forces
#     """
#     n_nodes = np.shape(particle_coordinates)[0]
#     max_n_family_members = np.shape(nlist)[1]

#     for node_i in prange(n_nodes):
#         for j in range(max_n_family_members):

#             node_j = nlist[node_i, j]

#             if (node_j != -1) and (node_i < node_j):

#                 xi_x = (particle_coordinates[node_j, 0]
#                         - particle_coordinates[node_i, 0])
#                 xi_y = (particle_coordinates[node_j, 1]
#                         - particle_coordinates[node_i, 1])
#                 xi_z = (particle_coordinates[node_j, 2]
#                         - particle_coordinates[node_i, 2])

#                 xi_eta_x = xi_x + (u[node_j, 0] - u[node_i, 0])
#                 xi_eta_y = xi_y + (u[node_j, 1] - u[node_i, 1])
#                 xi_eta_z = xi_z + (u[node_j, 2] - u[node_i, 2])

#                 xi = np.sqrt(xi_x**2 + xi_y**2 + xi_z**2)
#                 y = np.sqrt(xi_eta_x**2 + xi_eta_y**2 + xi_eta_z**2)
#                 stretch = (y - xi) / xi

#                 s0 = 1.05e-4
#                 s1 = 6.90e-4
#                 sc = 5.56e-3
#                 beta = 0.25
#                 bond_damage[node_i, j] = trilinear_constitutive_model(stretch, s0, s1, sc,
#                                                                     bond_damage[node_i, j],
#                                                                     beta)

#                 f = (stretch * bond_stiffness * (1 - bond_damage[node_i, j])
#                      * cell_volume)
#                 f_x[node_i, j] = f * xi_eta_x / y
#                 f_y[node_i, j] = f * xi_eta_y / y
#                 f_z[node_i, j] = f * xi_eta_z / y

#     for node_i in range(n_nodes):
#         for j in range(max_n_family_members):

#             particle_force[node_i, 0] += f_x[node_i, j]
#             particle_force[node_i, 0] -= f_x[node_i, j]
#             particle_force[node_i, 1] += f_y[node_i, j]
#             particle_force[node_i, 1] -= f_y[node_i, j]
#             particle_force[node_i, 2] += f_z[node_i, j]
#             particle_force[node_i, 2] -= f_z[node_i, j]

#     return particle_force, bond_damage

@njit(parallel=True)
def update_particle_positions(particle_force, u, ud, udd, damping,
                              particle_density, dt):
    """
    Update particle positions using an Euler-Cromer time integration scheme
    """

    n_nodes = np.shape(particle_force)[0]

    for node_i in prange(n_nodes):
        for dof in range(3):
            udd[node_i, dof] = (particle_force[node_i, dof]
                                - damping * ud[node_i, dof]) / particle_density
            ud[node_i, dof] = ud[node_i, dof] + (udd[node_i, dof] * dt)
            u[node_i, dof] = u[node_i, dof] + (ud[node_i, dof] * dt)

    return u, ud


@njit
def calculate_contact_force(pen, u, ud, displacement_increment,
                            dt, particle_density, cell_volume,
                            x_deformed, x):
    """
    Calculate contact force between rigid penetrator/support and deformable
    body
    """

    # TODO: is u the displacement or the coordinates of the displaced
    # particles?

    penetrator_f_x = 0
    penetrator_f_y = 0
    penetrator_f_z = 0
    u_previous = u.copy()
    ud_previous = ud.copy()

    # Move penetrator vertically (z-axis)
    pen_displacement = pen.centre[1] + displacement_increment

    # Calculate distance between penetrator centre and nodes in penetrator
    # family

    for i in range(len(pen.family)):

        node_i = pen.family[i]

        distance_x = x_deformed[node_i, 0] - pen.centre[0]
        distance_z = x_deformed[node_i, 2] - pen_displacement
        distance = np.sqrt(distance_x**2 + distance_z**2)

        if distance < pen.radius:

            # Calculate unit vector
            unit_x = distance_x / distance
            unit_z = distance_z / distance

            # Scale unit vector by penetrator radius
            unit_x_scaled = unit_x * pen.radius
            unit_z_scaled = unit_z * pen.radius

            # Calculate new particle positions
            x_deformed[node_i, 0] = pen.centre[0] + unit_x_scaled
            x_deformed[node_i, 2] = pen_displacement + unit_z_scaled

            u[node_i, 0] = x_deformed[node_i, 0] - x[node_i, 0]
            u[node_i, 2] = x_deformed[node_i, 2] - x[node_i, 2]

            # Calculate particle velocity
            ud[node_i, 0] = (u[node_i, 0] - u_previous[node_i, 0]) / dt
            ud[node_i, 2] = (u[node_i, 2] - u_previous[node_i, 2]) / dt

            # Calculate the reaction force from a particle on the penetrator
            # F = ma
            penetrator_f_x += (particle_density * cell_volume
                               * (ud[node_i, 0] - ud_previous[node_i, 0]) / dt)
            penetrator_f_y += (particle_density * cell_volume
                               * (ud[node_i, 1] - ud_previous[node_i, 1]) / dt)
            penetrator_f_z += (particle_density * cell_volume
                               * (ud[node_i, 2] - ud_previous[node_i, 2]) / dt)

    return u, ud, penetrator_f_z, x_deformed


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
