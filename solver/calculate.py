
import numpy as np

from constitutive_model import trilinear_constitutive_model


def calculate_particle_forces():
    """
    Calculate particle forces
    """

    for k_bond in range(n_bonds):
        
        node_i = bondlist[k_bond, 0]
        node_j = bondlist[k_bond, 1]
        
        xi_x = particle_coordinates[node_j, 0] - particle_coordinates[node_i, 0]
        xi_y = particle_coordinates[node_j, 1] - particle_coordinates[node_i, 1]
        xi_z = particle_coordinates[node_j, 2] - particle_coordinates[node_i, 2]
        
        xi_eta_x = xi_x + (u[node_j, 0] - u[node_i, 0])
        xi_eta_y = xi_y + (u[node_j, 1] - u[node_i, 1])
        xi_eta_z = xi_z + (u[node_j, 2] - u[node_i, 2])
        
        xi = np.sqrt(xi_x**2 + xi_y**2 + xi_z**2)
        y = np.sqrt(xi_eta_x**2 + xi_eta_y**2 + xi_eta_z**2)
        stretch = (y - xi) / xi
        
        bond_damage[k_bond] = trilinear_constitutive_model(stretch, s0, s1, sc,
                                                           bond_damage[k_bond],
                                                           beta)

        f = stretch * bond_stiffness * (1 - bond_damage) * cell_volume
        f_x[k_bond] = f * xi_eta_x / y
        f_y[k_bond] = f * xi_eta_y / y
        f_z[k_bond] = f * xi_eta_z / y

    # Reduce bond forces to particle forces
    for k_bond in range(n_bonds):

        node_i = bondlist[k_bond, 0]
        node_j = bondlist[k_bond, 1]

        particle_force[node_i, 0] += f_x[k_bond]
        particle_force[node_j, 0] -= f_x[k_bond]
        particle_force[node_i, 1] += f_y[k_bond]
        particle_force[node_j, 1] -= f_y[k_bond]
        particle_force[node_i, 2] += f_z[k_bond]
        particle_force[node_j, 2] -= f_z[k_bond]

    return particle_force, bond_damage


def update_particle_positions():
    """
    Update particle positions using an Euler-Cromer time integration scheme
    """
    
    for node_i in range(n_nodes):
        for dof in range(n_degrees_freedom):
            udd[node_i, dof] = (particle_force[node_i, dof]
                                - damping * ud[node_i, dof]) / particle_density
            ud[node_i, dof] = ud[node_i, dof] + (udd[node_i, dof] * dt)
            u[node_i, dof] = u[node_i, dof] + (ud[node_i, dof] * dt)

    return u, ud


def calculate_contact_force():
    """
    Calculate contact force between rigid penetrator/support and deformable
    body
    """

    # TODO: is u the displacement or the coordinates of the displaced
    # particles?

    counter = 0

    # Move penetrator vertically (z-axis)
    penetrator_centre = penetrator_displacement + displacement_increment

    # Calculate distance between penetrator centre and nodes in penetrator 
    # family

    for i in range(family_size):

        node_i = penetrator_family[i]

        distance_x = u[node_i, 0] - penetrator_centre_x
        distance_z = u[node_i, 2] - penetrator_centre_z
        distance = np.sqrt(distance_x**2 + distance_z**2)

        if distance < penetrator_radius:

            counter += 1

            # Calculate unit vector
            unit_x = distance_x / distance
            unit_z = distance_z / distance

            # Scale unit vector by penetrator radius
            unit_x_scaled = unit_x * penetrator_radius
            unit_z_scaled = unit_z * penetrator_radius

            # Calculate new particle positions
            u[node_i, 0] = penetrator_centre_x + unit_x_scaled
            u[node_i, 2] = penetrator_centre_z + unit_z_scaled

            # Calculate particle velocity
            udd[node_i, 0] = (u[node_i, 0] - u_previous[node_i, 0]) / dt
            udd[node_i, 2] = (u[node_i, 2] - u_previous[node_i, 2]) / dt 

            # Calculate the reaction force from a particle on the penetrator
            penetrator_f_x += (-1 * particle_density
                               * (udd[node_i, 0] - udd_previous[node_i, 0])
                               / dt * cell_volume)
            penetrator_f_y += (-1 * particle_density
                               * (udd[node_i, 1] - udd_previous[node_i, 1])
                               / dt * cell_volume)
            penetrator_f_z += (-1 * particle_density
                               * (udd[node_i, 2] - udd_previous[node_i, 2])
                               / dt * cell_volume)
