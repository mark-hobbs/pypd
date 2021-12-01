
import numpy as np

from solver.constitutive_model import trilinear_constitutive_model


def calculate_particle_forces(bondlist, particle_coordinates, u, bond_damage,
                              bond_stiffness, cell_volume, f_x, f_y, f_z):
    """
    Calculate particle forces
    """
    n_bonds = np.shape(bondlist)[0]
    n_nodes = np.shape(particle_coordinates)[0]
    particle_force = np.zeros([n_nodes, 3])

    for k_bond in range(n_bonds):
        
        node_i = bondlist[k_bond, 0] - 1
        node_j = bondlist[k_bond, 1] - 1
        
        xi_x = particle_coordinates[node_j, 0] - particle_coordinates[node_i, 0]
        xi_y = particle_coordinates[node_j, 1] - particle_coordinates[node_i, 1]
        xi_z = particle_coordinates[node_j, 2] - particle_coordinates[node_i, 2]
        
        xi_eta_x = xi_x + (u[node_j, 0] - u[node_i, 0])
        xi_eta_y = xi_y + (u[node_j, 1] - u[node_i, 1])
        xi_eta_z = xi_z + (u[node_j, 2] - u[node_i, 2])
        
        xi = np.sqrt(xi_x**2 + xi_y**2 + xi_z**2)
        y = np.sqrt(xi_eta_x**2 + xi_eta_y**2 + xi_eta_z**2)
        stretch = (y - xi) / xi
        
        s0 = 1.05e-4
        s1 = 6.90e-4
        sc = 5.56e-3
        beta = 0.25
        bond_damage[k_bond] = trilinear_constitutive_model(stretch, s0, s1, sc,
                                                           bond_damage[k_bond],
                                                           beta)

        f = stretch * bond_stiffness * (1 - bond_damage[k_bond]) * cell_volume
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

    return particle_force, bond_damage


def update_particle_positions(particle_force, u, ud, udd, damping,
                              particle_density, dt):
    """
    Update particle positions using an Euler-Cromer time integration scheme
    """

    n_nodes = np.shape(particle_force)[0]

    for node_i in range(n_nodes):
        for dof in range(3):
            udd[node_i, dof] = (particle_force[node_i, dof]
                                - damping * ud[node_i, dof]) / particle_density
            ud[node_i, dof] = ud[node_i, dof] + (udd[node_i, dof] * dt)
            u[node_i, dof] = u[node_i, dof] + (ud[node_i, dof] * dt)

    return u, ud


def calculate_contact_force(penetrator, u, ud, displacement_increment,
                            dt, particle_density, cell_volume):
    """
    Calculate contact force between rigid penetrator/support and deformable
    body
    """

    # TODO: is u the displacement or the coordinates of the displaced
    # particles?

    counter = 0
    penetrator_f_x = 0
    penetrator_f_y = 0
    penetrator_f_z = 0
    u_previous = u
    ud_previous = ud

    # Move penetrator vertically (z-axis)
    penetrator.centre = penetrator.centre + displacement_increment

    # Calculate distance between penetrator centre and nodes in penetrator 
    # family

    for i in range(len(penetrator.family)):

        node_i = penetrator.family[i]

        distance_x = u[node_i, 0] - penetrator.centre[0]
        distance_z = u[node_i, 2] - penetrator.centre[1]
        distance = np.sqrt(distance_x**2 + distance_z**2)

        if distance < penetrator.radius:

            counter += 1

            # Calculate unit vector
            unit_x = distance_x / distance
            unit_z = distance_z / distance

            # Scale unit vector by penetrator radius
            unit_x_scaled = unit_x * penetrator.radius
            unit_z_scaled = unit_z * penetrator.radius

            # Calculate new particle positions
            u[node_i, 0] = penetrator.centre[0] + unit_x_scaled
            u[node_i, 2] = penetrator.centre[1] + unit_z_scaled

            # Calculate particle velocity
            ud[node_i, 0] = (u[node_i, 0] - u_previous[node_i, 0]) / dt
            ud[node_i, 2] = (u[node_i, 2] - u_previous[node_i, 2]) / dt

            # Calculate the reaction force from a particle on the penetrator
            penetrator_f_x += (-1 * particle_density
                               * (ud[node_i, 0] - ud_previous[node_i, 0])
                               / dt * cell_volume)
            penetrator_f_y += (-1 * particle_density
                               * (ud[node_i, 1] - ud_previous[node_i, 1])
                               / dt * cell_volume)
            penetrator_f_z += (-1 * particle_density
                               * (ud[node_i, 2] - ud_previous[node_i, 2])
                               / dt * cell_volume)

    return u, ud, penetrator_f_z
