import numpy as np
from tqdm import trange

from solver.calculate import (calculate_particle_forces,
                              update_particle_positions,
                              calculate_contact_force,
                              smooth_step_data)

def run_simulation(bondlist, particle_coordinates, bond_stiffness, cell_volume,
                   damping, particle_density, dt, penetrator, support_1,
                   support_2, nlist):

    # Initialise arrays and variables
    n_nodes = np.shape(particle_coordinates)[0]
    n_bonds = np.shape(bondlist)[0]
    tmp = np.shape(nlist)[1]


    u = np.zeros([n_nodes, 3])
    ud = np.zeros([n_nodes, 3])
    udd = np.zeros([n_nodes, 3])
    particle_force = np.zeros([n_nodes, 3])
    particle_coordinates_deformed = np.zeros([n_nodes, 3])

    bond_damage = np.zeros([n_bonds, tmp])
    f_x = np.zeros([n_bonds, ])
    f_y = np.zeros([n_bonds, ])
    f_z = np.zeros([n_bonds, ])

    load = []
    cmod = []

    n_time_steps = 100000
    applied_displacement = -2e-4

    for i_time_step in trange(n_time_steps,
                              desc="Simulation Progress", unit="steps"):

        displacement_increment = smooth_step_data(i_time_step, 0, n_time_steps,
                                                  0, applied_displacement)

        # Calculate particle forces
        # (_particle_force,
        #  bond_damage) = calculate_particle_forces(bondlist,
        #                                           particle_coordinates,
        #                                           u,
        #                                           bond_damage,
        #                                           bond_stiffness,
        #                                           cell_volume,
        #                                           f_x, f_y, f_z,
        #                                           particle_force.copy())

        (_particle_force,
         bond_damage) = calculate_particle_forces(nlist,
                                                  particle_coordinates,
                                                  u,
                                                  bond_damage,
                                                  bond_stiffness,
                                                  cell_volume,
                                                  particle_force.copy())

        # Update particle positions
        (u, ud) = update_particle_positions(_particle_force,
                                            u, ud, udd,
                                            damping,
                                            particle_density,
                                            dt)

        particle_coordinates_deformed = particle_coordinates + u

        # Contact model
        (u,
         ud,
         _penetrator_f_z,
         particle_coordinates_deformed) = calculate_contact_force(penetrator, u, ud,
                                                                   displacement_increment,
                                                                   dt, particle_density,
                                                                   cell_volume,
                                                                   particle_coordinates_deformed,
                                                                   particle_coordinates)

        (u,
         ud,
         _support_1_f_z,
         particle_coordinates_deformed) = calculate_contact_force(support_1, u, ud,
                                                                   0,
                                                                   dt, particle_density,
                                                                   cell_volume,
                                                                   particle_coordinates_deformed,
                                                                   particle_coordinates)

        (u,
         ud,
         support_2_f_z,
         particle_coordinates_deformed) = calculate_contact_force(support_2, u, ud,
                                                                   0,
                                                                   dt, particle_density,
                                                                   cell_volume,
                                                                   particle_coordinates_deformed,
                                                                   particle_coordinates)

        if i_time_step % 100 == 0:
            load.append(_penetrator_f_z)
            cmod.append((u[24, 0] - u[9, 0]) * 1000)

    return load, cmod
