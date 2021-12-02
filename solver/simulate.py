import numpy as np
from tqdm import trange

from solver.calculate import (calculate_particle_forces,
                              update_particle_positions,
                              calculate_contact_force,
                              smooth_step_data)

def run_simulation(bondlist, particle_coordinates, bond_stiffness, cell_volume,
                   damping, particle_density, dt, penetrator, support_1,
                   support_2):

    # Initialise arrays and variables
    n_nodes = np.shape(particle_coordinates)[0]
    n_bonds = np.shape(bondlist)[0]

    u = np.zeros([n_nodes, 3])
    ud = np.zeros([n_nodes, 3])
    udd = np.zeros([n_nodes, 3])
    particle_force = np.zeros([n_nodes, 3])

    bond_damage = np.zeros([n_bonds, ])
    f_x = np.zeros([n_bonds, ])
    f_y = np.zeros([n_bonds, ])
    f_z = np.zeros([n_bonds, ])

    n_time_steps = 10000
    applied_displacement = -2e-4

    for i_time_step in trange(n_time_steps,
                              desc="Simulation Progress", unit="steps"):
        
        displacement_increment = smooth_step_data(i_time_step, 0, n_time_steps,
                                                  0, applied_displacement)

        # Calculate particle forces
        (particle_force,
         bond_damage) = calculate_particle_forces(bondlist,
                                                  particle_coordinates,
                                                  u,
                                                  bond_damage,
                                                  bond_stiffness,
                                                  cell_volume,
                                                  f_x, f_y, f_z,
                                                  particle_force)

        # Update particle positions
        (u,
         ud) = update_particle_positions(particle_force,
                                         u, ud, udd,
                                         damping,
                                         particle_density,
                                         dt)

        # Contact model
        (u,
         ud,
         penetrator_f_z) = calculate_contact_force(penetrator, u, ud,
                                                   displacement_increment,
                                                   dt, particle_density,
                                                   cell_volume)

        (u,
         ud,
         support_1_f_z) = calculate_contact_force(support_1, u, ud,
                                                  0,
                                                  dt, particle_density,
                                                  cell_volume)

        (u,
         ud,
         support_2_f_z) = calculate_contact_force(support_2, u, ud,
                                                  0,
                                                  dt, particle_density,
                                                  cell_volume)

        print(penetrator_f_z)

    return 0
