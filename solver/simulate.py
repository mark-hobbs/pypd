import numpy as np

from solver.calculate import calculate_particle_forces, update_particle_positions, calculate_contact_force


def run_simulation(bondlist, particle_coordinates, bond_stiffness, cell_volume,
                   damping, particle_density, dt, penetrator):

    # Initialise arrays and variables
    n_nodes = np.shape(particle_coordinates)[0]
    n_bonds = np.shape(bondlist)[0]

    u = np.zeros([n_nodes, 3])
    ud = np.zeros([n_nodes, 3])
    udd = np.zeros([n_nodes, 3])

    bond_damage = np.zeros([n_bonds, ])
    f_x = np.zeros([n_bonds, ])
    f_y = np.zeros([n_bonds, ])
    f_z = np.zeros([n_bonds, ])

    
    for t in range(100):
        
        displacement_increment = 1

        # Calculate particle forces
        (particle_force,
         bond_damage) = calculate_particle_forces(bondlist,
                                                  particle_coordinates,
                                                  u,
                                                  bond_damage,
                                                  bond_stiffness,
                                                  cell_volume,
                                                  f_x, f_y, f_z)

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

        print("Time step:", t)

    return 0
