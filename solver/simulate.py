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
    _n_nodes = np.shape(particle_coordinates)[0]
    _n_bonds = np.shape(bondlist)[0]

    _u = np.zeros([_n_nodes, 3])
    _ud = np.zeros([_n_nodes, 3])
    _udd = np.zeros([_n_nodes, 3])
    _particle_force = np.zeros([_n_nodes, 3])
    _particle_coordinates_deformed = np.zeros([_n_nodes, 3])

    _bond_damage = np.zeros([_n_bonds, ])
    _f_x = np.zeros([_n_bonds, ])
    _f_y = np.zeros([_n_bonds, ])
    _f_z = np.zeros([_n_bonds, ])

    load = []
    cmod = []

    n_time_steps = 20000
    applied_displacement = -2e-4

    for i_time_step in trange(n_time_steps,
                              desc="Simulation Progress", unit="steps"):

        displacement_increment = smooth_step_data(i_time_step, 0, n_time_steps,
                                                  0, applied_displacement)

        # Calculate particle forces
        (_particle_force_,
         _bond_damage) = calculate_particle_forces(bondlist,
                                                  particle_coordinates,
                                                  _u,
                                                  _bond_damage,
                                                  bond_stiffness,
                                                  cell_volume,
                                                  _f_x, _f_y, _f_z,
                                                  _particle_force.copy())

        # print(np.max(particle_force))

        # Update particle positions
        (_u, _ud) = update_particle_positions(_particle_force_,
                                            _u, _ud, _udd.copy(),
                                            damping,
                                            particle_density,
                                            dt)

        _particle_coordinates_deformed = particle_coordinates + _u

        # Contact model
        (_u,
         _ud,
         _penetrator_f_z,
         _particle_coordinates_deformed) = calculate_contact_force(penetrator, _u, _ud,
                                                                   displacement_increment,
                                                                   dt, particle_density,
                                                                   cell_volume,
                                                                   _particle_coordinates_deformed,
                                                                   particle_coordinates)

        (_u,
         _ud,
         _support_1_f_z,
         _particle_coordinates_deformed) = calculate_contact_force(support_1, _u, _ud,
                                                                   0,
                                                                   dt, particle_density,
                                                                   cell_volume,
                                                                   _particle_coordinates_deformed,
                                                                   particle_coordinates)

        (_u,
         _ud,
         support_2_f_z,
         _particle_coordinates_deformed) = calculate_contact_force(support_2, _u, _ud,
                                                                   0,
                                                                   dt, particle_density,
                                                                   cell_volume,
                                                                   _particle_coordinates_deformed,
                                                                   particle_coordinates)

        if i_time_step % 100 == 0:
            load.append(_penetrator_f_z)
            cmod.append((_u[24, 0] - _u[9, 0]) * 1000)

    return load, cmod
