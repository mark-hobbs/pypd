# --------------------------------------
#        Three-point bending test
# --------------------------------------

import pathlib

import h5py
import numpy as np
import matplotlib.pyplot as plt

from input import utilities, tools
# from input.application import InputFile
from input.material import Material
from solver.penetrator import Penetrator
from solver.simulate import run_simulation


def main():

    # --------------------------------------
    #           Read input file
    # --------------------------------------
    mat = utilities.read_input_file("/Users/mark/Documents/PhD/2 Code/2.1 PhD Code/BB_PD/input/inputdatafiles/",
                                    "Beam_4_UN_DX5mm.mat")

    test = utilities.read_mat_file("/Users/mark/Documents/PhD/2 Code/2.1 PhD Code/BB_PD/input/inputdatafiles/",
                                   "Beam_4_UN_DX5mm.mat")

    print(test.coordinates)

    return

    dx = mat['DX']
    particle_coordinates = mat['undeformedCoordinates']
    bondlist = mat['BONDLIST']

    # TODO: store all instances of Penetrator in a list

    # Penetrator
    penetrator = mat['penetrator']  # Structured ndarray
    ID = penetrator['ID'].item()
    centre = penetrator['centre'].item()
    radius = penetrator['radius'].item()
    search_radius = penetrator['searchRadius'].item()
    family = penetrator['family'].item()
    penetrator = Penetrator(ID, centre, radius, search_radius, family)

    # Supports
    supports = mat['supports']  # Structured ndarray

    ID = supports['ID'][0]
    centre = supports['centre'][0]
    radius = supports['radius'][0]
    search_radius = supports['searchRadius'][0]
    family = supports['family'][0]
    support_1 = Penetrator(ID, centre, radius, search_radius, family)

    ID = supports['ID'][1]
    centre = supports['centre'][1]
    radius = supports['radius'][1]
    search_radius = supports['searchRadius'][1]
    family = supports['family'][1]
    support_2 = Penetrator(ID, centre, radius, search_radius, family)

    # Parameters
    horizon = np.pi * dx
    cell_volume = dx**3
    damping = 1e5
    dt = 1.3e-6

    # Material properties / constitutive law
    concrete = Material(youngs_modulus=37e9,
                        fracture_energy=143.2,
                        density=2346,
                        poissons_ratio=0.2,
                        tensile_strength=3.9e6)
    bond_stiffness = tools.calculate_bond_stiffness(concrete.youngs_modulus,
                                                    horizon)
    s0 = concrete.tensile_strength / concrete.youngs_modulus
    beta = 0.25
    gamma = (3 + (2 * beta)) / ((2 * beta) * (1 - beta))
    sc_numerator = (10 * gamma * concrete.fracture_energy)
    sc_denominator = (np.pi * horizon**5 * bond_stiffness * s0
                      * (1 + (gamma * beta)))
    sc = (sc_numerator / sc_denominator)  + s0
    s1 = s0 + (sc - s0) / gamma

    nlist = tools.build_particle_families(particle_coordinates,
                                          horizon)

    # --------------------------------------
    #              Simulate
    # --------------------------------------

    num_load, num_cmod = run_simulation(bondlist, particle_coordinates,
                                        bond_stiffness, cell_volume, damping,
                                        concrete.density, dt, penetrator,
                                        support_1, support_2, s0, s1, sc, beta,
                                        nlist)

    # --------------------------------------
    #           Post-processing
    # --------------------------------------

    # Plot the experimental data
    exp_data_path = (pathlib.Path(__file__).parent.resolve()
                     / "experimental_data.h5")
    exp_data = h5py.File(exp_data_path, 'r')
    exp_load_CMOD = exp_data['load_CMOD']
    exp_CMOD = exp_load_CMOD[0, 0:20000]
    exp_load_mean = exp_load_CMOD[1, 0:20000] * 1000
    exp_load_min = exp_load_CMOD[2, 0:20000] * 1000
    exp_load_max = exp_load_CMOD[3, 0:20000] * 1000
    plt.plot(exp_CMOD, exp_load_mean, color=(0.8, 0.8, 0.8),
             label='Experimental')
    plt.fill_between(exp_CMOD, exp_load_min, exp_load_max,
                     color=(0.8, 0.8, 0.8))

    # Plot the verification data
    ver_data_path = (pathlib.Path(__file__).parent.resolve()
                     / "verification_data.h5")
    ver_data = h5py.File(ver_data_path, 'r')
    ver_load_CMOD = ver_data['load_CMOD']
    ver_load = ver_load_CMOD[0, 0:499]
    ver_CMOD = ver_load_CMOD[1, 0:499]
    plt.plot(ver_CMOD, ver_load, 'tab:orange', label='Verification data')

    plt.plot(num_cmod, num_load, label='Numerical')
    plt.savefig('load_CMOD', dpi=1000)


main()
