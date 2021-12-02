import numpy as np
import matplotlib.pyplot as plt

from input import utilities
from solver.penetrator import Penetrator
from solver.simulate import run_simulation


# --------------------------------------
#           Read input file
# --------------------------------------
mat = utilities.read_input_file("/Users/mark/Documents/PhD/2 Code/2.1 PhD Code/BB_PD/input/inputdatafiles/",
                                "Beam_4_UN_DX5mm.mat")


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
bondlist = mat['BONDLIST']
particle_coordinates = mat['undeformedCoordinates']
bond_stiffness = 2.32e+18
cell_volume = mat['DX']**3
damping = 1e5
particle_density = 2346
dt = 1.3e-6

# --------------------------------------
#              Simulate
# --------------------------------------

num_load, num_cmod = run_simulation(bondlist, particle_coordinates,
                                    bond_stiffness, cell_volume, damping,
                                    particle_density, dt, penetrator,
                                    support_1, support_2)


plt.plot(num_cmod, num_load, label='Numerical')
plt.savefig('load_CMOD', dpi=1000)
