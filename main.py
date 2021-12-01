import numpy as np

from input import utilities
from solver.penetrator import Penetrator
from solver.simulate import run_simulation


# --------------------------------------
#           Read input file
# --------------------------------------
mat = utilities.read_input_file("/Users/mark/Documents/PhD/2 Code/2.1 PhD Code/BB_PD/input/inputdatafiles/",
                                "Beam_4_UN_DX5mm.mat")


penetrator = mat['penetrator']  # Structured ndarray
ID = penetrator['ID'].item()
centre = penetrator['centre'].item()
radius = penetrator['radius'].item()
search_radius = penetrator['searchRadius'].item()
family = penetrator['family'].item()

penetrator = Penetrator(ID, centre, radius, search_radius, family)
bondlist = mat['BONDLIST']
particle_coordinates = mat['undeformedCoordinates']
bond_stiffness = 4.345e+19
cell_volume = mat['DX']**3
damping = 1e5
particle_density = 2346
dt = 6.459e-07

# --------------------------------------
#              Simulate
# --------------------------------------

run_simulation(bondlist, particle_coordinates, bond_stiffness, cell_volume,
               damping, particle_density, dt, penetrator)