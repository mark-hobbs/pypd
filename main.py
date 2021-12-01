import numpy as np

from input import utilities


# --------------------------------------
#           Read input file
# --------------------------------------
mat = utilities.read_input_file("/Users/mark/Documents/PhD/2 Code/2.1 PhD Code/BB_PD/input/inputdatafiles/",
                                "Beam_4_UN_DX2pt5mm.mat")

particle_coordinates = mat['undeformedCoordinates']
print(np.shape(particle_coordinates))



# --------------------------------------
#              Simulate
# --------------------------------------