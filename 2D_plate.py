
import numpy as np

from classes.boundary_conditions import BoundaryConditions
from classes.material import Material
from classes.particles import ParticleSet
from classes.bonds import BondSet
from classes.model import Model
from classes.simulation import Simulation


def build_particle_coordinates(dx, n_div_x, n_div_y):
    """
    Build particle coordinates
    
    Parameters
    ----------
    
    Returns
    -------

    Notes
    -----
    
    """
    particle_coordinates = np.zeros([n_div_x * n_div_y, 2])
    counter = 0

    for i_y in range(n_div_y):      # Depth
        for i_x in range(n_div_x):  # Length
            coord_x = dx * i_x
            coord_y = dx * i_y
            particle_coordinates[counter, 0] = coord_x
            particle_coordinates[counter, 1] = coord_y
            counter += 1

    return particle_coordinates


def build_boundary_conditions(particles):

    bc_flag = np.zeros((len(particles), 2), dtype=np.intc)
    bc_unit_vector = np.zeros((len(particles), 2), dtype=np.intc)

    for i, particle in enumerate(particles):
        if particle[0] < 50:
            bc_flag[i, 0] = 1
            bc_unit_vector[i, 0] = -1
        if particle[0] > 450:
            bc_flag[i, 0] = 1
            bc_unit_vector[i, 0] = 1

    return bc_flag, bc_unit_vector


def main():
    
    dx = 5E-3
    n_div_x = 100
    n_div_y = 50
    x = build_particle_coordinates(dx, n_div_x, n_div_y)
    flag, unit_vector = build_boundary_conditions(x)

    material = Material(name="quasi-brittle", E=33e9, Gf=130,
                        density=2400, ft=2.5)
    bc = BoundaryConditions(flag, unit_vector)
    particles = ParticleSet(x, dx, bc, material)
    cm = ConstitutiveModel()
    bonds = BondSet(particles.nlist, cm)
    simulation = Simulation()
    model = Model(particles, bonds, simulation)

    model.run_simulation()

main()