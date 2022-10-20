"""
Example: 2D Beam 4 half-notched [1]

[1] Grégoire, D., Rojas‐Solano, L. B., & Pijaudier‐Cabot, G. (2013). Failure
and size effect for notched and unnotched concrete beams. International Journal
for Numerical and Analytical Methods in Geomechanics, 37(10), 1434-1452.

------------------------------------------------

Run the following command from the root folder:

python -m examples.2D_B4_HN.py

"""

import numpy as np

from classes.boundary_conditions import BoundaryConditions
from classes.material import Material
from classes.particles import ParticleSet
from classes.bonds import BondSet
from classes.model import Model
from classes.simulation import Simulation
from classes.constitutive_law import Linear
from classes.integrator import EulerCromer
from classes.penetrator import Penetrator

mm_to_m = 1E-3


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


def build_boundary_conditions(particles, dx):

    bc_flag = np.zeros((len(particles), 2), dtype=np.intc)
    bc_unit_vector = np.zeros((len(particles), 2), dtype=np.intc)

    tol = 1e-6

    for i, particle in enumerate(particles):
        if particle[1] < (0.02 + tol):
            bc_flag[i, 1] = 1
            bc_unit_vector[i, 1] = -1
        if particle[1] > (0.23 - dx - tol):
            bc_flag[i, 1] = 1
            bc_unit_vector[i, 1] = 1

    return bc_flag, bc_unit_vector


def build_notch(x, bondlist, notch):

    n_nodes = np.shape(x)[0]
    n_bonds = np.shape(bondlist)[0]

    P1 = notch[0]
    P2 = notch[1]

    mask = []

    for k_bond in range(n_bonds):

        node_i = bondlist[k_bond, 0]
        node_j = bondlist[k_bond, 1]

        P3 = x[node_i]
        P4 = x[node_j]

        intersect = determine_intersection(P1, P2, P3, P4)

        if intersect == True:
            mask.append(k_bond)

    reduced_bondlist = np.delete(bondlist, mask, axis=0)
    n_family_members = rebuild_node_families(n_nodes, reduced_bondlist)

    return reduced_bondlist, n_family_members


def determine_intersection(P1, P2, P3, P4):
    """
    Determine if a bond intersects with a notch
        - Given two line segments, find if the 
          given line segments intersect with
          each other.

    Parameters
    ----------
    P : 
        P = (x, y)

    Returns
    ------
    Returns True if two lines intersect

    Notes
    -----
    * This solution is based on the following
      paper:

      Antonio, F. (1992). Faster line segment
      intersection. In Graphics Gems III
      (IBM Version) (pp. 199-202). Morgan
      Kaufmann.

    """

    A = P2 - P1
    B = P3 - P4
    C = P1 - P3

    denominator = (A[1] * B[0]) - (A[0] * B[1])

    alpha_numerator = (B[1] * C[0]) - (B[0] * C[1])
    beta_numerator = (A[0] * C[1]) - (A[1] * C[0])

    alpha = alpha_numerator / denominator
    beta = beta_numerator / denominator

    if (0 < alpha < 1) and (0 < beta < 1):
        intersect = True
    else:
        intersect = False

    return intersect


def rebuild_node_families(n_nodes, bondlist):

    n_bonds = np.shape(bondlist)[0]
    n_family_members = np.zeros(n_nodes)

    for k_bond in range(n_bonds):

        node_i = bondlist[k_bond, 0]
        node_j = bondlist[k_bond, 1]

        n_family_members[node_i] += 1
        n_family_members[node_j] += 1

    return n_family_members


def main():

    dx = 1.25 * mm_to_m
    n_div_x = np.rint((175 * mm_to_m) / dx).astype(int)
    n_div_y = np.rint((50 * mm_to_m) / dx).astype(int)
    notch = [np.array([0 - dx, 0.125 - (dx/2)]),
             np.array([0.2, 0.125 - (dx/2)])]  # TODO: update

    x = build_particle_coordinates(dx, n_div_x, n_div_y)
    flag, unit_vector = build_boundary_conditions(x, dx)  # TODO: not needed

    material = Material(name="quasi-brittle", E=37e9, Gf=143.2,
                        density=2346, ft=3.9E6)
    integrator = EulerCromer()
    bc = BoundaryConditions(flag, unit_vector, magnitude=1)
    particles = ParticleSet(x, dx, bc, material)
    linear = Linear(material, particles)
    bonds = BondSet(particles, linear)
    bonds.bondlist, particles.n_family_members = build_notch(particles.x,
                                                             bonds.bondlist,
                                                             notch)
    simulation = Simulation(dt=1e-8, n_time_steps=5000, damping=0)

    penetrators = []
    penetrators.append(Penetrator(np.array([0, 0]), 25 * mm_to_m, particles))
    penetrators.append(Penetrator(np.array([0, 0]), 25 * mm_to_m, particles))
    penetrators.append(Penetrator(np.array([0, 0]), 25 * mm_to_m, particles))

    model = Model(particles, bonds, simulation, integrator,
                  linear.calculate_bond_damage(linear.sc),
                  penetrators)

    model.run_simulation()


main()
