"""
Example: 2D plate with a notch (crack branching)
------------------------------------------------

See Section 5.2 Crack Branching in Homalite in [1]

[1] Bobaru, F., & Zhang, G. (2015). Why do cracks branch? A peridynamic 
investigation of dynamic brittle fracture. International Journal of Fracture,
196, 59-98.

Run
---
Run the following command from the root folder:

python -m examples.crack_branching

"""

import numpy as np

import pypd


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

    for i_y in range(n_div_y):  # Depth
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
        if particle[1] > (0.18 - dx - tol):
            bc_flag[i, 1] = 1
            bc_unit_vector[i, 1] = 1

    return bc_flag, bc_unit_vector

def main():
    dx = 1e-3
    n_div_x = np.rint(0.4 / dx).astype(int)
    n_div_y = np.rint(0.2 / dx).astype(int)
    notch = [np.array([0 - dx, 0.1 - (dx / 2)]), np.array([0.2, 0.1 - (dx / 2)])]

    x = build_particle_coordinates(dx, n_div_x, n_div_y)
    flag, unit_vector = build_boundary_conditions(x, dx)

    material = pypd.Material(name="homalite", E=4.55e9, Gf=38.46, density=1230, ft=2.5)
    bc = pypd.BoundaryConditions(flag, unit_vector, magnitude=1e-4)
    particles = pypd.Particles(x, dx, bc, material)
    bonds = pypd.Bonds(particles, influence=pypd.Constant, notch=notch)
    model = pypd.Model(particles, bonds)

    animation = pypd.Animation(
        frequency=100, sz=0.25, show_title=False, data="strain energy density"
    )
    simulation = pypd.Simulation(n_time_steps=5000, damping=0, animation=animation)
    simulation.run(model)
    model.save_final_state_fig(fig_title="crack-branching")


main()
