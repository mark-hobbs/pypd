
import matplotlib.pyplot as plt
import numpy as np

from input.tools import build_particle_coordinates


def build_boundary_conditions(x, dx, n_div_x, n_div_y, n_div_z):
    """
    Build boundary conditions - plate

    Parameters
    ----------
    x : ndarray (float)
        Material point coordinates in the reference configuration

    Returns
    -------
    bc: ndarray (int)
        Array of flags. The flag is set to 1 if a node is constrained, the
        flag is set to 0 if a node is free

    Notes
    -----
    """
    bc = np.zeros([n_div_x * n_div_y * n_div_z, 3])

    for i in range(len(x)):
        if x[i, 1] < 1:
            bc[i, 1] = 1
        if x[i, 1] > (dx * (n_div_y - 1.5)):
            bc[i, 1] = 1

    return bc


def main():

    # ---------------
    # Build particles
    # ---------------

    dx = 1
    length = 100   # length (mm)
    depth = 25     # depth (mm)
    thickness = 5  # thickness (mm)

    n_div_x = int(length / dx)
    n_div_y = int(depth / dx)
    n_div_z = int(thickness / dx)

    # TODO: use numpy.meshgrid
    x = build_particle_coordinates(dx, n_div_x, n_div_y, n_div_z)

    # plt.scatter(x[:, 0], x[:, 1], s=10)
    # plt.axis('scaled')
    # plt.show()

    # -------------------
    # Boundary conditions
    # -------------------

    bc = build_boundary_conditions(x, dx, n_div_x, n_div_y, n_div_z)

    plt.scatter(x[:, 0], x[:, 1], s=10, c=bc[:, 1])
    plt.axis('scaled')
    plt.show()

    # -----------------------
    # Build particle families
    # -----------------------


if __name__ == "__main__":
    main()
