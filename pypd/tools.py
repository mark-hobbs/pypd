import numpy as np
from numba import njit


@njit
def smooth_step_data(
    current_time_step, start_time_step, final_time_step, start_value, final_value
):
    """
    Smooth 5th order polynomial
    """
    xi = (current_time_step - start_time_step) / (final_time_step - start_time_step)
    alpha = start_value + (final_value - start_value) * xi**3 * (
        10 - 15 * xi + 6 * xi**2
    )

    return alpha


def calculate_stable_time_step(rho, dx, horizon, c):
    """
    Calculate minimum stable time step

    Parameters
    ----------
    rho : float
        Material density (kg/m^3)

    dx : float
        Mesh resolution (only valid for regular meshes)

    horizon : float
        Horizon radius

    c : float
        Bond stiffness

    Returns
    -------
    dt : float
        Minimum stable time step

    Notes
    -----
    The time step is determined using the stability condition derived in [1]

    [1] Silling, S. A., & Askari, E. (2005). A meshfree method based on the
    peridynamic model of solid mechanics. Computers & structures, 83(17-18),
    1526-1535.

    TODO: is this equation correct and is it valid for 2D and 3D problems?
    """
    return np.sqrt((2 * rho * dx) / (np.pi * horizon**2 * dx * c))


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

    if (0 <= alpha <= 1) and (0 <= beta <= 1):
        intersect = True
    else:
        intersect = False

    return intersect


@njit
def rebuild_node_families(n_nodes, bondlist):
    n_bonds = np.shape(bondlist)[0]
    n_family_members = np.zeros(n_nodes)

    for k_bond in range(n_bonds):
        node_i = bondlist[k_bond, 0]
        node_j = bondlist[k_bond, 1]

        n_family_members[node_i] += 1
        n_family_members[node_j] += 1

    return n_family_members
