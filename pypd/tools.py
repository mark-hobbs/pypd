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
