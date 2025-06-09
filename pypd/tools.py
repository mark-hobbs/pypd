import numpy as np
from numba import njit, cuda


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


def get_cuda_device_info(verbose=True):
    """
    Retrieve comprehensive information about the current CUDA device.

    Parameters:
    -----------
    verbose : bool, optional
        If True, print device information. If False, return as dictionary.

    Returns:
    --------
    dict or None
        Dictionary of device properties if verbose=False, otherwise None
    """
    try:
        device = cuda.get_current_device()
        context = cuda.current_context()

        device_info = {
            "name": device.name,
            "compute_capability": device.compute_capability,
            "total_memory_gb": context.get_memory_info().total / 1e9,
            "free_memory_gb": context.get_memory_info().free / 1e9,
            "multiprocessors": device.MULTIPROCESSOR_COUNT,
            "max_threads_per_block": device.MAX_THREADS_PER_BLOCK,
            "max_grid_dimensions": {
                "x": device.MAX_GRID_DIM_X,
                "y": device.MAX_GRID_DIM_Y,
                "z": device.MAX_GRID_DIM_Z,
            },
            "warp_size": device.WARP_SIZE,
            "clock_rate_khz": device.CLOCK_RATE,
            "memory_clock_rate_khz": device.MEMORY_CLOCK_RATE,
        }

        if verbose:
            print("CUDA Device Information:")
            print("-" * 40)
            print(f"{'Device Name:':<30} {device_info['name']}")
            print(f"{'Compute Capability:':<30} {device_info['compute_capability']}")

            print("\nMemory:")
            print(f"{'Total Memory:':<30} {device_info['total_memory_gb']:.2f} GB")
            print(f"{'Free Memory:':<30} {device_info['free_memory_gb']:.2f} GB")

            print("\nCompute Resources:")
            print(
                f"{'Streaming Multiprocessors:':<30} {device_info['multiprocessors']}"
            )
            print(
                f"{'Max Threads per Block:':<30} {device_info['max_threads_per_block']}"
            )

            print("\nGrid Limitations:")
            print(
                f"{'Max Grid Dimensions X:':<30} {device_info['max_grid_dimensions']['x']}"
            )
            print(
                f"{'Max Grid Dimensions Y:':<30} {device_info['max_grid_dimensions']['y']}"
            )
            print(
                f"{'Max Grid Dimensions Z:':<30} {device_info['max_grid_dimensions']['z']}"
            )

            print("\nAdditional Characteristics:")
            print(f"{'Warp Size:':<30} {device_info['warp_size']}")
            print(f"{'Clock Rate:':<30} {device_info['clock_rate_khz']/1e6:.2f} GHz")
            print(
                f"{'Memory Clock Rate:':<30} {device_info['memory_clock_rate_khz']/1e6:.2f} GHz"
            )

        return device_info if not verbose else None

    except Exception as e:
        print(f"Error retrieving CUDA device information: {e}")
        return None
