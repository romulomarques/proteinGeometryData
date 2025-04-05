import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


wd_dmdgp_HA9H_bsol = "dmdgp_HA9H_bsol"
wd_dmdgp_HA9H = "dmdgp_HA9H"

if not os.path.exists(wd_dmdgp_HA9H_bsol):
    os.makedirs(wd_dmdgp_HA9H_bsol)


def get_semispace_sign(point: np.ndarray, plane_points: np.ndarray) -> float:
    """
    Calculate the signed distance from a point to a plane defined by three points.
    Parameters:
        point (numpy array): The point [x, y, z] we want to check.
        plane_points (numpy array): 3x3 array where each row is a point [x, y, z] defining the plane.
    Returns:
        float: The signed distance from the point to the plane.
    """
    # Calculate the normal vector of the plane
    normal_vector = np.cross(
        plane_points[1] - plane_points[0], plane_points[2] - plane_points[0]
    )

    if np.linalg.norm(normal_vector) == 0:
        raise ValueError("The plane points are collinear")

    # Calculate the signed distance from the point to the plane
    u = point - plane_points[0]  # in case of duplicate points, u will be zero
    semispace_sign = np.dot(normal_vector, u)

    return semispace_sign


def get_bit(x: np.ndarray, i: int) -> int:
    if i < 3:
        return -1  # dummy value
    else:
        plane_points = x[i - 3 : i]
        semispace_sign = get_semispace_sign(x[i], plane_points)
        # duplicate atoms will have bit 0, because the semispace sign is 0
        return int(semispace_sign > 0)


def get_prune_bsol(bsol: list, row: pd.Series) -> tuple:
    i = int(row["i"])
    j = int(row["j"])
    str_b = "".join([str(bit) for bit in bsol[i + 3 : j + 1]])
    return str_b


def process_csv_sol(fn_sol):

    x = np.loadtxt(fn_sol)

    fn_timers = fn_sol.replace("_dfs.sol", "_dfs.tmr")
    df_timers = pd.read_csv(fn_timers)
    df_timers = df_timers[df_timers["code"] >= 0]

    bsol = list(map(lambda i: get_bit(x, i), np.arange(len(x))))
    df_timers["bsol"] = df_timers.apply(lambda row: get_prune_bsol(bsol, row), axis=1)

    fn_out = fn_timers.replace(wd_dmdgp_HA9H, wd_dmdgp_HA9H_bsol).replace(
        "_dfs.tmr", "_dfs.csv"
    )
    df_timers.to_csv(fn_out, index=False)


def process_csv_sol_test(fn_sol):
    process_csv_sol(fn_sol)
    exit()


if __name__ == "__main__":
    # Set the desired number of worker processes
    num_workers = os.cpu_count() - 1

    # select only csv files in wd_dmdgp_HA9H
    fn_sols = [fn for fn in os.listdir(wd_dmdgp_HA9H) if fn.endswith("_dfs.sol")]
    fn_sols = [os.path.join(wd_dmdgp_HA9H, fn) for fn in fn_sols]

    # process_csv_sol_test(fn_sols[0]) # call the function with a single file to test

    # Use ProcessPoolExecutor with the specified number of worker processes
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Use tqdm to show progress while processing files in parallel
        list(tqdm(executor.map(process_csv_sol, fn_sols), total=len(fn_sols)))
