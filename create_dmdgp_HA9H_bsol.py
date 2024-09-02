import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


dmdgp_HA9H_sbbu = "dmdgp_HA9H_sbbu"
dmdgp_HA9H = "dmdgp_HA9H"


def read_dmdgp(fname: str) -> pd.DataFrame:
    df_dmdgp = pd.read_csv(fname)
    return df_dmdgp


def read_sol(fname: str) -> np.ndarray:
    x = np.loadtxt(fname)
    return x


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
    i = row["i"]
    j = row["j"]
    str_b = "".join([str(bit) for bit in bsol[i + 3 : j + 1]])
    return str_b


def process_file(fn_csv):
    fn_dmdgp = os.path.join(dmdgp_HA9H, fn_csv)

    fn_sol = os.path.join(dmdgp_HA9H_sbbu, fn_csv)
    fn_sol = fn_sol.replace(".csv", ".sol")
    x = read_sol(fn_sol)
    b = list(map(lambda i: get_bit(x, i), np.arange(len(x))))

    df_dmdgp = read_dmdgp(fn_dmdgp)
    df_dmdgp["bsol"] = df_dmdgp.apply(lambda row: get_prune_bsol(b, row), axis=1)

    fn_out = os.path.join(dmdgp_HA9H_sbbu, fn_csv)
    df_dmdgp.to_csv(fn_out, index=False)


if __name__ == "__main__":
    files = os.listdir(dmdgp_HA9H)

    # Set the desired number of worker processes
    num_workers = os.cpu_count() - 1

    # Use ProcessPoolExecutor with the specified number of worker processes
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Use tqdm to show progress while processing files in parallel
        list(tqdm(executor.map(process_file, files), total=len(files)))
