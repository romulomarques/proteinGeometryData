import os
import time
import numpy as np
from numpy.linalg import norm
from typing import List, Tuple, Dict
from tqdm import tqdm

DATA = {
    "ddgp": {
        "n": 10,
        "seed": 42,
        "p": 0.5,
        "D": dict(),
    },
    "bp": {
        "x": list(),  # list of np.ndarray
        "b": list(),  # list of list of bool
    },
}


def solveEQ3(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    da: float,
    db: float,
    dc: float,
    stol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve a system of three equations to find a point in 3D space.

    Args:
    a, b, c (np.ndarray): 3D coordinates of three known points
    da, db, dc (float): Distances from the unknown point to a, b, and c respectively
    stol (float): Tolerance for numerical stability

    Returns:
    tuple: (proj_x, w)
        proj_x (np.ndarray): Projection of the solution on the plane formed by a, b, and c
        w (np.ndarray): Vector perpendicular to the plane
    """
    u = b - a
    A11 = np.linalg.norm(u)
    v = c - a
    A22 = np.linalg.norm(v)

    u = u / A11
    v = v / A22

    w = np.cross(u, v)
    w = w / np.linalg.norm(w)

    uv = np.inner(u, v)

    A12 = A11 * uv
    A21 = A22 * uv
    A = [[A11, A12], [A21, A22]]

    B = [(da**2 - db**2 + A11**2) / 2.0, (da**2 - dc**2 + A22**2) / 2.0]

    y0, y1 = np.linalg.solve(A, B)

    s = da**2 - y0**2 - y1**2 - 2.0 * y0 * y1 * uv

    if s < 0 and np.abs(s) > stol:
        w = np.array([np.inf, 0, 0])
        proj_x = np.zeros(3)
        return proj_x, w

    if s < 0:
        s = 0

    proj_x = a + y0 * u + y1 * v

    y2 = np.sqrt(s)
    w = y2 * w

    return proj_x, w


def calc_pw(
    i: int, x: np.ndarray, D: Dict[int, Dict[int, float]], debug: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the position and perpendicular vector for point i given its three parent points.

    Args:
    i (int): Index of the point to calculate
    x (np.ndarray): Array of current point positions
    D (Dict[int, Dict[int, float]]): Dictionary containing distance and parent information
    debug (bool): Flag for additional assertion checks

    Returns:
    tuple: (p, w)
        p (np.ndarray): Projection of the point on the plane formed by its parents
        w (np.ndarray): Vector perpendicular to the plane
    """
    # get the first three items in D[i]
    ia, ib, ic = i - 3, i - 2, i - 1
    da, db, dc = D[i][ia], D[i][ib], D[i][ic]

    A, B, C = x[ia], x[ib], x[ic]

    p, w = solveEQ3(A, B, C, da, db, dc)

    return p, w


def is_feasible(
    D: Dict[int, Dict[int, float]],
    i: int,
    x: np.ndarray,
    dtol: float = 1e-3,
    verbose: bool = False,
) -> bool:
    """
    Check if the position of point i satisfies all distance constraints within the tolerance.

    Args:
    D (Dict[int, Dict[int, float]]): Dictionary containing distance information
    i (int): Index of the point to check
    x (np.ndarray): Array of current point positions
    dtol (float): Tolerance for distance constraints

    Returns:
    bool: True if all distance constraints are satisfied, False otherwise
    """

    degree = len(D[i])
    if verbose:
        print(f"Checking point {i} with degree {degree}")
    xi = x[i]
    for j in D[i]:
        xj = x[j]
        dij_eval = norm(xi - xj)
        dij = D[i][j]
        if np.abs(dij - dij_eval) > dtol:
            return False
    return True


def init_xb(D: Dict[int, Dict[int, float]]) -> np.ndarray:
    """
    Initialize the first four points of the system.

    Args:
    D (Dict[int, Dict[int, float]]): Dictionary containing distance information

    Returns:
    np.ndarray: Array of initialized point positions
    """
    vertices = set()
    for i in D:
        vertices.add(i)
        for j in D[i]:
            vertices.add(j)
    n = max(vertices) + 1
    x = np.zeros((n, 3), dtype=float)

    # Set x1 along x-axis
    d01 = D[1][0]
    x[1][0] = d01

    # Set x2 in xy-plane
    d02 = D[2][0]
    d12 = D[2][1]
    cos_theta = (d02**2 + d01**2 - d12**2) / (2 * d01 * d02)
    sin_theta = np.sqrt(1 - cos_theta**2)
    x[2][0] = d02 * cos_theta
    x[2][1] = d02 * sin_theta

    # Set x3 in 3D space
    d03 = D[3][0]
    d13 = D[3][1]
    d23 = D[3][2]
    p, w = solveEQ3(x[0], x[1], x[2], d03, d13, d23)
    x[3] = p + w

    b = np.zeros(n, dtype=int)
    b[:4] = 1

    return x, b


def bp(
    D: Dict[int, Dict[int, float]],  # distance matrix
    i: int = 0,  # current index
    x: np.ndarray = None,  # current point positions
    b: List[int] = None,  # choice of solution for each point,
    single_solution: bool = False,  # flag to stop after finding a single solution
    finished: bool = False,  # flag to stop the recursion
):
    """
    Implement the Branch and Prune algorithm to solve the DDGP problem.

    Args:
    D (Dict[int, Dict[int, float]]): Dictionary containing distance information
    i (int): Current index being processed
    x (np.ndarray): Array of current point positions
    b (List[bool]): List of boolean values indicating the choice of solution for each point
    single_solution (bool): Flag to stop after finding a single solution
    finished (bool): Flag to stop the recursion

    Returns:
    None: The function updates x in-place and prints when a solution is found
    """

    if x is None:
        x, b = init_xb(D)
        i = 4

    if i == len(x):
        # Set DATA
        DATA["bp"]["x"].append(x.copy())
        DATA["bp"]["b"].append(b.copy())
        if single_solution:
            finished = True
        return x, b, finished

    p, w = calc_pw(i, x, D)

    # Try positive direction
    x[i] = p - w
    b[i] = 0
    if is_feasible(D, i, x) and i > 3:
        x, b, finished = bp(
            D, i + 1, x, b, single_solution=single_solution, finished=finished
        )
        if finished:
            return x, b, finished

    # Try negative direction
    x[i] = p + w
    b[i] = 1
    if is_feasible(D, i, x):
        x, b, finished = bp(
            D, i + 1, x, b, single_solution=single_solution, finished=finished
        )
        if finished:
            return x, b, finished

    return x, b, finished


def test_simple(n: int = 10, p: float = 0.5, seed: int = 42):
    """
    Perform a simple test of the BP algorithm using a random DDGP instance.

    Args:
    n (int): Number of points in the DDGP instance
    seed (int): Seed for random number generation

    Returns:
    None: The function prints the results of the test
    """
    # Set random seed
    used_seed = set_random_seed(seed)
    print(f"Using random seed: {used_seed}")

    # Create random 3D points
    x = create_random_matrix(n)

    # Generate DDGP instance
    D = create_random_ddgp(x, 0.5)

    print(f"Created DDGP instance with {n} points")

    # Set DATA
    DATA["ddgp"]["n"] = n
    DATA["ddgp"]["seed"] = used_seed
    DATA["ddgp"]["p"] = p
    DATA["ddgp"]["D"] = D

    # Run BP algorithm
    start_time = time.time()
    bp(D)
    end_time = time.time()

    print(f"BP algorithm completed in {end_time - start_time:.4f} seconds")
    print(f"Found {len(DATA['bp']['x'])} solutions")


def reset_data():
    DATA["bp"]["x"] = list()
    DATA["bp"]["b"] = list()


def test_random_batch(n: int = 10, p: float = 0.5):
    for seed in range(30):
        print("\n")
        reset_data()
        test_simple(n, p, seed)


def process_instance(fn_xbsol: str):
    import pandas as pd
    import pickle as pkl

    fn_dmdgp = fn_xbsol.replace("xbsol", "dmdgp").replace(".csv", ".pkl")
    with open(fn_dmdgp, "rb") as f:
        df_dmdgp = pkl.load(f)

    # Extract the distance matrix
    n = 0
    D = dict()
    for _, row in df_dmdgp.iterrows():
        i, j = row["i"], row["j"]
        # Ensure i < j
        if i < j:
            i, j = j, i
        if i not in D:
            D[i] = dict()
        D[i][j] = row["dij"]
        n = max(n, i, j)
    n += 1

    x, b, _ = bp(D, single_solution=True)

    df_xbsol = pd.read_csv(fn_xbsol)

    b_ref = df_xbsol["b"].values

    matching_result = np.equal(b[3:], b_ref[3:]).all()
    if not matching_result:
        print(f"Failed to match the solution for {fn_xbsol}")
    return matching_result


def test_xbsol_leftmost(sample_size: int = 10):
    xbsol_leftmost_dir = "xbsol_leftmost"
    fnames = os.listdir(xbsol_leftmost_dir)
    fnames_sizes = sorted(
        [(fn, os.path.getsize(os.path.join(xbsol_leftmost_dir, fn))) for fn in fnames],
        key=lambda x: x[1],
    )

    fmatchings = []
    for fname, fsize in tqdm(fnames_sizes[:sample_size], desc="Processing instances"):
        fn_xbsol = os.path.join(xbsol_leftmost_dir, fname)
        matching_result = process_instance(fn_xbsol)
        fmatchings.append(matching_result)


if __name__ == "__main__":
    process_instance("dmdgp/1qfr_model1_chainA_segment8.csv")
    # test_xbsol_leftmost(100)
