import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from fbs.algorithms import *
from typing import Tuple, List
from concurrent.futures import ProcessPoolExecutor
import time
import functools
import seaborn as sns
import matplotlib.pyplot as plt

os.makedirs("dmdgp_leftmost", exist_ok=True)


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        wrapper.total_time += elapsed_time
        wrapper.call_count += 1
        wrapper.times.append(elapsed_time)
        return result

    wrapper.total_time = 0
    wrapper.call_count = 0
    wrapper.times = []
    return wrapper


# @timeit
def read_dmdgp(fn: str) -> DDGP:
    """
    Read a DMDGP instance from a pickle file and convert it to a DDGP object.

    :param fn: str, path to the input pickle file
    :return: tuple(DataFrame, DDGP), the original dataframe and the DDGP object
    """
    with open(fn, "rb") as f:
        df = pickle.load(f)

    D = {}
    for _, row in df.iterrows():
        i, j, dij = row["i"], row["j"], row["dij"]
        i, j = min(i, j), max(i, j)
        if j not in D:
            D[j] = {}
        D[j][i] = dij

    parents = []
    for i in range(max(D.keys()) + 1):
        if i < 3:
            parents.append([])
        else:
            parents.append([i - 3, i - 2, i - 1])

    dmdgp = DDGP(D, parents)
    return df, dmdgp


# @timeit
def get_bsol(bsol: np.array, row: pd.Series) -> str:
    """
    Extract the binary solution string for a given row.

    :param bsol: np.array, the binary solution array
    :param row: pandas.Series, a row from the DataFrame
    :return: str, the binary solution string for the row
    """
    i, j = row["i"], row["j"]
    # discretization constraint has no binary solution
    if np.abs(i - j) < 3:
        return ""
    row_bsol = "".join([str(x) for x in bsol[(i + 3) : (j + 1)]])
    return row_bsol


# @timeit
def flip_bsol(bsol: list, repetitions: list) -> List[int]:
    # ensure that bsol[3] == 1
    if bsol[3] == 0:
        bsol[3:] = 1 - bsol[3:]
    # ensure that repeated vertices have bit 0
    for i, is_repetition in enumerate(repetitions):
        if is_repetition:
            bsol[i] = 0

    return bsol


# @timeit
def flip_bsol_by_symmetry_vertices(
    bsol: np.array, symmetries: list, repetitions: list
) -> List[int]:
    # Find the leftmost symmetry solution
    for i, is_symmetry in enumerate(symmetries):
        if i < 4:
            continue
        if is_symmetry and (bsol[i] == 1):
            bsol[i:] = 1 - bsol[i:]

    # ensure that repeated vertices have bit 0
    for j, is_repetition in enumerate(repetitions):
        if is_repetition:
            bsol[j] = 0

    return bsol


# row: row of the df_xbsol
# @timeit
def check_symmetry_vertex(df_dmdgp: pd.DataFrame, df_xbsol: pd.DataFrame) -> np.array:
    is_symmetry_vertex = np.ones(len(df_xbsol), dtype=int)
    for _, row in df_dmdgp.iterrows():
        i, j = row["i"], row["j"]
        if i > j:
            i, j = j, i
        if (i + 4) < j + 1:
            is_symmetry_vertex[(i + 4) : j + 1] = 0

    # ensure that repeated vertices have bit 0
    is_symmetry_vertex[df_xbsol["is_repetition"] == 1] = 0
    return is_symmetry_vertex


# @timeit
def check_repeated_vertex(reorder: list, index: int) -> int:
    current_atom_number = reorder[index]
    for i in range(index - 1, -1, -1):
        if reorder[i] == current_atom_number:
            return 1

    return 0


# @timeit
def read_xbsol(fn_xbsol: str) -> np.array:
    """
    Read the xbsol array from a CSV file.

    :param fn_xbsol: str, path to the input CSV file
    :return: np.array, the xbsol array
    """
    df = pd.read_csv(fn_xbsol)
    bsol: np.ndarray = df["bsol"].values

    # flip if bsol[3] == 0
    if bsol[3] == 0:
        bsol = 1 - bsol

    return bsol


# @timeit
def determineX(D: DDGP, T: list, x=[], start=4):
    # immerse the first 4 vertices in R^3
    if len(x) < 4:
        x = init_x(D, T[3])

    for i in range(start, len(x)):
        calc_x(i, T[i], x, D)
    return x


# @timeit
def get_vertices_without_bit(
    bsol: list, repetitions: list, dmdgp: DDGP
) -> Tuple[list, list]:
    vertices_without_bit = []
    # T = flip_bsol(T, repetitions)
    x = determineX(dmdgp, bsol)
    if not dmdgp.check_xsol(x):
        return [], []
    x_cp = x.copy()
    for i in range(dmdgp.n):
        if i < 4:
            continue
        if bsol[i] == 1:
            bsol[i] = 0
            determineX(dmdgp, bsol, x=x_cp, start=i)
            is_solution = dmdgp.check_xsol(x_cp)
            if is_solution:
                x[i:] = x_cp[i:]
                vertices_without_bit.append(i)
            else:
                x_cp[i:] = x[i:]
                bsol[i] = 1

    return bsol, x, vertices_without_bit


# @timeit
def process_instance(fn_xbsol_leftmost: str) -> None:
    """
    Process a single xbsol_leftmost instance file.

    :param fn: str, path to the input csv file
    """
    df_xbsol_leftmost = pd.read_csv(fn_xbsol_leftmost)

    fn_dmdgp = os.path.join(
        "dmdgp", os.path.basename(fn_xbsol_leftmost).replace(".csv", ".pkl")
    )
    with open(fn_dmdgp, "rb") as f:
        df_dmdgp = pickle.load(f)

    bsol = np.array(df_xbsol_leftmost["b"])

    # updating the binary solution of each prunning edge with the leftmost symmetric binary solution
    df_dmdgp["bsol"] = df_dmdgp.apply(lambda row: get_bsol(bsol, row), axis=1)

    fn_dmdgp = os.path.join("dmdgp_leftmost", os.path.basename(fn_dmdgp))
    with open(fn_dmdgp, "wb") as f:
        pickle.dump(df_dmdgp, f)
    df_dmdgp.to_csv(fn_dmdgp.replace(".pkl", ".csv"), index=False)


def test_single():
    """
    Test the process_instance function on a single instance.
    """
    # fn = "xbsol_leftmost/1a1u_model1_chainA_segment0.csv"
    # fn = "xbsol_leftmost/1adz_model1_chainA_segment2.csv"
    fn = "xbsol_leftmost/1a11_model1_chainA_segment0.csv"
    import time

    tic = time.time()
    solved = process_instance(fn)
    toc = time.time()
    print(f"Elapsed time: {toc - tic:.2f} seconds")


def test_profile(sample_size=20, timeit=True):
    dmdgp_dir = "dmdgp"
    dmdgp_files = [
        os.path.join(dmdgp_dir, fn)
        for fn in os.listdir(dmdgp_dir)
        if fn.endswith(".pkl")
    ]

    dmdgp_files = dmdgp_files[:sample_size]

    for fn in sorted(dmdgp_files):
        process_instance(fn)

    if not timeit:
        return

    # Collect profiling results
    profiling_data = []
    for func in [
        read_dmdgp,
        get_bsol,
        flip_bsol,
        flip_bsol_by_symmetry_vertices,
        check_symmetry_vertex,
        check_repeated_vertex,
        determineX,
        read_xbsol,
        get_vertices_without_bit,
        process_instance,
    ]:
        total_time = func.total_time
        num_calls = func.call_count
        avg_time = sum(func.times) / len(func.times) if func.times else float("nan")
        profiling_data.append(
            {
                "Function": func.__name__,
                "Total Time (s)": total_time,
                "Number of Calls": num_calls,
                "Average Time (s)": avg_time,
            }
        )

    # Create a DataFrame
    df_profiling = pd.DataFrame(profiling_data)

    # Plot the results
    plt.figure(figsize=(12, 8))

    # Plot total time
    plt.subplot(3, 1, 1)
    sns.barplot(x="Total Time (s)", y="Function", data=df_profiling, palette="viridis")
    plt.title("Total Time per Function")

    # Plot number of calls
    plt.subplot(3, 1, 2)
    sns.barplot(x="Number of Calls", y="Function", data=df_profiling, palette="viridis")
    plt.title("Number of Calls per Function")

    # Plot average time
    plt.subplot(3, 1, 3)
    sns.barplot(
        x="Average Time (s)", y="Function", data=df_profiling, palette="viridis"
    )
    plt.title("Average Time per Function")

    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to parallelize DMDGP instance processing with a progress bar.
    """
    xbsol_leftmost_dir = "xbsol_leftmost"
    xbsol_leftmost_files = [
        os.path.join(xbsol_leftmost_dir, fn) for fn in os.listdir(xbsol_leftmost_dir)
    ]

    num_cores = os.cpu_count()

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        list(
            tqdm(
                executor.map(process_instance, xbsol_leftmost_files),
                total=len(xbsol_leftmost_files),
                desc="Processing files",
            )
        )


if __name__ == "__main__":
    # test_single()
    # test_profile(timeit=True)
    main()
