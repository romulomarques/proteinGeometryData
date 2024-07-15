import os
import pickle
import numpy as np
from fbs.algorithms import *
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd

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


def get_bsol(bsol: np.array, row) -> str:
    """
    Extract the binary solution string for a given row.

    :param bsol: np.array, the binary solution array
    :param row: pandas.Series, a row from the DataFrame
    :return: str, the binary solution string for the row
    """
    i, j = row["i"], row["j"]
    if np.abs(i - j) < 3:
        return ""
    row_bsol = "".join([str(x) for x in bsol[i : (j + 1)]])
    return row_bsol


def read_xbsol(fn_xbsol:str) -> np.array:
    """
    Read the xbsol array from a CSV file.

    :param fn_xbsol: str, path to the input CSV file
    :return: np.array, the xbsol array
    """
    df = pd.read_csv(fn_xbsol)
    bsol:np.ndarray = df['bsol'].values
    
    # flip if bsol[3] == 0
    if bsol[3] == 0:
        bsol = 1 - bsol
    
    return bsol

def process_instance(fn_dmdgp: str):
    """
    Process a single DMDGP instance file.

    :param fn: str, path to the input pickle file
    """
    df, dmdgp = read_dmdgp(fn_dmdgp)
    dfs = DFS(dmdgp)
    
    fn_xbsol = fn_dmdgp.replace("dmdgp", "xbsol")
    xbsol = read_xbsol(fn_xbsol)

def main():
    """
    Main function to parallelize DMDGP instance processing with a progress bar.
    """
    dmdgp_dir = "dmdgp"
    file_dmdgp = [
        os.path.join(dmdgp_dir, fn)
        for fn in os.listdir(dmdgp_dir)
        if fn.endswith(".pkl")
    ]

    num_cores = mp.cpu_count()

    with mp.Pool(processes=num_cores) as pool:
        list(
            tqdm(
                pool.imap(process_instance, file_dmdgp),
                total=len(file_dmdgp),
                desc="Processing files",
            )
        )


def test_single():
    """
    Test the process_instance function on a single instance.
    """
    fn = "dmdgp/1a1u_model1_chainA_segment0.pkl"
    import time

    tic = time.time()
    process_instance(fn)
    toc = time.time()
    print(f"Elapsed time: {toc - tic:.2f} seconds")


if __name__ == "__main__":
    # test_single()
    main()
