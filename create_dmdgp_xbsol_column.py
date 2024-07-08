import os
import pickle
import numpy as np
from fbs.algorithms import *
import multiprocessing as mp
from tqdm import tqdm


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


def process_instance(fn: str):
    """
    Process a single DMDGP instance file.

    :param fn: str, path to the input pickle file
    """
    df, dmdgp = read_dmdgp(fn)
    dfs = DFS(dmdgp)
    x, _, _ = bp(dmdgp, dfs)
    bsol = determineT(dmdgp, x)
    df["bsol"] = df.apply(lambda row: get_bsol(bsol, row), axis=1)

    with open(fn, "wb") as f:
        pickle.dump(df, f)

    fn_csv = os.path.join(fn.replace(".pkl", ".csv"))
    df.to_csv(fn_csv, index=False)


def main():
    """
    Main function to parallelize DMDGP instance processing with a progress bar.
    """
    dmdgp_dir = "dmdgp"
    file_list = [
        os.path.join(dmdgp_dir, fn)
        for fn in os.listdir(dmdgp_dir)
        if fn.endswith(".pkl")
    ]

    num_cores = mp.cpu_count()

    with mp.Pool(processes=num_cores) as pool:
        list(
            tqdm(
                pool.imap(process_instance, file_list),
                total=len(file_list),
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
