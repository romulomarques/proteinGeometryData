"""Parallelized DMDGP script with progress bar.

This script processes multiple prune_bsol files in parallel, creating
corresponding dmdgp files. It uses multiprocessing for parallel execution
and displays a progress bar using tqdm.

Typical usage example:
    python script_name.py
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

wd_xbsol = "xbsol"
wd_prune = "prune_bsol"
wd_dmdgp = "dmdgp"


def read_prune(fn_prune: str) -> pd.DataFrame:
    """Reads a prune_bsol CSV file.

    Args:
        fn_prune: A string path to the prune_bsol CSV file.

    Returns:
        A pandas DataFrame containing the prune_bsol data.
    """
    return pd.read_csv(fn_prune, dtype={"bsol": "str"})


def read_xbsol(fn_xbsol: str) -> pd.DataFrame:
    """Reads an xbsol CSV file.

    Args:
        fn_xbsol: A string path to the xbsol CSV file.

    Returns:
        A pandas DataFrame containing the xbsol data with 'x' column parsed as numpy arrays.
    """
    return pd.read_csv(
        fn_xbsol, converters={"x": lambda x: np.fromstring(x[1:-1], sep=" ")}
    )


def create_discretization_edges(df_xbsol: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """Creates discretization edges from xbsol data.

    Args:
        df_xbsol: A pandas DataFrame containing xbsol data.
        n_bins: An integer specifying the number of bins for discretization (currently unused).

    Returns:
        A pandas DataFrame containing the discretization edges.
    """
    edges = []
    for i, irow in df_xbsol.iterrows():
        for j in range(i - 3, i):
            if j < 0:
                continue
            jrow = df_xbsol.iloc[j]
            dij = np.linalg.norm(irow["x"] - jrow["x"])
            edges.append(
                {
                    "i": i,
                    "j": j,
                    "i_name": irow["atom_name"],
                    "j_name": jrow["atom_name"],
                    "i_residue_number": irow["residue_number"],
                    "j_residue_number": jrow["residue_number"],
                    "dij": dij,
                    "bsol": "",
                }
            )

    return pd.DataFrame(edges)


def create_dmdgp(fn_prune: str) -> None:
    """Creates a dmdgp file from a prune_bsol file.

    This function reads the prune_bsol and corresponding xbsol files,
    creates discretization edges, merges the data, and saves the result
    as both CSV and pickle files.

    Args:
        fn_prune: A string path to the input prune_bsol file.

    Returns:
        None
    """
    df_prune = read_prune(fn_prune)
    fn_xbsol = fn_prune.replace(wd_prune, wd_xbsol)
    df_xbsol = read_xbsol(fn_xbsol)

    df_edges = create_discretization_edges(df_xbsol, 3)

    df_dmdgp = pd.concat([df_edges, df_prune])

    df_dmdgp = df_dmdgp.sort_values(["i", "j"])

    fn_dmdgp = fn_prune.replace(wd_prune, wd_dmdgp).replace(".csv", ".pkl")
    df_dmdgp.to_pickle(fn_dmdgp)
    df_dmdgp.to_csv(fn_dmdgp.replace(".pkl", ".csv"), index=False)


def main():
    # Create output directory if it doesn't exist
    os.makedirs(wd_dmdgp, exist_ok=True)

    # Get list of all prune_bsol CSV files
    prune_files = [
        os.path.join(wd_prune, fn) for fn in os.listdir(wd_prune) if fn.endswith(".csv")
    ]

    # Process files in parallel with progress bar
    with ProcessPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(create_dmdgp, prune_files),
                total=len(prune_files),
                desc="Processing files",
            )
        )


def test_create_dmdgp():
    # Create output directory if it doesn't exist
    os.makedirs(wd_dmdgp, exist_ok=True)

    fn_prune = "1a23_model1_chainA_segment2.csv"
    create_dmdgp(os.path.join(wd_prune, fn_prune))


if __name__ == "__main__":
    main()
    # test_create_dmdgp()
