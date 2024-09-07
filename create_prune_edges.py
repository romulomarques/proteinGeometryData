import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from create_xbsol import *

wd_segment = "segment"
wd_xbsol = "xbsol"


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

def get_prune_bsol(bsol: np.array, row: pd.Series) -> tuple:
    i = int(row["i"])
    j = int(row["j"])
    str_b = "".join([str(bit) for bit in bsol[i + 3 : j + 1]])
    return str_b


def extract_prune_edges(df_xbsol: pd.DataFrame, dij_max: float = 5) -> pd.DataFrame:
    edges = []
    
    def row_data(index:int, df_xbsol: pd.DataFrame) -> tuple:
        cols_to_extract = ["residue_number", "atom_name", "x"]
        row = df_xbsol.iloc[index]
        return row[cols_to_extract]
    
    cols_to_extract = ["residue_number", "atom_name", "x"]
    for i in range(len(df_xbsol)):
        ai_residue_number, ai_name, ai_x = row_data(i, df_xbsol)
        # add the edges for the first 4 atomss
        if ai_name == "CA" and i > 3:
            j = i - 4            
            aj_residue_number, aj_name, aj_x = row_data(j, df_xbsol)
            dij = np.linalg.norm(ai_x - aj_x)
            # j < i
            edges.append(
                (j, i, aj_name, ai_name, aj_residue_number, ai_residue_number, dij)
            )
            continue

        # ensure that the atom is H or HA
        if ai_name not in ["H", "HA"]:
            continue

        # skip the discretization edges
        for j in range(i + 4, len(df_xbsol)):
            aj_residue_number, aj_name, aj_x = row_data(j, df_xbsol)
            # ensure that the atom is H or HA
            if aj_name not in ["H", "HA"]:
                continue
            dij = np.linalg.norm(ai_x - aj_x)
            if dij < dij_max:
                edges.append(
                    (i, j, ai_name, aj_name, ai_residue_number, aj_residue_number, dij)
                )
    edges = sorted(edges)

    columns = [
        "i",
        "j",
        "i_name",
        "j_name",
        "i_residue_number",
        "j_residue_number",
        "dij",
    ]
    df_prune = pd.DataFrame(edges, columns=columns)
    return df_prune


def process_xbsol(fn_xbsol: str, verbose: bool = False) -> None:
    # read the xbsol file
    df_xbsol = read_xbsol(fn_xbsol)

    # create prune edges file as a csv
    df_prune = extract_prune_edges(df_xbsol)
    bsol = df_xbsol["b"].values
    df_prune["bsol"] = df_prune.apply(lambda row: get_prune_bsol(bsol, row), axis=1)

    fn_prune = os.path.join("prune_edges", os.path.basename(fn_xbsol))
    df_prune.to_csv(fn_prune, index=False)


def process_xbsol_test():
    fn_xbsol = "xbsol/1a1u_model1_chainA_segment0.csv"
    # fn_segment = 'segment/1ah1_model1_chainA_segment0.csv'
    process_xbsol(fn_xbsol)
    exit()


if __name__ == "__main__":
    # process_xbsol_test()

    # Ensure the dmdgp folder is created
    print("Creating prune_edges directory...")
    os.makedirs("prune_edges", exist_ok=True)

    # List all .csv files in the segment directory
    print("Processing xbsol files...")
    fn_xbsols = [f for f in os.listdir("xbsol") if f.endswith(".csv")]

    # Process the instances in parallel
    print("Creating prune_edges files...")
    with ProcessPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(
                    process_xbsol,
                    [os.path.join("xbsol", fn) for fn in fn_xbsols],
                ),
                total=len(fn_xbsols),
            )
        )
