import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import pickle
from sklearn.model_selection import train_test_split
from create_dmdgp_HA9H_order import set_edges_code
from typing import Tuple, List

wd_dmdgp_HA9A_sbbu = "dmdgp_HA9H_sbbu"


def bsol_split_train_test(random_state):

    # get all pdb codes
    pdb_codes = set()
    for fn in os.listdir(wd_dmdgp_HA9A_sbbu):
        if not fn.endswith(".csv"):
            continue
        pdb_code = fn.split("_")[0]
        pdb_codes.add(pdb_code)
    pdb_codes = list(pdb_codes)

    # split pdb codes into train and test
    train_pdb_codes, _ = train_test_split(
        pdb_codes, test_size=0.2, random_state=random_state
    )

    train_bsol_files = []
    test_bsol_files = []

    # split bsol files into train and test
    for fn in os.listdir(wd_dmdgp_HA9A_sbbu):
        if not fn.endswith(".csv"):
            continue
        pdb_code = fn.split("_")[0]
        if pdb_code in train_pdb_codes:
            train_bsol_files.append(fn)
        else:
            test_bsol_files.append(fn)

    return train_bsol_files, test_bsol_files


def get_edge_type(row: pd.Series) -> str:
    return f"{row['i_name']}{int(row['j']) - int(row['i'])}{row['j_name']}"


def get_repetition_indexes(fn_dmdgp):
    df = pd.read_csv(fn_dmdgp)
    return df[df["dij"] == 0.0]["j"].to_list()


def flip_bsol(bsol: str) -> str:
    # flip all the bits if the first bit is 1
    if bsol[0] == "1":
        bsol = "".join(["1" if bit == "0" else "0" for bit in bsol])
    return bsol


def count_bsol(fname: str) -> pd.DataFrame:
    df_dmdgp_HA9H_bsol = pd.read_csv(fname, dtype={"bsol": str})
    # keep only the pruning edges
    df_dmdgp_HA9H_bsol = df_dmdgp_HA9H_bsol[
        df_dmdgp_HA9H_bsol["j"] - df_dmdgp_HA9H_bsol["i"] > 3
    ]
    # set the edge type
    df_dmdgp_HA9H_bsol["type"] = df_dmdgp_HA9H_bsol.apply(get_edge_type, axis=1)

    df_freq = (
        df_dmdgp_HA9H_bsol.groupby(["type", "bsol"]).size().reset_index(name="count")
    )

    return df_freq


def collect_all_bsol_data(fnames: list) -> pd.DataFrame:

    print("Collecting all SBBU binary data...")
    print("Storing a dataframe for each instance data...")
    with ProcessPoolExecutor() as executor:
        data = list(tqdm(executor.map(count_bsol, fnames), total=len(fnames)))

    print("Concatenating all the dataframes into a single dataframe...")
    df_dmdgp_HA9H_bsol = pd.concat(data, ignore_index=True)
    df_dmdgp_HA9H_bsol = (
        df_dmdgp_HA9H_bsol[["type", "bsol", "count"]]
        .groupby(["type", "bsol"])["count"]
        .sum()
        .reset_index()
    )
    df_dmdgp_HA9H_bsol["total"] = df_dmdgp_HA9H_bsol.groupby("type")["count"].transform(
        "sum"
    )

    print("Calculating the relative frequencies of each pair (edge_type, bsol)...")
    df_dmdgp_HA9H_bsol["relfreq"] = (
        df_dmdgp_HA9H_bsol["count"] / df_dmdgp_HA9H_bsol["total"]
    )
    df_dmdgp_HA9H_bsol["len_bsol"] = df_dmdgp_HA9H_bsol["bsol"].apply(len)
    df_dmdgp_HA9H_bsol["code"] = df_dmdgp_HA9H_bsol["type"].apply(set_edges_code)

    df_dmdgp_HA9H_bsol.sort_values(
        ["code", "len_bsol", "relfreq"], ascending=[True, True, False], inplace=True
    )
    return df_dmdgp_HA9H_bsol


def save_pkl(fname: str, df: pd.DataFrame):
    with open(fname + ".pkl", "wb") as f:
        pickle.dump(df, f)


def save_train_test(df_train: pd.DataFrame, test_files: list):
    # saving the file names of the test files
    fn_txt = "test_files.txt"
    print(f"Saving the test files in {fn_txt}")
    with open(fn_txt, "w") as f:
        for fn_txt in test_files:
            f.write(fn_txt + "\n")

    # saving the dataframe results of the training files
    fn_csv = "df_train.csv"
    print(f"Saving the training data in {fn_csv}")
    df_train.to_csv(fn_csv, index=False)
    save_pkl("df_train", df_train)


if __name__ == "__main__":
    random_state = 42
    train_files, test_files = bsol_split_train_test(random_state)
    # csv_files = [os.path.join(wd_dmdgp_HA9A_sbbu, fn) for fn in os.listdir('dmdgp_HA9H_sbbu') if fn.endswith('.csv')]
    train_files = [
        os.path.join(wd_dmdgp_HA9A_sbbu, fn)
        for fn in train_files
        if fn.endswith(".csv")
    ]
    df = collect_all_bsol_data(train_files)
    df_train = df[
        (df["type"] == "C4CA") | (df["type"] == "HA9H") | (df["type"] == "HA6HA")
    ]

    # keep only the relevant columns
    df_train = df_train[
        ["code", "len_bsol", "bsol", "type", "count", "total", "relfreq"]
    ]

    # combining the symmetric bsol values
    df_train["bsol"] = df_train["bsol"].apply(lambda x: flip_bsol(x))
    df_train = (
        df_train.groupby(["bsol"])
        .agg(
            {
                "code": "first",
                "len_bsol": "first",
                "type": "first",
                "count": "sum",
                "total": "first",
                "relfreq": "sum",
            }
        )
        .reset_index()
    )

    df_train.sort_values(["code", "relfreq"], ascending=[True, False], inplace=True)

    save_train_test(df_train, test_files)
