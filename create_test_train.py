import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import pickle
from sklearn.model_selection import train_test_split
from create_dmdgp_HA9H import get_edge_code
from typing import Tuple, List

wd_dmdgp_HA9A_bsol = "dmdgp_HA9H_bsol"


def bsol_split_train_test(random_state):

    # get all pdb codes
    pdb_codes = set()
    for fn in os.listdir(wd_dmdgp_HA9A_bsol):
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
    for fn in os.listdir(wd_dmdgp_HA9A_bsol):
        if not fn.endswith(".csv"):
            continue
        pdb_code = fn.split("_")[0]
        if pdb_code in train_pdb_codes:
            train_bsol_files.append(fn)
        else:
            test_bsol_files.append(fn)

    total_files = len(train_bsol_files) + len(test_bsol_files)
    print(
        f"Number of training files: {len(train_bsol_files)} ({len(train_bsol_files)/total_files:.2f})"
    )
    print(
        f"Number of test files: {len(test_bsol_files)} ({len(test_bsol_files)/total_files:.2f})"
    )
    return train_bsol_files, test_bsol_files


def get_repetition_indexes(fn_dmdgp):
    df = pd.read_csv(fn_dmdgp)
    return df[df["dij"] == 0.0]["j"].to_list()


def normalize_bsol(bsol: str) -> str:
    if bsol[0] == "1":
        return "0" + bsol[1:]
    return bsol


def count_bsol(fname: str) -> pd.DataFrame:
    df_bsol = pd.read_csv(fname, dtype={"bsol": str})
    df_freq = df_bsol.groupby(["code", "bsol"]).size().reset_index(name="count")
    return df_freq


def convert_bsol_to_mirror(bsol: str) -> str:
    # convert from str to binary
    bsol = np.array([int(x) for x in bsol])
    for i in range(len(bsol)):
        if bsol[i]:
            bsol[i + 1 :] = 1 - bsol[i + 1 :]
    return "".join([str(int(x)) for x in bsol])


def concatenate_bsol_data(fnames: list) -> pd.DataFrame:

    print("Collecting all SBBU binary data...")
    print("Storing a dataframe for each instance data...")

    count_bsol(fnames[0])

    with ProcessPoolExecutor() as executor:
        data = list(tqdm(executor.map(count_bsol, fnames), total=len(fnames)))

    print("Concatenating all the dataframes into a single dataframe...")
    df_dmdgp_HA9H_bsol = pd.concat(data, ignore_index=True)
    df_dmdgp_HA9H_bsol = (
        df_dmdgp_HA9H_bsol[["code", "bsol", "count"]]
        .groupby(["code", "bsol"])["count"]
        .sum()
        .reset_index()
    )

    df_dmdgp_HA9H_bsol["total"] = df_dmdgp_HA9H_bsol.groupby("code")["count"].transform(
        "sum"
    )

    print("Calculating the relative frequencies of each pair (edge_type, bsol)...")
    df_dmdgp_HA9H_bsol["relfreq"] = (
        df_dmdgp_HA9H_bsol["count"] / df_dmdgp_HA9H_bsol["total"]
    )
    df_dmdgp_HA9H_bsol["len_bsol"] = df_dmdgp_HA9H_bsol["bsol"].apply(len)

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
    fn_csv = "df_train_mf.csv"
    print(f"Saving the training data in {fn_csv}")
    df_train.to_csv(fn_csv, index=False)
    save_pkl("df_train_mf", df_train)


if __name__ == "__main__":
    random_state = 42
    train_files, test_files = bsol_split_train_test(random_state)

    train_files = [
        os.path.join(wd_dmdgp_HA9A_bsol, fn)
        for fn in train_files
        if fn.endswith(".csv")
    ]

    df_train = concatenate_bsol_data(train_files)

    # combining the symmetric bsol values
    df_train["bsol"] = df_train["bsol"].apply(convert_bsol_to_mirror)
    df_train["bsol"] = df_train["bsol"].apply(normalize_bsol)

    df_train = (
        df_train.groupby(["bsol"])
        .agg(
            {
                "code": "first",
                "len_bsol": "first",
                "count": "sum",
                "total": "first",
                "relfreq": "sum",
            }
        )
        .reset_index()
    )

    # resetting the order of the columns
    df_train = df_train[["code", "len_bsol", "bsol", "count", "total", "relfreq"]]

    df_train.sort_values(["code", "relfreq"], ascending=[True, False], inplace=True)

    save_train_test(df_train, test_files)
