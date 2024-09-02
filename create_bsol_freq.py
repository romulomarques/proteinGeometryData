import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import pickle
from sklearn.model_selection import train_test_split


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
    return f"{row['i_name']} {int(row['j']) - int(row['i'])} {row['j_name']}"


def count_bsol(fname: str) -> pd.DataFrame:
    df_dmdgp_HA9H_bsol = pd.read_csv(fname, dtype={'bsol': str})
    df_dmdgp_HA9H_bsol = df_dmdgp_HA9H_bsol[df_dmdgp_HA9H_bsol["j"] - df_dmdgp_HA9H_bsol["i"] > 3]
    df_dmdgp_HA9H_bsol["type"] = df_dmdgp_HA9H_bsol.apply(get_edge_type, axis=1)
    
    df_freq = df_dmdgp_HA9H_bsol.groupby(["type", "bsol"]).size().reset_index(name='count')
    
    return df_freq


def collect_all_bsol_data(fnames: list) -> pd.DataFrame:

    print("Collecting all SBBU binary data...")
    print("Storing a dataframe for each instance data...")
    with ProcessPoolExecutor() as executor:
        data = list(tqdm(executor.map(count_bsol, fnames), total=len(fnames)))

    print("Concatenating all the dataframes into a single dataframe...")
    df_dmdgp_HA9H_bsol = pd.concat(data, ignore_index=True)
    df_dmdgp_HA9H_bsol = df_dmdgp_HA9H_bsol[["type", "bsol", "count"]].groupby(["type", "bsol"])["count"].sum().reset_index()
    df_dmdgp_HA9H_bsol['total'] = df_dmdgp_HA9H_bsol.groupby('type')['count'].transform('sum')
    
    print("Calculating the relative frequencies of each pair (edge_type, bsol)...")
    df_dmdgp_HA9H_bsol['relfreq'] = df_dmdgp_HA9H_bsol['count'] / df_dmdgp_HA9H_bsol['total']
    df_dmdgp_HA9H_bsol['len_bsol'] = df_dmdgp_HA9H_bsol['bsol'].apply(len)
    df_dmdgp_HA9H_bsol.sort_values(['len_bsol','type','relfreq'], ascending=[True,True,False], inplace=True)

    print("Done!")

    return df_dmdgp_HA9H_bsol


def save_pkl(fname: str, df: pd.DataFrame):
    with open(fname + '.pkl', 'wb') as f:
        pickle.dump(df, f)


def save_train_test(df_train: pd.DataFrame, test_files: list):
    # saving the file names of the test files
    with open('test_files.txt', 'w') as f:
        for fname in test_files:
            f.write(fname + '\n')
    
    # saving the dataframe results of the training files
    df_train.to_csv('df_train.csv', index=False)
    save_pkl('df_train', df_train)


if __name__ == "__main__":
    # fname = "dmdgp_HA9H_sbbu/9pcy_model1_chainA_segment9.csv"
    # count_bsol(fname)

    random_state = 42
    train_files, test_files = bsol_split_train_test(random_state)
    # csv_files = [os.path.join(wd_dmdgp_HA9A_sbbu, fn) for fn in os.listdir('dmdgp_HA9H_sbbu') if fn.endswith('.csv')]
    train_files = [os.path.join(wd_dmdgp_HA9A_sbbu, fn) for fn in train_files if fn.endswith('.csv')]
    df = collect_all_bsol_data(train_files)
    df_filtered = df[ (df["type"] == "C 4 CA") | (df["type"] == "HA 9 H") | (df["type"] == "HA 6 HA") ]
    save_train_test(df_filtered, test_files)