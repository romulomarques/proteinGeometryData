import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

wd_xbsol = "xbsol"
wd_prune = "prune_bsol"

# flip binary array considering it starts in the first non-fixed bit
def flip_if_needed(b):
    if b[0] == 1:
        b = [1 - x for x in b]
    return b


def get_constraint(row):
    i = int(row["i"])
    j = int(row["j"])
    return str(row["i_name"]) + " " + str(j-i) + " " + str(row["j_name"])


def read_csv_file(fn_csv):
    return pd.read_csv(fn_csv, dtype={'bsol': 'str'})


def create_df_frequencies(train_prune_bsol_files):
    csv_files = [os.path.join(wd_prune, fn) for fn in train_prune_bsol_files]

    with ProcessPoolExecutor() as executor:
        data = list(tqdm(executor.map(read_csv_file, csv_files), total=len(csv_files)))

    # dataframe containing all the data of the 'prune_bsol' files
    df_prune = pd.concat(data, ignore_index=True)
    
    # flip bsol that starts with 1
    df_prune["bsol"] = df_prune["bsol"].apply(lambda b_str : "".join(str(y) for y in flip_if_needed([int(x) for x in b_str])))
    
    # df_prune = pd.read_csv(os.path.join(wd_prune, fn), dtype={"bsol": "str"})

    # # flip bsol that starts with 1
    # df_prune["bsol"] = df_prune["bsol"].apply(lambda b_str : "".join(str(y) for y in flip_if_needed([int(x) for x in b_str])))

    # categorize the prune edges
    df_prune["constraint"] = df_prune.apply(get_constraint, axis=1)

    # count the frequency of each binary subsequence of each constraint type.
    df_prune["count"] = 0
    df_prune = df_prune[["bsol", "constraint", "count"]].groupby(["bsol", "constraint"]).count().reset_index()

    # count the number of occurrencies of each constraint type.
    df_prune["total"] = df_prune.groupby(["constraint"])["count"].transform("sum")

    # calculate the relative frequency of each binary subsequency given its constraint type.
    df_prune["rel_freq"] = df_prune["count"] / df_prune["total"]

    # check if the relative frequencies have been correctly calculated.
    assert df_prune.groupby(["constraint"])["rel_freq"].apply("sum").all()

    return df_prune


def test_create_slices():
    fn_prune = "9pcy_model1_chainA_segment7.csv"
    create_df_frequencies(os.listdir("prune_bsol"))


if __name__ == "__main__":
    test_create_slices()