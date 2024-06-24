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


# def create_slices(fnames):
    wd_bsol = "bsol"

    # establishes a numerical label for each kind of binary subsequence to be collected.
    constraint_types = [0, 1]
    # establishes the start and the end of a slice, considering the start is in a residue 'i'
    # and the end in the residue 'i+1'.
    start_atoms = ['HA', 'H']
    end_atoms = ['HA', 'H']
    slice_size = [6, 6]

    B = {x: [] for x in constraint_types}  # lists of b by their kind of constraint
    for fn_bsol in tqdm(fnames):
        df = pd.read_csv(os.path.join(wd_bsol, fn_bsol))
        n_df = len(df)

        for this_type in constraint_types:
            thisB = [] # list of b
            current_start = start_atoms[this_type]
            current_end = end_atoms[this_type]
            step = slice_size[this_type]

            # get all the binary subsequences from the current segment
            slice_start = 4
            while slice_start + step < n_df:
                # looking for the correct first atom of the slice
                if df.iloc[slice_start]["atom_name"] != current_start:
                    slice_start += 1
                    continue

                # checking the correctness of the slice type
                if df.iloc[slice_start + step]["atom_name"] != current_end:
                    raise Exception(f"Slice Error in {fn_bsol}: the slice does not end in {current_end} !")

                # add the first three fixed atoms and the first branching atom (the fourth one in the order).
                b = b_full[(slice_start - 4) : (slice_start + step)]  # int values

                # flip the binary subsequence around the fourth atom if it is necessary.
                b = flip_if_needed(b)
                b = b[4:]

                b_str = "".join([str(i) for i in b])
                thisB.append(b_str)

                slice_start += step

            B[this_type] += thisB

    # counting, for each constraint type, the frequency of each binary subsequency
    count = {x: [] for x in constraint_types}
    for this_type in constraint_types:
        B[this_type], count[this_type] = np.unique(B[this_type], return_counts=True) # it also converts lists to np.array

    # creating a dataframe column that indicates the constraint type of a binary subsequency
    constraint_column = []
    for this_type in constraint_types:
        constraint_column += len(B[this_type]) * [this_type]

    # joining in a single column the binary subsequencies of all constraint types.
    # we do the same action for the binary subsequencyy frequencies.
    b_column, count_column = [], []
    for this_type in constraint_types:
        b_column += list(B[this_type])
        count_column += list(count[this_type])

    df = pd.DataFrame({"bsol": b_column, "count": count_column, "constraint": constraint_column})

    df.sort_values(["constraint", "count"], inplace=True)
    df_total = df[["constraint", "count"]].groupby(["constraint"]).sum().reset_index()

    df["relfreq"] = df.apply(
        lambda x: x["count"]
        / df_total[df_total["constraint"] == x["constraint"]]["count"].values[0],
        axis=1,
    )

    print("   assertions")
    for constraint in df["constraint"].unique():
        # assert sum relfreq = 1
        assert np.isclose(df[df["constraint"] == constraint]["relfreq"].sum(), 1.0)

    # save the slices to csv
    fn = "df_dmdgp_slices.csv"
    print(f"   save slices to {fn}")
    df.to_csv(fn, index=False)


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