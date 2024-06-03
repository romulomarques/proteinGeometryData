import os
import pickle
import unittest
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures
from sklearn.model_selection import train_test_split
from numpy.linalg import norm
from fbs.algorithms import DDGP, DistanceBounds

wd_xsol = "xsol"
wd_bsol = "bsol"
wd_ddgp = "ddgp"

os.makedirs(wd_ddgp, exist_ok=False)


def bsol_split_train_test(random_state):
    print("train_test_split_bsol")

    # get all pdb codes
    pdb_codes = set()
    for fn in os.listdir(wd_bsol):
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
    for fn in os.listdir(wd_bsol):
        if not fn.endswith(".csv"):
            continue
        pdb_code = fn.split("_")[0]
        if pdb_code in train_pdb_codes:
            train_bsol_files.append(fn)
        else:
            test_bsol_files.append(fn)

    return train_bsol_files, test_bsol_files


def create_ddgp(fn_bsol, noise=1.0, verbose=False):
    if verbose:
        print("create_ddgp")
    bsol = pd.read_csv(os.path.join(wd_bsol, fn_bsol))

    # p[index_new] = index_old
    p = bsol["order"].to_list()
    n = len(p)
    x = np.loadtxt(os.path.join(wd_xsol, fn_bsol.replace("_binary.csv", ".sol")))

    # pinv[index_old] = index_new
    pinv = np.zeros(n, dtype=int)
    for i in range(n):
        pinv[p[i]] = i

    x = x[p]

    parents = [[], [], []]
    for i in range(3, len(p)):
        parents_i = bsol[["N_1", "N_2", "N_3"]].iloc[i].values
        parents_i = [int(parents_i[j]) for j in range(3)]
        parents.append([pinv[j] for j in parents_i])

    d = {i: dict() for i in range(n)}

    d[0][1] = norm(x[0] - x[1])
    d[0][2] = norm(x[0] - x[2])
    d[1][2] = norm(x[1] - x[2])

    d[0][1] = DistanceBounds(d[0][1], d[0][1])
    d[0][2] = DistanceBounds(d[0][2], d[0][2])
    d[1][2] = DistanceBounds(d[1][2], d[1][2])

    # adding discretizable distances
    for i, row_i in bsol.iterrows():
        if i < 3:
            continue
        d[i] = dict()
        for j in parents[i]:
            dij = norm(x[i] - x[j])
            d[i][j] = DistanceBounds(dij, dij)

        # add (CA,CA) distances
        if row_i["atom_name"] == "CA":
            j = i - 5
            row_j = bsol.loc[j]
            assert row_j["atom_name"] == "CA"
            dij = norm(x[i] - x[j])
            d[i][j] = DistanceBounds(dij, dij)

    # symmetric
    for i in d.keys():
        for j, dij in d[i].items():
            d[j][i] = dij

    # adding distance that depending on x
    bsol_H = bsol[bsol["atom_name"].str.startswith("H")]
    for i in range(bsol_H.shape[0]):
        row_i = bsol.loc[i]
        for j in range(i + 1, bsol_H.shape[0]):
            dij = norm(x[i] - x[j])
            if dij > 5.0:
                continue
            if i in d and j in d[i]:
                continue
            t_upper = np.random.uniform(0.0, noise)
            t_lower = np.random.uniform(0.0, noise)
            d[i][j] = DistanceBounds(dij - t_lower, dij + t_upper)
            d[j][i] = d[i][j]

    # symmetric
    for i in d.keys():
        for j, dij in d[i].items():
            d[j][i] = dij

    D = DDGP(d, parents)

    # save DDGP to pkl
    fn = os.path.join(wd_ddgp, fn_bsol.replace("_binary.csv", "_ddgp.pkl"))
    if verbose:
        print(f"save DDGP to {fn}")
    with open(fn, "wb") as f:
        pickle.dump(D, f)

    return D


def flip_if_needed(b):
    # Padronizando o quarto elemento como 1
    if len(b) > 3 and b[3] == 0:
        b = [1 - x for x in b]
    return b


def create_slices(train_bsol_files):
    print("create_slices")
    # establishes the size of binary subsequences to be collected as the numbers of consecutive residues in the pieces.
    res_quantities = [1, 2, 3, 4, 5]
    n_atoms_per_res = 5
    slice_sizes = [
        n_atoms_per_res * res_quantities[i] for i in range(len(res_quantities))
    ]

    B = []  # list of b[4:]
    for fn_bsol in tqdm(train_bsol_files):
        df = pd.read_csv(os.path.join(wd_bsol, fn_bsol))
        b_full = [int(i) for i in df["b"].to_list()[4:]]  # int values

        for size in slice_sizes:
            # get the first binary subsequence from the current segment: the first 'size' atoms.
            b = b_full[0:size]

            b_str = "".join([str(i) for i in b])
            B.append(b_str)

            # get all the other binary subsequences from the current segment: stepping from
            # a residue to the next residue (5-atoms steps).
            for slice_start in range(5, len(b_full), 5):
                # add the first three fixed atoms and the first branching atom (the fourth one in the order).
                b = b_full[(slice_start - 4) : (slice_start + size)]  # int values

                # flip the binary subsequence around the fourth atom if it is necessary.
                b = flip_if_needed(b)
                b = b[4:]

                b_str = "".join([str(i) for i in b])
                B.append(b_str)

    B, count = np.unique(B, return_counts=True)

    df = pd.DataFrame({"bsol": B, "count": count})
    df["size"] = df["bsol"].apply(len)

    df.sort_values(["size", "count"], inplace=True)
    df_total = df[["size", "count"]].groupby(["size"]).sum().reset_index()

    df["relfreq"] = df.apply(
        lambda x: x["count"]
        / df_total[df_total["size"] == x["size"]]["count"].values[0],
        axis=1,
    )

    print("   assertions")
    for size in df["size"].unique():
        # assert sum relfreq = 1
        assert np.isclose(df[df["size"] == size]["relfreq"].sum(), 1.0)

    # save the slices to csv
    fn = "df_slices.csv"
    print(f"   save slices to {fn}")
    df.to_csv(fn, index=False)


class TestCreateDDGP(unittest.TestCase):
    def test_create_ddgp(self):
        fn_bsol = "1abt_model1_chainA_segment0_binary.csv"
        D = create_ddgp(fn_bsol)

        bsol = pd.read_csv(os.path.join(wd_bsol, fn_bsol))
        b = bsol["b"].fillna(1).values
        b = "".join(b.astype(str))
        p = bsol["order"].to_list()

        x = np.loadtxt(os.path.join(wd_xsol, fn_bsol.replace("_binary.csv", ".sol")))
        x = x[p]

        self.assertTrue(D.check_bsol(b))
        self.assertTrue(D.check_xsol(x))


if __name__ == "__main__":
    # unittest.main()

    random_state = 42
    train_bsol_files, test_bsol_files = bsol_split_train_test(random_state)

    # create ddgp tests
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Wrap the executor map with tqdm for progress bar
        list(
            tqdm(executor.map(create_ddgp, test_bsol_files), total=len(test_bsol_files))
        )

    # create slices
    create_slices(train_bsol_files)
