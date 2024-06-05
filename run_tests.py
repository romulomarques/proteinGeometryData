import os
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from fbs.algorithms import FBS, DFS, DDGP, bp
import timeit


def process_file(fn, n, states, p, verbose=False):
    tic = time.time()
    with open(fn, "rb") as f:
        D = pickle.load(f)
    toc = time.time() - tic
    if verbose:
        print(f"Loaded {fn} in {toc:.2E} secs")

    tic = time.time()
    fbs = FBS(D, states, p)
    dfs = DFS(D)
    toc = time.time() - tic
    if verbose:
        print(f"Preprocessed {fn} in {toc:.2E} secs")

    statistic = []

    n_experiments = 20
    item = {"fn": fn, "n": n, "method": "dfs"}
    try:
        # mytime = timeit.timeit(stmt="bp(D, dfs)", number=n_experiments)
        # item["time_secs"] = mytime / n_experiments
        tic = time.time()
        _, n_infeasible, n_constraints = bp(D, dfs)
        toc = time.time() - tic
        item["time_secs"] = toc
        item["n_bt_nodes"] = dfs.n_bt_nodes
        item["n_infeasible"] = n_infeasible
        item["n_constraints"] = n_constraints
        item["n_one_nodes"] = sum(dfs.T[4:])/(dfs.n - 4)
    except:
        item["time_secs"] = None
    statistic.append(item)
    print(item)

    # print statistics
    if verbose:
        print(f"DFS: {item['time_secs']:.2E} secs")

    D.n_infeasible = 0
    item = {"fn": fn, "n": n, "method": "fbs"}
    try:
        # mytime = timeit(stmt="bp(D, fbs)", number=n_experiments)
        # item["time_secs"] = mytime / n_experiments
        tic = time.time()
        _, n_infeasible, n_constraints = bp(D, fbs)
        toc = time.time() - tic
        item["time_secs"] = toc
        item["n_bt_nodes"] = fbs.n_bt_nodes
        item["n_infeasible"] = n_infeasible
        item["n_constraints"] = n_constraints
    except:
        item["time_secs"] = None
    statistic.append(item)
    print(item)
    print()

    # print statistics
    if verbose:
        print(f"FBS: {item['time_secs']:.2E} secs")

    return statistic


def set_ddgp_bins():
    from tqdm import tqdm

    df = {"fn": [], "n": []}

    for fn in tqdm(os.listdir("ddgp")):
        fn = os.path.join("ddgp", fn)
        with open(fn, "rb") as f:
            D: DDGP = pickle.load(f)

        df["fn"].append(fn)
        df["n"].append(D.n)

    df = pd.DataFrame(df)

    # set bins
    bins = [0, 50, 100, 200, 300, np.inf]
    df["bin"] = df["n"].apply(lambda x: np.digitize(x, bins))

    return df


def ddgp_samples(ntests, df_tests):
    import random

    random.seed(42)

    df_sample = []
    df_tests = df_tests[df_tests['bin'] >= 3]
    for bin in df_tests["bin"].unique():
        df_tests_bin = df_tests[df_tests["bin"] == bin]
        if len(df_tests_bin) < ntests:
            ntests = len(df_tests_bin)
        for _ in range(ntests):
            idx = random.choice(df_tests_bin.index)
            df_sample.append(df_tests_bin.loc[idx])

    df_sample = pd.DataFrame(df_sample)
    df_sample.sort_values(["bin", "n"], inplace=True)
    return df_sample


def read_states():
    df_slices = pd.read_csv("df_slices.csv")
    df_slices.sort_values(["size", "count"], ascending=[False, False], inplace=True)

    df_slices = df_slices[df_slices["size"] <= 5]

    states = df_slices["bsol"].to_list()
    p = df_slices["relfreq"].to_list()

    num_size_5 = sum(df_slices["size"] == 5)
    assert num_size_5 == 32, f"Expected 1 size 5, got {num_size_5}"

    return states, p


def test_instance():
    states, p = read_states()

    fn = '2jxc_model1_chainA_segment2'
    fn_bsol = os.path.join('bsol', fn + '_binary.csv')
    fn_ddgp = os.path.join('ddgp', fn + '_ddgp.pkl')    
    
    bsol = pd.read_csv(fn_bsol)
    b = bsol['b'].fillna(1).astype(int).to_numpy()
    
    with open(fn_ddgp, 'rb') as f:
        D = pickle.load(f)
    
    dfs = DFS(D)
    bp(D, dfs, verbose=True)
    # fbs = FBS(D, states, p)
    # bp(D, fbs, verbose=True)

def main():
    states, p = read_states()

    df_ddgp_bins = set_ddgp_bins()

    df_ddgp_sample = ddgp_samples(10, df_ddgp_bins)

    statistic = []
    for _, row in tqdm(list(df_ddgp_sample.iterrows())):
        statistic.extend(process_file(row["fn"], row["n"], states, p))

    df_statistic = pd.DataFrame(statistic)
    df_statistic.to_csv("df_statistic.csv", index=False)


if __name__ == "__main__":
    # test_instance()
    main()
