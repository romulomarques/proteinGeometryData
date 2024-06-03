import os
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from fbs.bpfbs import FBS, DFS, DDGP, bp, load_states


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

    item = {"fn": fn, "n":n, "method": "dfs"}
    try:
        tic = time.time()
        bp(D, dfs)
        toc = time.time() - tic
        item["time_secs"] = toc
    except:
        item["time_secs"] = None
    statistic.append(item)

    # print statistics
    if verbose:
        print(f"DFS: {item['time_secs']:.2E} secs")

    item = {"fn": fn, "n":n, "method": "fbs"}
    try:
        tic = time.time()
        bp(D, fbs)
        toc = time.time() - tic
        item["time_secs"] = toc
    except:
        item["time_secs"] = None
    statistic.append(item)

    # print statistics
    if verbose:
        print(f"FBS: {item['time_secs']:.2E} secs")

    return statistic


def get_tests():
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


def create_samples(ntests, df_tests):
    import random
    random.seed(42)
    
    df_sample = []
    for bin in df_tests['bin'].unique():
        df_tests_bin = df_tests[df_tests['bin'] == bin]
        if len(df_tests_bin) < ntests:
            ntests = len(df_tests_bin)
        for _ in range(ntests):
            idx = random.choice(df_tests_bin.index)
            df_sample.append(df_tests_bin.loc[idx])
    
    df_sample = pd.DataFrame(df_sample)
    df_sample.sort_values(['bin','n'], inplace=True)
    return df_sample


def main():

    # set random seed

    states, p = load_states()

    df_tests = get_tests()

    df_sample = create_samples(5, df_tests)

    statistic = []
    for _, row in tqdm(list(df_sample.iterrows())):        
        statistic.extend(process_file(row['fn'], row['n'], states, p))

    df_statistic = pd.DataFrame(statistic)
    df_statistic.to_csv("df_statistic.csv", index=False)


def test_instance():

    import fbs
    import time
    import pickle
    import numpy as np
    import pandas as pd

    fn_ddgp = "ddgp/1nm4_model1_chainA_segment0_ddgp.pkl"
    with open(fn_ddgp, "rb") as f:
        D:DDGP = pickle.load(f)

    fn_sol = "original_sol/" + os.path.basename(fn_ddgp).replace('_ddgp.pkl', '.sol')
    xsol = np.loadtxt(fn_sol)
    
    fn_bin = "binary/" + os.path.basename(fn_ddgp).replace('_ddgp.pkl', '_binary.csv')
    xbin = pd.read_csv(fn_bin)
    xbin['b'] = xbin['b'].fillna(1)
    xbin['b'] = xbin['b'].astype(int)

    # permute the solution
    xsol = xsol[xbin['order']]
    assert D.check_coord_solution(xsol)

    assert D.check_binary_solution(xbin['b'].values)
    
    print(f"xbin: {xbin['b'].tolist()}")

    states, p = load_states()

    fbs = FBS(D, states, p)
    dfs = DFS(D)

    tic = time.time()
    x = bp(D, dfs)
    toc = time.time() - tic
    print(f"DFS: {toc:.2E} secs")

    tic = time.time()
    x = bp(D, fbs)
    toc = time.time() - tic
    print(f"FBS: {toc:.2E} secs")


if __name__ == "__main__":
    # main()
    test_instance()
