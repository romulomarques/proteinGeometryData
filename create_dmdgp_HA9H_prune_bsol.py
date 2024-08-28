import os
import numpy as np
import pandas as pd
from tqdm import tqdm


solution_dir = "dmdgp_HA9H_sol"
dmdgp_HA9H_dir = "dmdgp_HA9H"


def read_dmdgp(fname : str) -> pd.DataFrame:
    df_dmdgp = pd.read_csv(fname)
    return df_dmdgp


def read_sol(fname : str) -> np.ndarray:
    x = np.loadtxt(fname)
    return x


def process_file(fname):
    fn_dmdgp = os.path.join(dmdgp_HA9H_dir, fname)
    df_dmdgp = read_dmdgp(fn_dmdgp)

    fn_sol = os.path.join(solution_dir, fname)
    fn_sol = fn_sol.replace(".csv", "_sbbu.sol")
    x = read_sol(fn_sol)

    df_dmdgp["x"] = df_dmdgp.apply(lambda row : row.index)    


if __name__ == "__main__":
    files = os.listdir(dmdgp_HA9H_dir)
    for fn in tqdm(files):
        process_file(fn)