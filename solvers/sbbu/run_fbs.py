import os
from tqdm import tqdm
from run_dfs import *
import pathlib as Path


if __name__ == "__main__":
    fn_tests = Path.Path(__file__).resolve().parents[2] / "test_files.txt"
    with open(fn_tests, "r") as fid:
        test_files = [s.strip() for s in fid.readlines()]

    fbs_active = 1

    for fn_dmdgp in tqdm(test_files):
        fn_dmdgp = os.path.join(wd_dmdgp, fn_dmdgp)

        if fn_dmdgp.endswith(".csv"):
           run_sbbu(fn_dmdgp, fbs_active)
