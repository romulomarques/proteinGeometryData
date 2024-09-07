import os
import subprocess
from tqdm import tqdm
from pathlib import Path

wd = Path(__file__).resolve().parents[2]
wd_dmdgp = os.path.join(wd, "dmdgp_HA9H")
fn_solver = os.path.join(wd, "solvers", "sbbu.exe")


def run_sbbu(fn_dmdgp, fbs_active, dfs_all=0, dtol=1e-7):
    # Check if the sbbu.exe exists
    if not os.path.isfile(fn_solver):
        raise FileNotFoundError(f"Error: {fn_solver} does not exist.")

    # Construct the path to the instance file
    fn_dmdgp = os.path.join(wd_dmdgp, fn_dmdgp)

    if not os.path.isfile(fn_dmdgp):
        raise FileNotFoundError(f"Error: {fn_dmdgp} does not exist.")

    try:
        cmd = f'{fn_solver} -nmr "{fn_dmdgp}" -tmax 300 -dfs_all {dfs_all} -fbs {fbs_active} -dtol {dtol}'
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    fbs_active = 0
    for fn_dmdgp in tqdm(os.listdir(wd_dmdgp)):
        if fn_dmdgp.endswith(".csv"):
            run_sbbu(fn_dmdgp, fbs_active)
