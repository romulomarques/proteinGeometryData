import os
import subprocess
from tqdm import tqdm


def run_sbbu(sbbu_exe: str, instance_file: str, outdir: str, fbs: int):
    try:
        # Run the sbbu.exe with the instance file as an argument
        cmd = f'{sbbu_exe} -nmr "{instance_file}" -tmax 300 -dfs_all 0 -outdir "{outdir}" -fbs {fbs} -dtol 1e-5'
        subprocess.run(cmd, check=True, shell=True)
        # print(f"Finished running sbbu.exe with instance {instance_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error running sbbu.exe with instance {instance_file}: {e}")


def main():
    # fn_tests = "/home/michael/gitrepos/rs_ROMULO/test_files.txt"
    fn_tests = "/home/romulosmarques/Projects/proteinGeometryData/test_files.txt"
    with open(fn_tests, "r") as fid:
        test_files = [s.strip() for s in fid.readlines()]

    # wd_HA9H = "/home/michael/gitrepos/rs_ROMULO/dmdgp_HA9H"
    # sbbu_exe = "/home/michael/gitrepos/rs_ROMULO/solvers/sbbu/sbbu.exe"
    # outdir_fbs = "/home/michael/gitrepos/rs_ROMULO/results/fbs"
    # outdir_dfs = "/home/michael/gitrepos/rs_ROMULO/results/dfs"
    
    wd_HA9H = "/home/romulosmarques/Projects/proteinGeometryData/dmdgp_HA9H"
    sbbu_exe = "/home/romulosmarques/Projects/proteinGeometryData/solvers/sbbu/sbbu.exe"
    outdir_fbs = "/home/romulosmarques/Projects/proteinGeometryData/results/fbs"
    outdir_dfs = "/home/romulosmarques/Projects/proteinGeometryData/results/dfs"

    if not os.path.isfile(sbbu_exe):
        raise FileNotFoundError(f"The {sbbu_exe} does not exist.")

    for fn in test_files:
        fn = os.path.join(wd_HA9H, fn)

        if not fn.endswith(".csv"):
            continue

        if not os.path.isfile(fn):
            raise FileNotFoundError(f"The {fn} does not exist.")

        run_sbbu(sbbu_exe, fn, outdir=outdir_dfs, fbs=0)
        run_sbbu(sbbu_exe, fn, outdir=outdir_fbs, fbs=1)


if __name__ == "__main__":
    main()
