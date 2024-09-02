import os
import subprocess
from tqdm import tqdm


def run_sbbu(sbbu_exe:str, instance_file:str, cwd:str, fbs:int):    
    try:
        # Run the sbbu.exe with the instance file as an argument
        subprocess.run(
            [sbbu_exe, "-nmr", instance_file, "-tmax", "300", "-fbs", f"{fbs}", "-outdir", cwd],
            cwd=cwd,
            check=True,
        )
        # print(f"Finished running sbbu.exe with instance {instance_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error running sbbu.exe with instance {instance_file}: {e}")


def main():
    with open("test_files.txt", "r") as fid:
        test_files = [s.strip() for s in fid.readlines()]

    wd_HA9H = "/home/michael/gitrepos/rs_ROMULO/dmdgp_HA9H"
    sbbu_exe = "/home/michael/gitrepos/rs_ROMULO/solvers/sbbu/sbbu.exe"
    cwd_fbs = "/home/michael/gitrepos/rs_ROMULO/results/fbs"
    cwd_dfs = "/home/michael/gitrepos/rs_ROMULO/results/dfs"

    if not os.path.isfile(sbbu_exe):
        raise FileNotFoundError(f"The {sbbu_exe} does not exist.")

    for fn in test_files:
        fn = os.path.join(wd_HA9H, fn)

        if not fn.endswith(".csv"):
            continue

        if not os.path.isfile(fn):
            raise FileNotFoundError(f"The {fn} does not exist.")

        run_sbbu(sbbu_exe, fn, cwd=cwd_dfs, fbs=0)
        run_sbbu(sbbu_exe, fn, cwd=cwd_fbs, fbs=1)


if __name__ == "__main__":
    main()
