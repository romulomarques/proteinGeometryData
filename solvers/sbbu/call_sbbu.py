import os
import subprocess
from tqdm import tqdm


def run_sbbu(sbbu_dir, dmdgp_dir, out_dir, instance_file):
    # Construct the path to the sbbu.exe
    exe_path = os.path.join(sbbu_dir, "sbbu.exe")

    # Check if the sbbu.exe exists
    if not os.path.isfile(exe_path):
        raise FileNotFoundError(f"Error: {exe_path} does not exist.")

    # Construct the path to the instance file
    instance_path = os.path.join(dmdgp_dir, instance_file)

    if not os.path.isfile(instance_path):
        raise FileNotFoundError(f"Error: {instance_path} does not exist.")

    try:
        # Run the sbbu.exe with the instance file as an argument
        subprocess.run(
            [exe_path, "-nmr", instance_path, "-tmax", "300", "-dfs_all", '1', "-outdir", out_dir], cwd=sbbu_dir, check=True
        )
        # print(f"Finished running sbbu.exe with instance {instance_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error running sbbu.exe with instance {instance_file}: {e}")


def main():
    # Specify the path to the root directory
    root_dir = "/home/romulosmarques/Projects/proteinGeometryData"

    # Specify the path to the dmdgp_HA9H directory
    dmdgp_dir = os.path.join(root_dir, "dmdgp_HA9H")

    # Specify the path to the output directory
    out_dir = os.path.join(root_dir, "dmdgp_HA9H_sbbu")

    # Specify the path to the sbbu directory
    sbbu_dir = os.path.join(root_dir, "solvers", "sbbu")

    # Check if the sbbu directory exists
    if not os.path.isdir(sbbu_dir):
        raise NotADirectoryError(f"Error: {sbbu_dir} does not exist.")

    # List all files in the dmdgp_HA9H directory
    for filename in tqdm(os.listdir(dmdgp_dir)):
        # Check if the file is an instance file (e.g., ends with .csv)
        if filename.endswith(".csv"):
            run_sbbu(sbbu_dir, dmdgp_dir, out_dir, filename)


if __name__ == "__main__":
    main()
