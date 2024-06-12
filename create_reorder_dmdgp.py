import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def extract_atoms_with_reorder(df):
    min_residue_number = int(df.iloc[0]["residue_number"])
    atoms = [
        df.query(f"residue_number=={min_residue_number} and atom_name=='N'").values[0],
        df.query(f"residue_number=={min_residue_number} and atom_name=='HA'").values[0],
        df.query(f"residue_number=={min_residue_number} and atom_name=='C'").values[0],
        df.query(f"residue_number=={min_residue_number} and atom_name=='CA'").values[0],
    ]

    residue_number = min_residue_number + 1
    for i in range(4, len(df)):
        atom = df.iloc[i]
        if atom["residue_number"] == residue_number:
            atoms.append(atom.values)
        else:
            atoms.append(df.iloc[i - 3].values)
            atoms.append(atom.values)
            residue_number = atom["residue_number"]

    return atoms


def read_instance(file_path):
    # read csv file
    df = pd.read_csv(file_path)

    # Convert the x_coord, y_coord, and z_coord columns to a single column
    # with 3D point representation (np.array)
    df["x"] = list(zip(df["x_coord"], df["y_coord"], df["z_coord"]))
    df["x"] = df["x"].apply(np.array)

    # Drop the original x_coord, y_coord, and z_coord columns
    df.drop(columns=["x_coord", "y_coord", "z_coord"], inplace=True)

    def comparison(x):
        atom_score = 0
        if x == "H":
            atom_score = 0
        elif x == "N":
            atom_score = 1
        elif x == "CA":
            atom_score = 2
        elif x == "HA":
            atom_score = 3
        elif x == "C":
            atom_score = 4

        return atom_score

    df["atom_score"] = df["atom_name"].apply(comparison)
    df.sort_values(by=["residue_number", "atom_score"], inplace=True)
    df.drop(columns=["atom_score"], inplace=True)

    return df


def extract_prune_edges(atoms, dij_max=5):
    edges = []
    for i in range(len(atoms)):
        ai_residue_number, ai_name, ai_x = atoms[i][0], atoms[i][3], atoms[i][5]
        # add the only one CA pruning edge
        if ai_name == "CA" and i > 3:
            j = i - 4
            aj_residue_number, aj_name, aj_x = atoms[j][0], atoms[j][3], atoms[j][5]
            dij = np.linalg.norm(ai_x - aj_x)
            # j < i
            edges.append(
                (j, i, aj_name, ai_name, aj_residue_number, ai_residue_number, dij)
            )
            continue

        # keep only hydrogens
        if ai_name not in ["H", "HA"]:
            continue

        # skip the discretization edges
        for j in range(i + 4, len(atoms)):
            aj_residue_number, aj_name, aj_x = atoms[j][0], atoms[j][3], atoms[j][5]
            if aj_name not in ["H", "HA"]:
                continue
            dij = np.linalg.norm(ai_x - aj_x)
            if dij < dij_max:
                edges.append(
                    (i, j, ai_name, aj_name, ai_residue_number, aj_residue_number, dij)
                )
    edges = sorted(edges)
    return edges


def process_instance(fn_segment):
    # Read and process the instance
    df = read_instance(fn_segment)

    # create/save atoms file as a csv
    atoms = extract_atoms_with_reorder(df)

    fn_xsol = os.path.join("xsol", os.path.basename(fn_segment))
    df_atoms = pd.DataFrame(atoms, columns=df.columns)
    df_atoms.to_csv(fn_xsol, index=False)

    # create prune edges file as a csv
    prune_edges = extract_prune_edges(atoms)

    columns = [
        "i",
        "j",
        "i_name",
        "j_name",
        "i_residue_number",
        "j_residue_number",
        "dij",
    ]
    df = pd.DataFrame(prune_edges, columns=columns)
    fn_dmdgp = os.path.join("dmdgp", os.path.basename(fn_segment))
    df.to_csv(fn_dmdgp, index=False)


def test_process_instance():
    os.makedirs("segment", exist_ok=True)
    os.makedirs("xsol", exist_ok=True)

    fn_segment = "segment/1a1u_model1_chainA_segment0.csv"
    process_instance(fn_segment)


def main():
    # Ensure the dmdgp folder is created
    print("Creating dmdgp directory...")
    os.makedirs("dmdgp", exist_ok=True)

    print("Creating xsol directory...")
    os.makedirs("xsol", exist_ok=True)

    # List all .csv files in the segment directory
    print("Processing instances...")
    fn_segments = [f for f in os.listdir("segment") if f.endswith(".csv")]

    # Process the instances in parallel
    print("Creating DMDGP files...")
    with ProcessPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(
                    process_instance,
                    [os.path.join("segment", fn) for fn in fn_segments],
                ),
                total=len(fn_segments),
            )
        )


if __name__ == "__main__":
    # test_process_instance()
    main()
