import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


# add, for each residue, a repetition of the CA atom at the end of the residue
def extract_atoms_with_reorder(df: pd.DataFrame) -> list:
    
    def atom_score_first_residue(row: pd.Series) -> int:
        atom_score = 0
        if row["atom_name"] == "N":
            atom_score = 0
        elif row["atom_name"] == "HA":
            atom_score = 1
        elif row["atom_name"] == "C":
            atom_score = 2
        elif row["atom_name"] == "CA":
            atom_score = 3            
        return atom_score

    df_1st_4 = df.iloc[:4].copy()
    df_1st_4["score"] = df.apply(atom_score_first_residue, axis=1)
    df_1st_4.sort_values(by=["score"], inplace=True)
    df_1st_4 = df_1st_4.drop(["score"], axis=1)
    atoms = [atom for atom in df_1st_4.values]

    residue_number = int(df.iloc[0]["residue_number"]) + 1
    for i in range(4, len(df)):
        atom = df.iloc[i]
        if atom["residue_number"] == residue_number:
            atoms.append(atom.values)
        else:
            # append duplicate CA atom
            atoms.append(df.iloc[i - 3].values)
            atoms.append(atom.values)
            residue_number = atom["residue_number"]

    return atoms


def check_segment(df: pd.DataFrame) -> bool:
    # check if the segment is valid

    df_grouped_by_residue = (
        df[["residue_number", "atom_name"]]
        .groupby("residue_number")
        .agg(list)
        .reset_index()
    )

    # the first residue should have the following atoms: N, CA, C, HA
    expected_atoms = {"N", "HA", "C", "CA"}
    # check if there is the exactly number of expected atoms
    if len(df_grouped_by_residue.iloc[0]["atom_name"]) != len(expected_atoms):
        return False
    # check if the atoms are the expected ones
    if set(df_grouped_by_residue.iloc[0]["atom_name"]) != expected_atoms:
        return False

    # the rest of the residues should have the following atoms: N, CA, C, H, HA
    expected_atoms = {"N", "HA", "H", "C", "CA"}

    def is_valid_residue(row: pd.Series) -> bool:
        # check if there is the exactly number of expected atoms
        if len(row["atom_name"]) != len(expected_atoms):
            return False
        # check if the atoms are the expected ones
        if set(row["atom_name"]) != expected_atoms:
            return False
        # check if the atoms are in the correct order
        return True

    return all(df_grouped_by_residue[1:].apply(is_valid_residue, axis=1))


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

    # check if the segment is valid
    if not check_segment(df):
        raise ValueError(f"Invalid segment: {file_path}")

    return df


def extract_prune_edges(atoms, dij_max=5):
    edges = []
    for i in range(len(atoms)):
        ai_residue_number, ai_name, ai_x = atoms[i][0], atoms[i][3], atoms[i][5]
        # add the only CA pruning edge
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

    columns = [
        "i",
        "j",
        "i_name",
        "j_name",
        "i_residue_number",
        "j_residue_number",
        "dij",
    ]
    df_prune = pd.DataFrame(edges, columns=columns)
    return df_prune


def read_xsol(fn_xsol):
    df = pd.read_csv(fn_xsol)
    df["x"] = df["x"].apply(
        lambda x: np.array(list(filter(None, x[1 : len(x) - 1].split(" ")))).astype(
            np.double
        )
    )

    return df


def get_semispace_sign(point, plane_points):
    """
    Calculate the signed distance from a point to a plane defined by three points.
    Parameters:
        point (numpy array): The point [x, y, z] we want to check.
        plane_points (numpy array): 3x3 array where each row is a point [x, y, z] defining the plane.
    Returns:
        float: The signed distance from the point to the plane.
    """
    # Calculate the normal vector of the plane
    normal_vector = np.cross(
        plane_points[1] - plane_points[0], plane_points[2] - plane_points[0]
    )

    if np.linalg.norm(normal_vector) == 0:
        raise ValueError("The plane points are collinear")

    # Calculate the signed distance from the point to the plane
    u = point - plane_points[0]  # in case of duplicate points, u will be zero
    semispace_sign = np.dot(normal_vector, u)

    return semispace_sign


def get_bit(df_xbsol: pd.DataFrame, row: pd.Series) -> int:
    i = row.name  # row original index
    if i < 3:
        return -1  # dummy value
    else:
        plane_points = df_xbsol.iloc[i - 3 : i]["x"].values
        semispace_sign = get_semispace_sign(row["x"], plane_points)
        # duplicate atoms will have bit 0, because the semispace sign is 0
        return int(semispace_sign > 0)


def get_prune_bsol(bsol: np.array, row: pd.Series) -> tuple:
    i = int(row["i"])
    j = int(row["j"])
    str_b = "".join([str(bit) for bit in bsol[i + 3 : j + 1]])
    return str_b


def process_instance(fn_segment: str, verbose: bool = False) -> None:
    # Read and process the instance
    try:
        df = read_instance(fn_segment)
    except ValueError as e:
        if verbose:
            print(f"Error processing {fn_segment}: {e}")
        return

    # create/save xbsol file as a csv
    atoms = extract_atoms_with_reorder(df)
    df_xbsol = pd.DataFrame(atoms, columns=df.columns)

    df_xbsol["b"] = df_xbsol.apply(lambda row: get_bit(df_xbsol, row), axis=1)

    fn_xbsol = os.path.join("xbsol", os.path.basename(fn_segment))
    df_xbsol.to_csv(fn_xbsol, index=False)

    # create prune edges file as a csv
    df_prune = extract_prune_edges(atoms)
    bsol = df_xbsol["b"].values
    df_prune["bsol"] = df_prune.apply(lambda row: get_prune_bsol(bsol, row), axis=1)

    fn_dmdgp = os.path.join("prune_bsol", os.path.basename(fn_segment))
    df_prune.to_csv(fn_dmdgp, index=False)


def test_process_instance():
    os.makedirs("segment", exist_ok=True)
    os.makedirs("xbsol", exist_ok=True)
    os.makedirs("prune_bsol", exist_ok=True)

    fn_segment = "segment/1a1u_model1_chainA_segment0.csv"
    # fn_segment = 'segment/1ah1_model1_chainA_segment0.csv'
    process_instance(fn_segment)


def main():
    # Ensure the dmdgp folder is created
    print("Creating prune_bsol directory...")
    os.makedirs("prune_bsol", exist_ok=True)

    print("Creating xbsol directory...")
    os.makedirs("xbsol", exist_ok=True)

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
