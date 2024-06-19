import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


# add, for each residue, a repetition of the CA atom at the end of the residue
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
    return edges


def read_xsol(fn_xsol):
    df = pd.read_csv(fn_xsol)
    df["x"] = df["x"].apply(lambda x: np.array(list(filter(None, x[1:len(x)-1].split(" ")))).astype(np.double))
    
    return df


def point_plane_distance(point, plane_points):
    """
    Calculate the signed distance from a point to a plane defined by three points.
    Parameters:
        point (numpy array): The point [x, y, z] we want to check.
        plane_points (numpy array): 3x3 array where each row is a point [x, y, z] defining the plane.
    Returns:
        float: The signed distance from the point to the plane.
    """
    # Calculate the normal vector of the plane
    normal_vector = np.cross(plane_points[1] - plane_points[0], plane_points[2] - plane_points[0])
    
    norm_nv = np.linalg.norm(normal_vector)
    
    # if norm_nv == 0:
    #     raise Exception("Normal vector is null!")

    # Calculate the signed distance
    signed_distance = np.dot(normal_vector, point - plane_points[0]) / norm_nv
    
    return signed_distance


# def get_binary_solution(df):
#     b_values = []
#     for i in range(len(df)):
#         # set the binaries of the first three atoms as 'None'
#         if i < 3:
#             # Not enough points to define a plane
#             b_values.append(None)
#             continue

#         current_atom = df.iloc[i]
#         antecessors = df.iloc[i-3:i]
#         if current_atom["atom_number"] == antecessors.iloc[0]["atom_number"]:
#             b_values.append(0)
#         else:
#             # plane_points = np.array(antecessors["x"].apply(lambda x: np.array(list(filter(None, x[1:len(x)-1].split(" ")))).astype(np.double)))
#             plane_points = np.array(antecessors["x"])
#             current_point = np.array(current_atom["x"])
#             distance_to_plane = point_plane_distance(current_point, plane_points)
#             b_values.append(int(distance_to_plane >= 0))
#     return b_values


def get_binary_bit(df_4v):
    # set the binaries of the first three atoms as 'None'
    if int(df_4v.iloc[-1]["order"]) < 3:
        # Not enough points to define a plane
        return None

    if df_4v.iloc[3]["atom_number"] == df_4v.iloc[0]["atom_number"]:
        return 0
    else:   
        plane_points = np.array(df_4v.iloc[0:3]["x"])
        current_point = np.array(df_4v.iloc[3]["x"])
        distance_to_plane = point_plane_distance(current_point, plane_points)
        return int(distance_to_plane >= 0)
 

def get_binary_solution(df_xsol):
    df_xsol["order"] = list(range(len(df_xsol)))
    b_values = [None] * len(df_xsol)
    for i in range(len(df_xsol)):
        j = np.max([0, i-3])
        b_values[i] = get_binary_bit(df_xsol.iloc[j:i+1])
    df_xsol.drop(columns=["order"], inplace=True)
    return b_values


### Parallelizing get_binary_solution()
# def process_row_xsol(i, df_xsol):
#     j = np.max([0, i - 3])
#     return get_binary_bit(df_xsol.iloc[j:i + 1])


# def get_binary_solution(df_xsol):
#     df_xsol["order"] = list(range(len(df_xsol)))
#     b_values = [None] * len(df_xsol)
#     with ProcessPoolExecutor() as executor:
#         # Create a list of futures
#         futures = [executor.submit(process_row_xsol, i, df_xsol) for i in range(len(df_xsol))]
        
#         # Collect results as they complete
#         for i, future in enumerate(futures):
#             b_values[i] = future.result()
    
#     df_xsol.drop(columns=["order"], inplace=True)
#     return b_values


def test_get_binary_solution():
    fn = "1a1u_model1_chainA_segment0.csv"
    fn_segment = os.path.join("segment/", fn)
    
    df_segment = read_instance(fn_segment)

    atoms = extract_atoms_with_reorder(df_segment)
    df_xsol = pd.DataFrame(atoms, columns=df_segment.columns)
    df_xsol["b"] = get_binary_solution(df_xsol)


def get_prune_b(row, bsol):
    i = int(row["i"])
    j = int(row["j"])
    str_b = "".join([str(bit) for bit in bsol[i+3:j+1]])
    constraint = str(row["i_name"]) + " " + str(j-i) + " " + str(row["j_name"])
    
    return str_b, constraint


def get_prune_bsol(df_prune, bsol):
    bsol_values = [None] * len(df_prune)
    constraint_values = [None] * len(df_prune)
    for i in range(len(df_prune)):
        bsol_values[i], constraint_values[i] = get_prune_b(df_prune.iloc[i], bsol)
    
    return bsol_values, constraint_values


### Parallelizing get_prune_bsol()
# def process_row_prune(i, df_prune, bsol):
#     return get_prune_b(df_prune.iloc[i], bsol)


# def get_prune_bsol(df_prune, bsol):
#     bsol_values = [None] * len(df_prune)
#     constraint_values = [None] * len(df_prune)
#     with ProcessPoolExecutor() as executor:
#         # Create a list of futures
#         futures = [executor.submit(process_row_prune, i, df_prune, bsol) for i in range(len(df_prune))]
        
#         # Collect results as they complete
#         for i, future in enumerate(futures):
#             bsol_values[i] = future.result()[0]
#             constraint_values[i] = future.result()[1]
    
#     return bsol_values, constraint_values


def test_get_prune_bsol():
    fn = "1a1u_model1_chainA_segment0.csv"
    fn_segment = os.path.join("segment/", fn)
    
    df_segment = read_instance(fn_segment)

    atoms = extract_atoms_with_reorder(df_segment)
    df_xsol = pd.DataFrame(atoms, columns=df_segment.columns)
    bsol = get_binary_solution(df_xsol)

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
    df_prune = pd.DataFrame(prune_edges, columns=columns)
    prune_bsol, constraint = get_prune_bsol(df_prune, bsol)
    df_prune["bsol"] = prune_bsol
    df_prune["constraint"] = constraint

    constraint_types = df_prune["constraint"].unique()

    print("arroz")


def process_instance(fn_segment):
    # Read and process the instance
    df = read_instance(fn_segment)

    # create/save atoms file as a csv
    atoms = extract_atoms_with_reorder(df)

    fn_xsol = os.path.join("xsol", os.path.basename(fn_segment))
    df_atoms = pd.DataFrame(atoms, columns=df.columns)
    bsol = get_binary_solution(df_atoms)
    df_atoms["b"] = bsol
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
    df_prune = pd.DataFrame(prune_edges, columns=columns)
    prune_bsol, constraint = get_prune_bsol(df_prune, bsol)
    df_prune["bsol"] = prune_bsol
    df_prune["constraint"] = constraint

    fn_dmdgp = os.path.join("prune_bsol", os.path.basename(fn_segment))
    df_prune.to_csv(fn_dmdgp, index=False)


def test_process_instance():
    os.makedirs("segment", exist_ok=True)
    os.makedirs("xsol", exist_ok=True)

    fn_segment = "segment/1a1u_model1_chainA_segment0.csv"
    process_instance(fn_segment)


def main():
    # Ensure the dmdgp folder is created
    print("Creating dmdgp directory...")
    os.makedirs("prune_bsol", exist_ok=True)

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

    # fns = [os.path.join("segment", fn) for fn in fn_segments]
    # for fn in tqdm(fns):
    #     process_instance(fn)


if __name__ == "__main__":
    # test_process_instance()
    main()
    # test_get_binary_solution()
    # test_get_prune_bsol()
