import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

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
    
    if norm_nv == 0:
        raise Exception("Normal vector is null!")

    # Calculate the signed distance
    signed_distance = np.dot(normal_vector, point - plane_points[0]) / norm_nv
    
    return signed_distance


def get_binary_solution(df):
    b_values = []
    for i in range(len(df)):
        # set the binaries of the first three atoms as 'None'
        if i < 3:
            # Not enough points to define a plane
            b_values.append(None)
            continue

        current_atom = df.iloc[i]
        antecessors = df.iloc[i-3:i]
        if current_atom["atom_number"] == antecessors.iloc[0]["atom_number"]:
            b_values.append(b_values[i-3])
        else:
            # plane_points = np.array(antecessors["x"].apply(lambda x: np.array(list(filter(None, x[1:len(x)-1].split(" ")))).astype(np.double)))
            plane_points = np.array(antecessors["x"])
            current_point = np.array(current_atom["x"])
            distance_to_plane = point_plane_distance(current_point, plane_points)
            b_values.append(int(distance_to_plane >= 0))
    return b_values


def read_xsol(fn_xsol):
    df = pd.read_csv(fn_xsol)
    df["x"] = df["x"].apply(lambda x: np.array(list(filter(None, x[1:len(x)-1].split(" ")))).astype(np.double))
    
    return df


def flip_if_needed(b):
    # Padronizando o quarto elemento como 1
    if len(b) > 3 and b[3] == 0:
        b = [int(1 - x) for x in b]
    return b


def create_slices(fnames):
    wd_bsol = "bsol"
    
    # establishes a numerical label for each kind of binary subsequence to be collected.
    constraint_types = [0, 1]
    # establishes the start and the end of a slice, considering the start is in a residue 'i'
    # and the end in the residue 'i+1'.
    start_atoms = ['HA', 'H']
    end_atoms = ['HA', 'H']
    slice_size = [6, 6]

    B = {x: [] for x in constraint_types}  # lists of b by their kind of constraint
    for fn_bsol in tqdm(fnames):
        df = pd.read_csv(os.path.join(wd_bsol, fn_bsol))
        n_df = len(df)
        b_full = [i for i in df["b"].to_list()]  # int values
        b_full[0] = 1
        b_full[1] = 1
        b_full[2] = 1
        b_full = [int(i) for i in b_full]
        b_full = flip_if_needed(b_full)

        for this_type in constraint_types:
            thisB = [] # list of b
            current_start = start_atoms[this_type]
            current_end = end_atoms[this_type]
            step = slice_size[this_type]

            # get all the binary subsequences from the current segment
            slice_start = 4
            while slice_start + step < n_df:
                # looking for the correct first atom of the slice
                if df.iloc[slice_start]["atom_name"] != current_start:
                    slice_start += 1
                    continue

                # checking the correctness of the slice type
                if df.iloc[slice_start + step]["atom_name"] != current_end:
                    raise Exception(f"Slice Error in {fn_bsol}: the slice does not end in {current_end} !")
                
                # add the first three fixed atoms and the first branching atom (the fourth one in the order).
                b = b_full[(slice_start - 4) : (slice_start + step)]  # int values

                # flip the binary subsequence around the fourth atom if it is necessary.
                b = flip_if_needed(b)
                b = b[4:]

                b_str = "".join([str(i) for i in b])
                thisB.append(b_str)

                slice_start += step
            
            B[this_type] += thisB
    
    # counting, for each constraint type, the frequency of each binary subsequency
    count = {x: [] for x in constraint_types}
    for this_type in constraint_types:
        B[this_type], count[this_type] = np.unique(B[this_type], return_counts=True) # it also converts lists to np.array

    # creating a dataframe column that indicates the constraint type of a binary subsequency
    constraint_column = []
    for this_type in constraint_types:
        constraint_column += len(B[this_type]) * [this_type]
    
    # joining in a single column the binary subsequencies of all constraint types.
    # we do the same action for the binary subsequencyy frequencies.
    b_column, count_column = [], []
    for this_type in constraint_types:
        b_column += list(B[this_type])
        count_column += list(count[this_type])
    
    df = pd.DataFrame({"bsol": b_column, "count": count_column, "constraint": constraint_column})

    df.sort_values(["constraint", "count"], inplace=True)
    df_total = df[["constraint", "count"]].groupby(["constraint"]).sum().reset_index()

    df["relfreq"] = df.apply(
        lambda x: x["count"]
        / df_total[df_total["constraint"] == x["constraint"]]["count"].values[0],
        axis=1,
    )

    print("   assertions")
    for constraint in df["constraint"].unique():
        # assert sum relfreq = 1
        assert np.isclose(df[df["constraint"] == constraint]["relfreq"].sum(), 1.0)

    # save the slices to csv
    fn = "df_dmdgp_slices.csv"
    print(f"   save slices to {fn}")
    df.to_csv(fn, index=False)


def test_create_slices():
    fn = "1a1u_model1_chainA_segment0.csv"
    create_slices([fn])


def process_instance(fn_xsol):
    try:
        df = read_xsol(fn_xsol)

        df["b"] = get_binary_solution(df)

        fn_bsol = os.path.join("bsol", os.path.basename(fn_xsol))
        df.to_csv(fn_bsol, index=False)
    except Exception as e:
        print(f"File {os.path.basename(fn_xsol)}: {e.args[0]}")

def test_process_instance():
    fn = "1a1u_model1_chainA_segment0.csv"
    fn_xsol = os.path.join("xsol/", fn)

    os.makedirs("bsol", exist_ok=False)
    
    process_instance(fn_xsol)


def main():
    # Creating directory to store the binary solutioon files
    os.makedirs("bsol", exist_ok=False)
    print("Creating 'bsol' directory...")

    # List all .csv files in the segment directory
    wd_xsol = 'xsol'
    fn_xsols = [fn for fn in os.listdir(wd_xsol) if fn.endswith(".csv")]
    print("Getting the files...")   
    
    # set the number of threads to be one less than the number of cores
    num_threads = os.cpu_count() - 1
    # Creating the Binary Solution files
    print("Creating the 'bsol' files...")
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        list(
            tqdm(
                executor.map(
                    process_instance, [os.path.join(wd_xsol, fn) for fn in fn_xsols]
                ), 
                total=len(fn_xsols),
            )
        )
    
    # for fn_xsol in tqdm([os.path.join(wd_xsol, fn) for fn in fn_xsols]):
    #     process_instance(fn_xsol)



if __name__ == "__main__":
    # test_process_instance()
    # test_create_slices()
    # results = main()
    
    fnames = os.listdir("bsol")
    create_slices(fnames)

