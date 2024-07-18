import os
import pickle
import numpy as np
from fbs.algorithms import *
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
<<<<<<< HEAD

=======
>>>>>>> refs/remotes/origin/main

def read_dmdgp(fn: str) -> DDGP:
    """
    Read a DMDGP instance from a pickle file and convert it to a DDGP object.

    :param fn: str, path to the input pickle file
    :return: tuple(DataFrame, DDGP), the original dataframe and the DDGP object
    """
    with open(fn, "rb") as f:
        df = pickle.load(f)

    D = {}
    for _, row in df.iterrows():
        i, j, dij = row["i"], row["j"], row["dij"]
        i, j = min(i, j), max(i, j)
        if j not in D:
            D[j] = {}
        D[j][i] = dij

    parents = []
    for i in range(max(D.keys()) + 1):
        if i < 3:
            parents.append([])
        else:
            parents.append([i - 3, i - 2, i - 1])

    dmdgp = DDGP(D, parents)
    return df, dmdgp


def get_bsol(bsol: np.array, row) -> str:
    """
    Extract the binary solution string for a given row.

    :param bsol: np.array, the binary solution array
    :param row: pandas.Series, a row from the DataFrame
    :return: str, the binary solution string for the row
    """
    i, j = row["i"], row["j"]
    if np.abs(i - j) < 3:
        return ""
    row_bsol = "".join([str(x) for x in bsol[i : (j + 1)]])
    return row_bsol


<<<<<<< HEAD
def flip_bsol(bsol, repetitions: list):
    if (bsol[3] == 0):
        bsol[3:] = 1 - bsol[3:]
    for i, is_repetition in enumerate(repetitions):
        if is_repetition:
            bsol[i] = 0

    return bsol


def flip_bsol_by_symmetry_vertices(bsol: np.array, symmetries: list, repetitions: list):
    for i, is_symmetry in enumerate(symmetries):
        if i < 4:
            continue
        if is_symmetry and (bsol[i] == 1):
            bsol[i:] = 1 - bsol[i:]

    for j, is_repetition in enumerate(repetitions):
        if is_repetition:
            bsol[j] = 0
    
    return bsol


# row: row of the df_xbsol
def check_symmetry_vertex(df_dmdgp: pd.DataFrame, df_xbsol: pd.DataFrame, row) -> int:
    if row["is_repetition"]:
        return 0
    else:
        v = row.name
        vm1_atom_number = df_xbsol.iloc[v-1]["atom_number"]
        vm2_atom_number = df_xbsol.iloc[v-2]["atom_number"]
        vm3_atom_number = df_xbsol.iloc[v-3]["atom_number"]
        for _, row_dmdgp in df_dmdgp.iterrows():
            i, j = row_dmdgp["i"], row_dmdgp["j"]
            if i > j:
                aux = i
                i = j
                j = aux
            if (i < v-3) and (j >= v):
                isnot_i_vm1 = df_xbsol.iloc[i]["atom_number"] != vm1_atom_number
                isnot_i_vm2 = df_xbsol.iloc[i]["atom_number"] != vm2_atom_number
                isnot_i_vm3 = df_xbsol.iloc[i]["atom_number"] != vm3_atom_number
                if (isnot_i_vm1) and (isnot_i_vm2) and (isnot_i_vm3):
                    return 0
        
        return 1


def check_repeated_vertex(reorder: list, index: int) -> int:
    current_atom_number = reorder[index]
    for i in range(index-1, -1, -1):
        if reorder[i] == current_atom_number:
            return 1
    
    return 0


def determineX(D: DDGP, repetitions: list, T: list):
    # x = np.array([0, 0, 0] * len(T))
    x = init_x(D)
    T = flip_bsol(T, repetitions)
    for i in range(4, len(x)):
        calc_x(i, T[i], x, D)
        # ia, ib, ic = D.parents[i]
        # da, db, dc = [D.d[i][ia], D.d[i][ib], D.d[i][ic]]
        # A, B, C = x[ia], x[ib], x[ic]
        # p, w = solveEQ3(A, B, C, da, db, dc)
        # if norm(x[i] - (p - w)) < dtol:
        #     T[i] = 0
        # elif norm(x[i] - (p + w)) < dtol:
        #     T[i] = 1
        # else:
        #     print("Error creating T from x!")
        #     exit(0)
    return x


def process_instance(fn: str):
=======
def read_xbsol(fn_xbsol:str) -> np.array:
    """
    Read the xbsol array from a CSV file.

    :param fn_xbsol: str, path to the input CSV file
    :return: np.array, the xbsol array
    """
    df = pd.read_csv(fn_xbsol)
    bsol:np.ndarray = df['bsol'].values
    
    # flip if bsol[3] == 0
    if bsol[3] == 0:
        bsol = 1 - bsol
    
    return bsol

def process_instance(fn_dmdgp: str):
>>>>>>> refs/remotes/origin/main
    """
    Process a single DMDGP instance file.

    :param fn: str, path to the input pickle file
    """
    df, dmdgp = read_dmdgp(fn_dmdgp)
    dfs = DFS(dmdgp)
<<<<<<< HEAD

    df_xbsol = pd.read_csv(os.path.join("xbsol", os.path.basename(fn).replace(".pkl", ".csv")))
    reorder = list(df_xbsol["atom_number"])

    df_xbsol["is_repetition"] = df_xbsol.apply(lambda row: check_repeated_vertex(reorder, row.name), axis=1)
    df_xbsol["is_symmetry_vertex"] = df_xbsol.apply(lambda row: check_symmetry_vertex(df, df_xbsol, row), axis=1)
    
    x, _, _, last_vertex, total_time = bp(dmdgp, dfs)
    if last_vertex == dmdgp.n - 1:
        bsol = determineT(dmdgp, x)
        df["bsol"] = df.apply(lambda row: get_bsol(bsol, row), axis=1)

        # # df_xbsol = pd.read_csv(fn.replace("dmdgp", "xbsol").replace(".pkl", ".csv"))
        # mybsol = np.array(df_xbsol['b'])
        # mybsol[0], mybsol[1], mybsol[2] = 1, 1, 1
        # repetitions = list(df_xbsol["is_repetition"])
        # mybsol = flip_bsol(mybsol, repetitions)
        # mybsol = flip_bsol_by_symmetry_vertices(mybsol, list(df_xbsol["is_symmetry_vertex"]), repetitions)
        # my_x = determineX(dmdgp, repetitions, mybsol)
        # is_solution = False
        # is_solution = dmdgp.check_xsol(my_x)
        # is_same_solution = all(bsol == mybsol)
        # lala = 1
        # # print(f"{fn}: {is_same_solution}")
        
        # # with open(fn, "wb") as f:
        # #     pickle.dump(df, f)

        # # fn_csv = os.path.join(fn.replace(".pkl", ".csv"))
        # # df.to_csv(fn_csv, index=False)

        return True
    else:
        return False


def test_single():
    """
    Test the process_instance function on a single instance.
    """
    # fn = "dmdgp/1a1u_model1_chainA_segment0.pkl"
    fn = "dmdgp/1adz_model1_chainA_segment2.pkl"
    # fn = "dmdgp/1a11_model1_chainA_segment0.pkl"
    import time

    tic = time.time()
    solved = process_instance(fn)
    toc = time.time()
    print(f"Elapsed time: {toc - tic:.2f} seconds")

    hard_instances = []
    if not solved:
        hard_instances.append(fn)
    
    print(f"hard_instances: {hard_instances}")

=======
    
    fn_xbsol = fn_dmdgp.replace("dmdgp", "xbsol")
    xbsol = read_xbsol(fn_xbsol)
>>>>>>> refs/remotes/origin/main

def main():
    """
    Main function to parallelize DMDGP instance processing with a progress bar.
    """
    dmdgp_dir = "dmdgp"
    file_dmdgp = [
        os.path.join(dmdgp_dir, fn)
        for fn in os.listdir(dmdgp_dir)
        if fn.endswith(".pkl")
    ]

    # num_cores = mp.cpu_count()

<<<<<<< HEAD
    # with mp.Pool(processes=num_cores) as pool:
    #     list(
    #         tqdm(
    #             pool.imap(process_instance, file_list),
    #             total=len(file_list),
    #             desc="Processing files",
    #         )
    #     )
    
    hard_instances = []
    for fn in tqdm(file_list):
        solved = process_instance(fn)
        if not solved:
            hard_instances.append(fn)
    
    print(f"hard_instances: {len(hard_instances)}")
=======
    with mp.Pool(processes=num_cores) as pool:
        list(
            tqdm(
                pool.imap(process_instance, file_dmdgp),
                total=len(file_dmdgp),
                desc="Processing files",
            )
        )


def test_single():
    """
    Test the process_instance function on a single instance.
    """
    fn = "dmdgp/1a1u_model1_chainA_segment0.pkl"
    import time

    tic = time.time()
    process_instance(fn)
    toc = time.time()
    print(f"Elapsed time: {toc - tic:.2f} seconds")
>>>>>>> refs/remotes/origin/main


if __name__ == "__main__":
    # test_single()
    main()
