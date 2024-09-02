import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

dmdgp_HA9H = "dmdgp_HA9H"
if not os.path.exists(dmdgp_HA9H):
    os.makedirs(dmdgp_HA9H)


class edge_t:
    def __init__(self, i, j, type, order):
        self.i = i
        self.j = j
        self.type = type
        self.order = order

    def __str__(self):
        return f"({self.i}, {self.j}, {self.type}, {self.order})"


def read_dmdgp(fn_pkl: str) -> list:
    with open(fn_pkl, "rb") as f:
        dmdgp = pickle.load(f)
    # swap i and j if i > j
    edges = []
    cols_to_swap = ["name", "residue_number"]
    for _, edge in dmdgp.iterrows():
        if edge["i"] > edge["j"]:
            edge["i"], edge["j"] = edge["j"], edge["i"]
            for col in cols_to_swap:
                edge[f"i_{col}"], edge[f"j_{col}"] = edge[f"j_{col}"], edge[f"i_{col}"]
        edges.append(edge)

    # add column 'type' f"{i_name}{j-i}{j_name}"
    for edge in edges:
        i_name = edge["i_name"]
        j_name = edge["j_name"]
        edge["type"] = f"{i_name}{edge['j'] - edge['i']}{j_name}"

    # convert the edges to dataframe
    dmdgp = pd.DataFrame(edges)
    return dmdgp


def sort_prune_edges(pruned_edges: list, verbose: bool) -> list:
    pruned_edges = sorted(pruned_edges, key=lambda x: (x.j, -x.i))

    if verbose:
        print("Sorted edges:")
        for edge in pruned_edges:
            print(f"({edge.i}, {edge.j}, {edge.type})")

    for i in range(len(pruned_edges)):
        # edge = {'i': int, 'j': int, 'type': 'HA9H', 'order': 0}
        edge_i: edge_t = pruned_edges[i]

        if edge_i.type not in ["HA9H", "HA6HA"]:
            continue

        # j from i-1 to 0
        iloc = i
        for j in range(i - 1, -1, -1):
            edge_j: edge_t = pruned_edges[j]
            if edge_j.j < edge_i.i + 3:
                break
            iloc = j

        if verbose:
            print(f"before:")
            for edge in pruned_edges[iloc : i + 1]:
                print(f"({edge.i}, {edge.j}, {edge.type})")

        pruned_edges[iloc + 1 : (i + 1)] = pruned_edges[iloc:i]
        pruned_edges[iloc] = edge_i

        if verbose:
            print(f"after:")
            for edge in pruned_edges[iloc : i + 1]:
                print(f"({edge.i}, {edge.j}, {edge.type})")

    return pruned_edges


def split_prune_and_discrete_edges(dmdgp: pd.DataFrame) -> list:
    # keep only rows where j > i + 3
    is_prune_edge = dmdgp["j"] > dmdgp["i"] + 3
    prune_edges = [edge for _, edge in dmdgp[is_prune_edge].iterrows()]
    discrete_edges = [edge for _, edge in dmdgp[~is_prune_edge].iterrows()]
    return prune_edges, discrete_edges


def set_edges_code(edge_type: str) -> int:
    if edge_type == "HA9H":
        return 0
    elif edge_type == "HA6HA":
        return 1
    elif edge_type == "C4CA":
        return 2
    return -1


def save_dmdg_order(fn_pkl: str, discrete_edges: list, prune_edges: list) -> list:
    df = pd.DataFrame(discrete_edges + prune_edges).reset_index(drop=True)
    df["order"] = df.index

    df["code"] = df["type"].apply(set_edges_code)

    cols = ["i", "j", "i_name", "j_name", "dij", "order", "code"]
    df = df[cols]

    # export to CSV
    fn_csv = fn_pkl.replace(".pkl", ".csv")
    df[cols].to_csv(fn_csv, index=False)


def process_instance(fn_pkl):
    dmdgp = read_dmdgp(fn_pkl)
    prune_edges, discrete_edges = split_prune_and_discrete_edges(dmdgp)
    prune_edges = sort_prune_edges(prune_edges, verbose=False)

    # Modify the filename for saving
    fn_pkl_new = fn_pkl.replace("dmdgp", "dmdgp_HA9H")
    save_dmdg_order(fn_pkl_new, discrete_edges, prune_edges)


def main():
    dmdgp_directory = "dmdgp"
    dmdgph_paths = []
    for fn in os.listdir(dmdgp_directory):
        if fn.endswith(".pkl"):
            dmdgph_paths.append(os.path.join(dmdgp_directory, fn))

    # Create a ThreadPoolExecutor
    with ProcessPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(process_instance, dmdgph_paths),
                total=len(dmdgph_paths),
            )
        )


if __name__ == "__main__":
    # fn_pkl = "dmdgp/1wj5_model1_chainA_segment5.pkl"
    # process_file(fn_pkl)

    main()
