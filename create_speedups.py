import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

wd_dmdgp_HA9H = "dmdgp_HA9H"
wd_speedups = "speedups"
os.makedirs(wd_speedups, exist_ok=True)


def calculate_speed_up(row: pd.Series, col1_name: str, col2_name: str):
    a = row.loc[col1_name]
    b = row.loc[col2_name]
    
    if (a == 0.0) and (b == 0.0):
        return 1.0
    
    if (b == 0.0):
        b = 1e-7
    
    return a / b


def process_file(fn: str):
    # Read the CSV files
    df_dfs = pd.read_csv(os.path.join(wd_dmdgp_HA9H, fn + '_dfs.tmr'))
    df_fbs = pd.read_csv(os.path.join(wd_dmdgp_HA9H, fn + '_fbs.tmr'))

    # Rename columns
    df_dfs = df_dfs.rename(columns={'edge_time': 'edge_time_dfs', 'edge_niters': 'edge_niters_dfs'})
    df_fbs = df_fbs.rename(columns={'edge_time': 'edge_time_fbs', 'edge_niters': 'edge_niters_fbs'})

    # Concatenate DataFrames
    df_speedups = pd.concat([
        df_dfs[['i', 'j', 'code', 'edge_time_dfs', 'edge_niters_dfs']],
        df_fbs[['edge_time_fbs', 'edge_niters_fbs']]
    ], axis=1)

    # Calculate speedups
    df_speedups['speed_up'] = df_speedups.apply(lambda row: calculate_speed_up(row, 'edge_time_dfs', 'edge_time_fbs'), axis=1)
    df_speedups['speed_up_niters'] = df_speedups.apply(lambda row: calculate_speed_up(row, 'edge_niters_dfs', 'edge_niters_fbs'), axis=1)

    # Save the result to a CSV file
    df_speedups.to_csv(os.path.join(wd_speedups, fn + '.csv'), index=False)


def main():
    
    fnames_dfs = os.listdir(wd_dmdgp_HA9H)
    fnames_dfs = set(fn.replace('_dfs.tmr', '') for fn in fnames_dfs if fn.endswith('_dfs.tmr'))
    fnames_fbs = os.listdir(wd_dmdgp_HA9H)
    fnames_fbs = set(fn.replace('_fbs.tmr', '') for fn in fnames_fbs if fn.endswith('_fbs.tmr'))

    # getting the intersection of dfs and fbs solved problems
    common_fnames = list(fnames_dfs & fnames_fbs)
    
    # for fn in tqdm(common_fnames):
    #     process_file(fn)
    
    # Using ProcessPoolExecutor to parallelize the processing of files
    with ProcessPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(process_file, common_fnames),
                total=len(common_fnames),
            )
        )


if __name__ == "__main__":
    main()