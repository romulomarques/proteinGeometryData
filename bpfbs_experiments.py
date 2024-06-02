import os
import time
import pickle
import pandas as pd
from tqdm import tqdm
from fbs.bpfbs import FBS, DFS, bp
from sklearn.model_selection import train_test_split

def load_states():
    # Load 'states' and 'p' (relative frequencies)
    fn_states = 'pickled_states.pkl'
    with open(fn_states, 'rb') as f:
        data = pickle.load(f)
        states, p = data

    df = {'states':[], 'p':[], 'type':[]}
    for s, p_ in zip(states, p):
        df['states'].append(s)
        df['p'].append(p_)
        # df['len'].append(len(s))
        df['type'].append(type(s).__name__)
    df = pd.DataFrame(df)

    print(df['type'].groupby(df['type']).count())

    count_states_with_len_5 = sum([len(s) == 5 for s in states])
    if count_states_with_len_5 != 32:
        raise ValueError(f'Expected 32 states with length 5, got {count_states_with_len_5}')
    return states, p

def process_file(fn):
    with open(os.path.join('ddgp', fn), 'rb') as f:
        D = pickle.load(f)

    fbs = FBS(D, states, p)
    dfs = DFS(D)

    statistic = {'fn': [], 'method':[], 'time_secs': []}

    statistic['fn'].append(fn)
    statistic['method'].append('dfs')
    try:
        tic = time.time()
        bp(D, dfs)
        toc = time.time() - tic
        statistic['time_secs'].append(toc)
    except:
        statistic['time_secs'].append(None)

    statistic['fn'].append(fn)
    statistic['method'].append('fbs')
    try:
        tic = time.time()
        bp(D, fbs)
        toc = time.time() - tic    
        statistic['time_secs'].append(toc)
    except:
        statistic['time_secs'].append(None)

    return statistic

if __name__ == '__main__':
    states, p = load_states()

    # Get list of files with their sizes
    files_with_sizes = [(fn, os.stat(os.path.join('ddgp', fn)).st_size) for fn in os.listdir('ddgp')]

    # Sort files by size
    sorted_files = [fn for fn, _ in sorted(files_with_sizes, key=lambda x: x[1])]

    statistic = {'fn': [], 'method':[], 'time_secs': []}
    for k, fn in tqdm(enumerate(sorted_files)):
        try:
            statistic_file = process_file(fn)
            for key in statistic_file.keys():
                statistic[key].extend(statistic_file[key])
        except:
            print(f'Error processing file {fn}')
        
        if k > 10:
            break

    statistic = pd.DataFrame(statistic)
    statistic.to_csv('df_times.csv', index=False)
