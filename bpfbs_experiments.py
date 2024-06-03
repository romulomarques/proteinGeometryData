import os
import time
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from fbs.bpfbs import FBS, DFS, bp
from sklearn.model_selection import train_test_split
from openpyxl.workbook import Workbook

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

    # # Sort files by size
    sorted_files = [fn for fn, _ in sorted(files_with_sizes, key=lambda x: x[1])]
    # sorted_sizes = [int(mysize) for _, mysize in sorted(files_with_sizes, key=lambda x: x[1])]

    
    # instâncias de tamanho nao maior que algo por volta de 50 vertices
    sorted_files_with_sizes = [(fn, mysize) for fn, mysize in sorted(files_with_sizes, key=lambda x: x[1])]
    last_index = 0
    for i in range(len(sorted_files_with_sizes)):
        if sorted_files_with_sizes[i][1] > 11000:
            last_index = i
            break    

    # tomando 800 instancias de tamanho menor ou igual a 50 vertices
    small_test_indexes = sorted(np.random.choice(last_index, 800, replace=False))
    sorted_small_test_files = [sorted_files[index] for index in small_test_indexes]

    statistic = {'fn': [], 'method':[], 'time_secs': []}
    # for k, fn in tqdm(enumerate(sorted_files)):
    for k, fn in tqdm(enumerate(sorted_small_test_files)):
        try:
            statistic_file = process_file(fn)
            for key in statistic_file.keys():
                statistic[key].extend(statistic_file[key])
        except:
            print(f'Error processing file {fn}')
        
        # if k > 10:
        #     break

    statistic = pd.DataFrame(statistic)
    methods = statistic['method'].unique()

    statistic.to_csv('df_times.csv', index=False)
    statistic.to_excel('df_times.xlsx', index=False)

    statistic_speedup = statistic.groupby(['fn']).agg({'method' : lambda x : list(x), 'time_secs' : lambda y : list(y)}).reset_index()
    for i in range(len(methods)):
        this_method = 'method_' + methods[i]
        statistic_speedup[this_method] = statistic_speedup['method'].apply(lambda x : x[i])
        this_time = 'time_' + methods[i]
        statistic_speedup[this_time] = statistic_speedup['time_secs'].apply(lambda x : x[i])
    statistic_speedup = statistic_speedup[['fn', 'method_dfs', 'method_fbs', 'time_dfs', 'time_fbs']]
    
    dtol=1e-6
    statistic_speedup['speed_up'] = 0.0
    statistic_speedup['speed_up'] = statistic_speedup.apply(lambda x : 1.0 if (float(x.loc['time_dfs']) == 0.0) & (float(x.loc['time_fbs']) == 0.0) else float(x.loc['speed_up']), axis=1)
    statistic_speedup['time_dfs'] = statistic_speedup.apply(lambda x : dtol if (float(x.loc['time_dfs']) == 0.0) & (float(x.loc['time_fbs']) != 0.0) else float(x.loc['time_dfs']), axis=1)
    statistic_speedup['time_fbs'] = statistic_speedup.apply(lambda x : dtol if (float(x.loc['time_dfs']) != 0.0) & (float(x.loc['time_fbs']) == 0.0) else float(x.loc['time_fbs']), axis=1)
    statistic_speedup['speed_up'] = statistic_speedup.apply(lambda x : None if x.loc['speed_up'] == None else x.loc['speed_up'], axis=1)
    
    statistic_speedup['speed_up'] = statistic_speedup.apply(lambda x : float(x.loc['time_dfs']) / float(x.loc['time_fbs']) if float(x.loc['speed_up'] == 0.0) else x.loc['speed_up'], axis=1)
    statistic_speedup.to_excel('df_speedup.xlsx', index=False)
    arroz = 1