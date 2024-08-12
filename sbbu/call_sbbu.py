import os
import subprocess
import pandas as pd

def run_cmd(cmd):
    output = [cmd + '\n']
    print(cmd)
    try:
        if os.name == 'nt':  # windows
            cmd_out = subprocess.check_output(cmd, shell=True).decode('windows-1252')
        else:  # unix
            cmd_out = subprocess.check_output(cmd, shell=True).decode('utf-8')
        cmd_out = cmd_out.split('\n')
        for line in cmd_out:
            print(line)
            output.append(line + '\n')
    except subprocess.CalledProcessError as e:
        print(e)
    return output


def create_table(flog):
    with open(flog, 'r') as fid:
        data = fid.readlines()

    df = {'pid': [], 'nnodes': [], 'nedges': [], 'tsec': [], 'mde': [], 'lde': []}
    for row in data:
        if 'Reading file' in row:
            pid = row.split('/')[-1].split('.')[0]
            for col in df:
                df[col].append(None)
            df['pid'][-1] = pid
        if 'NMR: nnodes' in row:
            df['nnodes'][-1] = int(row.split()[-1])
        if 'NMR: nedges' in row:
            # edges (i, j) and (j, i) are included (duplicated)
            df['nedges'][-1] = int(int(row.split()[-1]) / 2)
        if 'MDE' in row and 'LDE' in row:
            df['mde'][-1] = float(row.split()[3].replace(',', ''))
            df['lde'][-1] = float(row.split()[-1])
        if 'solution found after' in row:
            df['tsec'][-1] = float(row.split()[-2])

    df = pd.DataFrame.from_dict(df)
    ftab = flog.replace('.log', '.csv')
    print('\nTABLE ' + ftab)
    print(df)
    df.to_csv(ftab)

if __name__ == "__main__":
    tmax = 300
    WDIR = ['DATA_EPSD_00_DMAX_50', 'DATA_EPSD_00_DMAX_60']
    solver = 'sbbu.exe' if os.name == 'nt' else './sbbu.exe'
    for wdir in WDIR:
        FILES = []
        for fname in os.listdir(wdir):
            fname = os.path.join(wdir, fname)
            if fname.endswith('.nmr'):
                FILES.append({'name': fname, 'size': os.path.getsize(fname)})
        FILES = sorted(FILES, key=lambda x: x['size'])

        output = []
        for k in range(len(FILES)):
            f = FILES[k]
            print('[%2d/%2d] %d : %s' % (k+1, len(FILES), f['size'], f['name']))
            cmd = '%s -nmr %s -tmax %f' % (solver, f['name'], tmax)
            output += run_cmd(cmd)

        # create log file
        flog = wdir + '.log'
        print('saving file ' + flog)
        with open(flog, 'w') as fid:
            for row in output:
                fid.write(row)

        # create table of results
        create_table(wdir + '.log')
