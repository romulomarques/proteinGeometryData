import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


class nmr_t:
    def __init__(self, fnmr):
        self.fnmr = fnmr
        self.i = []
        self.j = []
        self.l = []
        self.u = []
        self.iAtomName = []
        self.jAtomName = []
        self.iResName = []
        self.jResName = []
        print('Reading ' + fnmr)
        with open(fnmr, 'r') as fid:
            for row in fid.readlines():
                row = row.split()
                self.i.append(int(row[0]))
                self.j.append(int(row[1]))
                self.l.append(float(row[2]))
                self.u.append(float(row[3]))
                self.iAtomName.append(row[4])
                self.jAtomName.append(row[5])
                self.iResName.append(row[6])
                self.jResName.append(row[7])
        self.nedges = len(self.i)
        self.nnodes = np.max((np.max(self.i), np.max(self.j)))
        fn_sol = fnmr.replace('.nmr', '.sol')
        self.x = np.zeros((self.nnodes, 3), dtype=float)
        if os.path.isfile(fn_sol):
            print('Reading ' + fn_sol)
            with open(fn_sol, 'r') as fid:
                for k, row in enumerate(fid.readlines()):
                    if 'x,y,z' in row:
                        continue  # skip header (first line)
                    row = row.split(',')
                    x, y, z = float(row[0]), float(row[1]), float(row[2])
                    self.x[k-1, :] = [x, y, z]

    def plot(self):
        print('Plotting')
        pid = self.fnmr.split(
            os.path.pathsep)[-1].split('.')[0]  # protein code

        plt.figure()
        plt.plot(self.i, self.j, 'o')
        plt.title('%s, nnodes:%d, nedges:%d' % (pid, self.nnodes, self.nedges))
        plt.savefig(self.fnmr.replace('.nmr', '_nmr.png'))
        plt.close()

        plt.title('%s, nnodes:%d, nedges:%d' % (pid, self.nnodes, self.nedges))
        plt.subplot(1, 2, 2)
        ax = plt.axes(projection='3d')
        ax.plot3D(self.x[:, 0], self.x[:, 1], self.x[:, 2], '-o')
        plt.savefig(self.fnmr.replace('.nmr', '_sol.png'))
        plt.close()

        plt.figure()
        accnedges = np.zeros(self.nnodes, dtype=int)
        for j in self.j:
            accnedges[j-1] += 1  # number of edges associated to node j
        # accumulating
        for i in range(1, len(accnedges)):
            accnedges[i] += accnedges[i-1]
        plt.plot(accnedges)
        plt.xlabel('node')
        plt.ylabel('acc number of edges')
        plt.savefig(self.fnmr.replace('.nmr', '_accnedges.png'))
        plt.close()

    def symmetryNodes(self):
        S = {i:True for i in range(4, self.nnodes + 1)}
        for k in range(self.nedges):
            i, j = self.i[k], self.j[k]
            for u in range(i + 4, j + 1):
                S[u] = False
        return [u for u in S if S[u]]


if __name__ == "__main__":
    for k, arg in enumerate(sys.argv):
        if arg == '-wdir':
            folder = sys.argv[k+1]        
    print('Reading folder ' + folder)
    FILES = []
    for fnmr in os.listdir(folder):        
        if fnmr.endswith('.nmr'):
            fpath = os.path.join(folder, fnmr)
            fsize = os.path.getsize(fpath)
            FILES.append({'fpath': fpath, 'fsize': fsize, 'pdb':fnmr.replace('.nmr', '')})
    FILES = sorted(FILES, key=lambda x: x['fsize'])

    df = {'pdb':[], 'nnodes':[], 'nedges':[], 'nsymm':[], 'edgeDensity':[]}
    for fdata in FILES:
        print('file: %s\t\t size(bytes): %d' % (fdata['fpath'], fdata['fsize']))
        nmr = nmr_t(fdata['fpath'])        
        edgeDensity = (nmr.nedges / (0.5 * nmr.nnodes * (nmr.nnodes - 1)))
        nSymm = len(nmr.symmetryNodes())        

        print('   nnodes ....... %d' % nmr.nnodes)
        print('   nedes ........ %d' % nmr.nedges)
        print('   nSymm ........ %d' % nSymm)
        print('   edgeDensity .. %g' % edgeDensity)

        df['pdb'].append(fdata['pdb'])
        df['nnodes'].append(nmr.nnodes)
        df['nedges'].append(nmr.nedges)
        df['nsymm'].append(nSymm)
        df['edgeDensity'].append(edgeDensity)
        nmr.plot()
    df = pd.DataFrame(df)
    fcsv = os.path.join(folder, 'stats.csv')
    print('Writing ' + fcsv)
    df.to_csv(fcsv, index=False)
