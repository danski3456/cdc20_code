import dill
import numpy as np
import networkx as nx
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')

import os
import sys
from pathlib import Path

from sim.constants import OUTDIR

outdir = Path.home() / OUTDIR

game_files = outdir.glob('*_game.pkl')
games = {}
for gf in game_files:
    with open(gf, 'rb') as fh: data = dill.load(fh)
    id_ = os.path.basename(gf).split('_')[0]
    games[id_] = data
    

dist_files = outdir.glob('*_dist.pkl')
dists = {}
for gf in dist_files:
    with open(gf, 'rb') as fh: data = dill.load(fh)
    id_ = os.path.basename(gf).split('_')[0]
    dists[id_] = data

data = []
for i, k in enumerate(games.keys()):
    g = games[k]
    if k in dists:
        d = dists[k]
        pc = g.get_payoff_core()
        di = d[0]
        distime = d[1].sum()
        A = nx.adj_matrix(g.G).A
        eg = sorted(np.linalg.eigvals(A).real, reverse=True) 
        eg2 = nx.linalg.spectrum.normalized_laplacian_spectrum(g.G)
        N, T, gt, alpha = g.N, g.T, g.graphtype, g.alpha
        sg = 1 / np.sqrt(eg2[1])
        sg2 = eg2[-1] - eg2[-2]
        if gt != 'expander':
            err = np.linalg.norm(pc - di)
            tup = (N, T, gt, distime, err, alpha, eg[0], eg[1], sg, sg2)
            if N in [5 ,7, 11, 13, 17, 19, 23, 29, 31]:
                data.append(tup)

df = pd.DataFrame(data)
df.columns = ['N', 'T', 'graph', 'time', 'error', 'alpha', 'eg1', 'eg2', 'sg',
'sg2']
df['gap'] = 1 / (df['eg1'] - df['eg2'])
df = df.sort_values(['N', 'graph'])

sgap = pd.pivot_table(df, index='N', columns='graph', values='sg')
sgap

sgap3 = pd.pivot_table(df, index='N', columns='graph', values='sg2')
sgap3

sgap2 = pd.pivot_table(df, index='N', columns='graph', values='gap')
sgap2

fig, ax = plt.subplots(figsize=(14, 10))
hueo = ['regular', 'chordal', 'cycle', 'complete', 'wheel', 'tree', 'path']
sns.barplot(data=df, x='N', y='time', hue='graph', ax=ax, hue_order=hueo)
ax.set_xlabel('Number of players')
ax.set_ylabel('Elapsed time (seconds)')
fig.show()
fig.savefig(outdir / 'compare_topo.pdf')


