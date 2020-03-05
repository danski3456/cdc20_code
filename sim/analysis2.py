import dill
import numpy as np
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

for i, k in enumerate(games.keys()):
    g = games[k]
    if k in dists:
        d = dists[k]
        pc = g.get_payoff_core()
        di = d[0]
        err = np.linalg.norm(pc - di)
        if err > 1e-4:
            print(i, g.N, g.T, g.graphtype, d[2], (err /
            np.linalg.norm(pc)).round(4))
            print(di, pc)
            break
    else:
        print(i, g.N, g.T, g.graphtype)

data = []
for k in games:
    gm = games[k] 
    if k in dists:
        dis = dists[k]
    else:
        dis = (None, None)
    partial = gm + dis
    data.append(partial)

df = pd.DataFrame(data)
df.columns = ['game', 'N', 'T', 'centralized', 'valfunc', 'distributed', 'iterations']
df['graphtype'] = df.game.map(lambda x: x[:-1].split('_')[-1])
df = df.sort_values(['distributed', 'iterations'])
df =  df[~(df.graphtype == 'expander')]
df = df[df.N % 2 != 0]

df48 = df[df['T'] == 48].copy()
df48m = pd.melt(df48, id_vars=['game', 'N'], value_vars=['centralized', 'valfunc', 'distributed']) 

fig, ax = plt.subplots()
sns.lineplot(data=df48m, x='N', y='value', hue='variable', ax=ax)
ax.set_xlabel('Number of players')
ax.set_ylabel('Elapsed time (seconds)')
ax.legend(['Centralized algorithm', 'Naive core computation', 'Distributed algorithm'])
ax.set_title('Running time of the different methods to obtain a payoff in the core')
fig.show()
#fig.savefig(outdir / 'temp.pdf')


fig, ax = plt.subplots()
hueo = ['chordal', 'regular', 'cycle', 'complete', 'wheel', 'path', 'tree']
sns.barplot(data=df, x='N', y='distributed', hue='graphtype', ax=ax,
hue_order=hueo)
ax.set_xlabel('Number of players')
ax.set_ylabel('Elapsed time (seconds)')
fig.show()
