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

game_files = outdir.glob('*_game.pkl.proc')
games = {}
for gf in game_files:
    with open(gf, 'r') as fh: data = fh.read()
    name_, N, T, tsf, tgval = data[1:-1].split(',')
    N = int(N)
    T = int(T)
    tsf = float(tsf)
    tgval = float(tgval) if 'None' not in tgval else -1
    id_ = os.path.basename(gf).split('_')[0]
    par = (name_, N, T, tsf, tgval)
    games[id_] = par
    

dist_files = outdir.glob('*_dist.pkl.proc')
dists = {}
for gf in dist_files:
    with open(gf, 'r') as fh: data = fh.read()
    titer, niter = data[1:-1].split(',')
    titer = float(titer)
    niter = int(niter)
    id_ = os.path.basename(gf).split('_')[0]
    par = (titer, niter)
    dists[id_] = par


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
