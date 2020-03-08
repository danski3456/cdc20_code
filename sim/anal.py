import pickle
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

#from sim.constants import OUTDIR

params = pd.read_csv('sim/params1.csv')

OUTDIR = 'Outputs/cdc6'

outdir = Path.home() / OUTDIR

params['game'] = None
params['time_dist'] = 0
params['iters'] = 0
for i, r in params.iterrows():
    fn = '{}_{}_{}_{}.pkl'.format(r.N, r['T'], r.G.strip(), r.S)
    try:
        with open(outdir / fn, 'rb') as fh: data = pickle.load(fh)
        params.loc[i, 'game'] = data[0]
        params.loc[i, 'time_dist'] = data[4]
        params.loc[i, 'iters'] = data[-1]
    except:
        pass

params = params.dropna()

params['G'] = params['G'].map(str.strip)
params = params.sort_values(['N', 'time_dist'])
params['time_cent'] = params.game.map(lambda x: x.time_solve_fast)


hueo = params[params.N == 23].groupby('G').time_dist.mean().sort_values().index.values
fig, ax = plt.subplots(figsize=(14, 10))
sns.barplot(data=params, x='N', y='time_dist', hue='G', ax=ax, hue_order=hueo)
ax.set_xlabel('Number of players')
ax.set_ylabel('Elapsed time (seconds)')
fig.show()
fig.savefig(outdir / 'elapsec.pdf')



fig, ax = plt.subplots(figsize=(14, 10))
sns.barplot(data=params, x='N', y='iters', hue='G', ax=ax)
ax.set_xlabel('Number of players')
ax.set_ylabel('Number of iterations before convergence')
fig.show()
fig.savefig(outdir / 'niters.pdf')

