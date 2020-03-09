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

#params = pd.read_csv('sim/params1.csv')

OUTDIR = 'Outputs/cdc_5'

outdir = Path.home() / OUTDIR


rows = []
files = outdir.glob('*_13.pkl')
for fn in files:
    with open(fn, 'rb') as fh: data = pickle.load(fh)
    N, T, G, S = fn.name[:-4].split('_')
    g = data[0]
    time_dist = data[4]
    n_iter = data[-1]

    tup = (N, G, S, g, time_dist, n_iter)
    rows.append(tup)
#    print(data)
    
df = pd.DataFrame(rows)
df.columns = ['N', 'G', 'S', 'game', 'dist', 'iters']
df['cent'] = df.game.map(lambda x: x.time_solve_fast)
df['naive'] = df.game.map(lambda x: x.time_get_valfunc)
df['N'] = df['N'].astype(int)
df = df.sort_values('N')

melt = pd.melt(df, id_vars=['N'], value_vars=['dist', 'cent', 'naive'])

fig, ax = plt.subplots(figsize=(14, 10))
ax.plot(df.N, df.naive, marker='*')
ax.set_yscale('log')
ax.plot(df.N, df.cent, marker='*')
ax.plot(df.N, df.dist, marker='*')
ax.set_xlabel('Number of players')
ax.set_ylabel('Elapsed time (seconds)')
ax.legend(['Naive', 'Centralized', 'Distributed'])
fig.show()
fig.savefig(outdir / 'compalgs.pdf')

from string import Template

figure = Template("""
\\begin{tikzpicture}
\\begin{axis}[
ymode=log,
log ticks with fixed point,
height=4.5cm,
width=8cm,
xlabel=Number of players,
ylabel=Elapsed time (seconds),
ylabel near ticks,
cycle list name=color,
legend style={at={(0.5,-0.35)},
anchor=north,legend columns=-1},
]

\\addplot coordinates { $naive };
\\addplot coordinates { $cent };
\\addplot coordinates { $dist };

\legend{Naive, Centralized, Distributed};
\\end{axis}
\\end{tikzpicture}
""")

naive_ = ' '.join(map(str, [(x, y) for x, y in df[['N','naive']].dropna().values]))
cent_ = ' '.join(map(str, [(x, y) for x, y in df[['N','cent']].dropna().values]))
dist_ = ' '.join(map(str, [(x, y) for x, y in df[['N','dist']].dropna().values]))

t = figure.safe_substitute(
    naive=naive_,
    cent=cent_,
    dist=dist_
)
print(t)

with open(outdir / 'compalg.tex', 'w') as fh: fh.write(t)
