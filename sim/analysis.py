import dill
import numpy as np
import scipy as sp
import pandas as pd

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
for k in games:
   gm = games[k] 
   dis = dists[k]
   
   time_d = np.sum(dis[1])
   partial = (gm.__str__(),gm.N, gm.T, gm.time_solve_fast, gm.time_get_valfunc, )
   partial = partial + (time_d, dis[2])
   data.append(partial)

df = pd.DataFrame(data)
df.columns = ['game', 'N', 'T', 'centralized', 'valfunc', 'distributed', 'iterations']
