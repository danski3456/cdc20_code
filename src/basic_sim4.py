import sys
import time
import os
import pickle
from src.newdist import main_dist
from src.game import *
from pathlib import Path

OUTDIR = 'Outputs/cdc_5'

params = [
    (5, 48, 'regular', 13, True),
    (7, 48, 'regular', 13, True),
    (9, 48, 'regular', 13, True),
    (11, 48, 'regular', 13, True),
    (17, 48, 'regular', 13, False),
    (30, 48, 'regular', 13, False),
    (50, 48, 'regular', 13, False),
    (70, 48, 'regular', 13, False),
    (90, 48, 'regular', 13, False),
]


for N, T, G, seed, VF in params:
    path_file = Path.home() / OUTDIR / '{}_{}_{}_{}.pkl'.format(N, T, G,seed)
    if os.path.isfile(path_file):
        print('File aready exits')
    else:
        start = time.time()
        g = generate_random_uniform(N, T, G, seed)
        g.init()
        x, gr, tim, niter = main_dist(g)
        end = time.time()
        costs = np.sum(x.mean(axis=0) * gr, axis=1)
        pc = g.get_payoff_core()
        if VF is True:
            _ = g.get_valfunc()
        data = [g, x, gr, end - start, tim.sum(), costs, pc, niter]
        with open(path_file, 'wb') as fh:
            pickle.dump(data, fh)
