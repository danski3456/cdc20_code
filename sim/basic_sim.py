from src.multiproc import run_distributed
from src.game import *
from sim.constants import OUTDIR

import os
import sys
import time
import dill
from pathlib import Path

outdir = os.path.expanduser('~/' + OUTDIR)
outdir_ = Path.home() / OUTDIR
outdir_.mkdir(parents=True, exist_ok = True)

outdir = str(outdir)

def run_sim(N, T, graph, reps, allcoal=True, distalg=True):
    start = time.time()
    for i in range(reps):
        g = generate_random_uniform(N, T, graph, i * 10)

        ex = game_exists(g, outdir)
        if not ex:
            if allcoal:
                g.get_valfunc()
            g.save(outdir)
        
        if distalg:
            dist_file = outdir + g.str_dig() + '_dist.pkl'
            if not os.path.isfile(dist_file):
                if g._model is None:
                    g.init()
                sol = run_distributed(g)
                with open(dist_file, 'wb') as fh:
                    dill.dump(sol, fh)
            else:
                print('Distributed Already exists')
            

    end = time.time()
    print('Elapsed time with {0} {1} {2}: {3:0.2f}'.format(N, T, graph, end - start))

### First round

run_sim(6, 4, 'wheel', 5)
run_sim(8, 4, 'wheel', 5)
run_sim(10, 4, 'wheel', 5)
run_sim(10, 4, 'complete', 5, allcoal=False)
run_sim(12, 4, 'complete', 5)

