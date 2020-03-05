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
        print('Processing graph {0} out of {1}'.format(i, reps))
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

### First round finding the core in the naive way

#for N in [6, 8, 10, 12, 14]:
#    try:
#        run_sim(N, 48, 'complete', 10)
#    except Exception as e:
#        print(e)
#

TYPES = ['complete', 'path', 'cycle', 'regular', 'wheel', 'tree', 'chordal']

for topo in TYPES:
    run_sim(7, 10, topo, 5, allcoal=False)

for topo in TYPES:
    run_sim(11, 10, topo, 5, allcoal=False)

for topo in TYPES:
    run_sim(17, 10, topo, 5, allcoal=False)

for topo in TYPES:
    run_sim(23, 10, topo, 5, allcoal=False)

for topo in TYPES:
    run_sim(29, 10, topo, 5, allcoal=False)

#for topo in TYPES:
#    run_sim(20, 10, topo, 5, allcoal=False)

### Testing different topologies
#for topo in TYPES:
#    run_sim(25, 10, topo, 2, allcoal=False)


            

