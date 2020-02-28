import numpy as np
import time
import hashlib
import dill
import os
import sys
from pathlib import Path
import networkx as nx
from src.player import Player
from src.utils import powerset
from src.build import solve_centralized, extract_core_payment, to_matrix_form


class Game(object):

    """This is a base class for the coopearative game"""

    def __init__(self, player_list, buying_price, selling_price, G):
        """TODO: to be defined.

        :player_list: TODO
        :buying_price: TODO
        :selling_price: TODO

        """
        self._player_list = player_list
        self.N = len(player_list)
        self._buying_price = buying_price
        self.T = len(buying_price)
        self._selling_price = selling_price
        self._model = None
        self._payoff_core = None
        self._res = None
        self._valfunc = None

        self.G = G
        self.time_solve_fast = None
        self.time_get_valfunc = None
        
        L = nx.laplacian_matrix(G).A
        ev = np.linalg.eigvals(L)
        self.alpha = 1 / (np.absolute(ev).max() * 1.1)
        
    def init(self):
        A,b,c = to_matrix_form(self)
        self.A = A
        self.b = b
        self.c = c


    def solve(self):
        if self._model is None:
            start_ = time.time()
            res = solve_centralized(self._player_list, self._buying_price,
            self._selling_price)
            self._model = res[0]
            self._res = res
            self._payoff_core = extract_core_payment(self)
            end_ = time.time()
            self.time_solve_fast = end_ - start_
        return self._model

    def get_payoff_core(self):
        if self._payoff_core is None:
            self.solve()

        return self._payoff_core

    def get_valfunc(self):
        if self._valfunc is None:
            start_ = time.time()
            pl = self._player_list
            pb = self._buying_price
            ps = self._selling_price
            valfunc = {}
            L = range(self.N)
            for S in powerset(L):
                r = solve_centralized([pl[i] for i in S], pb, ps)
                valfunc[S] = r[0].objective.value()
            self._valfunc = valfunc
            end_ = time.time()
            self.time_get_valfunc = end_ - start_

        return self._valfunc

    def __str__(self):
        
        string = '_'.join(map(str,
            [self.N, self.T, self.seed, self.graphtype] 
        ))
        return string

    def str_dig(self):

        m = hashlib.md5()
        m.update(self.__str__().encode('utf-8'))
        string = m.hexdigest()
        return string

    def save(self, outdir):
        string = self.str_dig() +  '_game.pkl'
        Path(outdir).mkdir(parents=True, exist_ok = True)
        with open(outdir + string, 'wb') as fh:
            dill.dump(self, fh)

def game_exists(game, directory):


    id_ = game.str_dig() + '_game.pkl'
    if os.path.isfile(directory + id_ ):
        print('File already exists')
        return True
    else:
        game.init()
        return False
                
            
def generate_random_uniform(N, T, G_method, seed=1234):

    rng = np.random.RandomState(seed)
    N_ = 4 if N >= 5 else N - 1
    switcher = {
        'complete': nx.complete_graph(N),
        'path': nx.path_graph(N),
        'cycle': nx.cycle_graph(N),
        'regular': nx.random_regular_graph(N_, N, seed=rng),
        'wheel': nx.wheel_graph(N),
    }
    G = switcher.get(G_method)

    player_list = []
    for n in range(N):
        p = Player(x=rng.uniform(3, -3, T),
                    sm = 13.5,
                    s0 = 0,
                    ram = 5,
                    ec = 0.9,
                    ed = 0.9)
        player_list.append(p)


    buying_price = np.ones(T) * 3.0
    selling_price = np.ones(T) * 1.0

    game = Game(player_list, buying_price, selling_price, G)
    game.graphtype = G_method
    game.seed = seed
    return game


