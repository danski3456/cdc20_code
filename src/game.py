import numpy as np
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
        
        L = nx.laplacian_matrix(G).A
        ev = np.linalg.eigvals(L)
        self.alpha = 1 / (max(ev) * 1.1)
        
        A,b,c = to_matrix_form(self)
        self.A = A
        self.b = b
        self.c = c

    def solve(self):
        if self._model is None:
            res = solve_centralized(self._player_list, self._buying_price,
            self._selling_price)
            self._model = res[0]
            self._res = res
            self._payoff_core = extract_core_payment(self)
        return self._model

    def get_payoff_core(self):
        if self._payoff_core is None:
            self.solve()

        return self._payoff_core

    def get_valfunc(self):
        if self._valfunc is None:
            pl = self._player_list
            pb = self._buying_price
            ps = self._selling_price
            valfunc = {}
            L = range(self.N)
            for S in powerset(L):
                r = solve_centralized([pl[i] for i in S], pb, ps)
                valfunc[S] = r[0].objective.value()
            self._valfunc = valfunc

        return self._valfunc
                
            
