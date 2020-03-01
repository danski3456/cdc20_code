import numpy as np
import networkx as nx
from src.player import Player
from src.game import Game

T = 1
N = 2
G = nx.path_graph(N)


p2 = Player(x=np.array([4]),
            sm = 10,
            s0 = 0,
            ram = 3,
            ec = 1,
            ed = 1)

#p1 = Player(x=np.array([0, 4, 1]),
#            sm = 10,
#            s0 = 0,
#            ram = 1,
#            ec = 0.9,
#            ed = 0.9)


p3 = Player(x=np.array([-2]),
            sm = 10,
            s0 = 0,
            ram = 3,
            ec = 1,
            ed = 1)

player_list = [p2, p3]

buying_price = np.ones(T) * 3
selling_price = np.ones(T) * 1

g = Game(player_list, buying_price, selling_price, G)

m = g.solve()

### Extra

from scipy.optimize import linprog

g.init()

A_ = g.A.T
c_= g.b
b_ = g.c

s = linprog(c_, -A_, b_)

