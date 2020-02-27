import numpy as np
import networkx as nx
from src.player import Player
from src.game import Game

T = 48
N = 10
G = nx.complete_graph(N)

player_list = []
for n in range(N):
    p = Player(x=np.random.rand(T),
                sm = 10,
                s0 = 0,
                ram = 3,
                ec = 0.9,
                ed = 0.9)
    player_list.append(p)


buying_price = np.ones(T) * 3
selling_price = np.ones(T) * 1

g = Game(player_list, buying_price, selling_price, G)

m = g.solve()
