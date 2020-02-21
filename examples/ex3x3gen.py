import numpy as np
from src.player import Player
from src.game import Game

T = 3

p4 = Player(x=np.array([0, 1, 0]),
            sm = 10,
            s0 = 0,
            ram = 1,
            ec = 0.9,
            ed = 0.9)

p2 = Player(x=np.array([0, 4, 1]),
            sm = 10,
            s0 = 0,
            ram = 1,
            ec = 0.9,
            ed = 0.9)

p1 = Player(x=np.array([0, 4, 1]),
            sm = 10,
            s0 = 0,
            ram = 1,
            ec = 0.9,
            ed = 0.9)


p3 = Player(x=np.array([-2,0, 0]),
            sm = 10,
            s0 = 0,
            ram = 1,
            ec = 0.9,
            ed = 0.9)

player_list = [p1, p2, p3, p4]

buying_price = np.ones(T) * 3
selling_price = np.ones(T) * 1

g = Game(player_list, buying_price, selling_price)

m = g.solve()
