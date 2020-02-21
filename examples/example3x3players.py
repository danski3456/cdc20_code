import numpy as np
from src.player import Player
from src.game import Game

T = 3

p1 = Player(x=np.ones(3),
            sm = 1,
            s0 = 0,
            ram = 1,
            ec = 0.5,
            ed = 0.5)

p2 = Player(x=np.ones(3),
            sm = 1,
            s0 = 0,
            ram = 1,
            ec = 0.5,
            ed = 0.5)

p3 = Player(x=np.ones(3),
            sm = 1,
            s0 = 0,
            ram = 1,
            ec = 0.5,
            ed = 0.5)

player_list = [p1, p2, p3]

buying_price = np.ones(T) * 3
selling_price = np.ones(T) * 1

g = Game(player_list, buying_price, selling_price)

g.solve()
