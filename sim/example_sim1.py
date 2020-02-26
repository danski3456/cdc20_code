import numpy as np
import networkx as nx
from src.game import Game
from src.multiproc import run_distributed
from src.player import Player


## First simulation

T = 10
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

res_dis = run_distributed(g)
print('Distance between methods:  {0:0.4f}'.format(np.linalg.norm(res_dis[0] -
g.get_payoff_core())))
print('Number of iterations: ', np.argmin(res_dis[1]))
print('Total running time: {0:0.2f} s'.format(np.sum(res_dis[1])))

## Second simulation

T = 20
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

res_dis = run_distributed(g)
print('Distance between methods:  {0:0.4f}'.format(np.linalg.norm(res_dis[0] -
g.get_payoff_core())))
print('Number of iterations: ', np.argmin(res_dis[1]))
print('Total running time: {0:0.2f} s'.format(np.sum(res_dis[1])))


## Third simulation
 
T = 20
N = 10
G = nx.path_graph(N)

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

res_dis = run_distributed(g)
print('Distance between methods:  {0:0.4f}'.format(np.linalg.norm(res_dis[0] -
g.get_payoff_core())))
print('Number of iterations: ', np.argmin(res_dis[1]))
print('Total running time: {0:0.2f} s'.format(np.sum(res_dis[1])))

## Fourth simulation
 
T = 20
N = 10
G = nx.random_regular_graph(4, N)

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

res_dis = run_distributed(g)
print('Distance between methods:  {0:0.4f}'.format(np.linalg.norm(res_dis[0] -
g.get_payoff_core())))
print('Number of iterations: ', np.argmin(res_dis[1]))
print('Total running time: {0:0.2f} s'.format(np.sum(res_dis[1])))
