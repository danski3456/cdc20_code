import quadprog
import numpy as np
from src.game import Game
from src.build import build_proyection_player, to_matrix_form
from src.proyection import proyect_into_linear
import time


def algorithm_init(g):

    PL = g._player_list
    N = g.N
    T = g.T
    n_var = 2 * T * N + 2 * T
    n_cons = 4 * T * N + 2 * T
    
    proy_params = []
    Cs, bs, mas = [], [], []
    for n in range(N):
        C_, b_, ma_ = build_proyection_player(n, g)
        Cs.append(C_)
        bs.append(b_)
        mas.append(ma_)

    grads = []
    for n, pl in enumerate(PL):
        gf = np.hstack([
            np.ones(T) * pl._sm - pl._s0,
            np.ones(T) * pl._s0,
            np.ones(2 * T) * pl._ram,
            np.ones(T) * pl._x,
            np.ones(T) * (- pl._x),
        ])
        grads.append(gf)

    return Cs, bs, mas, grads


def algorithm_main(game):

    Cs, bs, mas, grads = algorithm_init(game)
    mac = [x for x in mas[0] if x in mas[1]]
    
    N = game.N
    T = game.T
    nn_var = mas[0].shape[0]
    n_var = 2 * T * N + 2 * T
    n_cons = 4 * T * N + 2 * T
    n_iters = 6000

#    A = game.A.T.copy()
#    c = np.hstack([game.c, np.zeros(A.shape[1])])
#    A = np.vstack([A, np.eye(A.shape[1])])


    WS = np.zeros((N, n_cons))
    XS = np.zeros((N, n_cons))
    XS_ = np.zeros((N, n_cons))
    aph = game.alpha 
    NE = [list(game.G.neighbors(n)) for n in range(N)]

    for i in range(n_iters):
        for n in range(N):
            Cn = Cs[n]
            bn = bs[n] 
            man = mas[n] 

            tmp = np.zeros(n_cons)

            tmp = XS[n, :].copy()
            tmp[man] -= aph * grads[n]
            tmp -= aph * WS[n, :]

            for neig in NE[n]:
                tmp -= aph * (XS[n, :] - XS[neig, :])

            sec_copy = tmp.copy()
            sol = proyect_into_linear(tmp[man], Cn, bn)
            sec_copy[man] = sol[0]
            #sol = proyect_into_linear(tmp, A, c)
            #XS_[n, :] = sol[0]
            XS_[n, :] = sec_copy


        if i % 50 == 0:
            lll = np.diff(XS_.reshape(N, -1), axis=0).max().max()
            print(i, lll)
            if lll  < 1e-10:
                break
        XS = XS_.copy()
        for n in range(N):
            for neig in NE[n]:
                WS[n, :] += XS[n, :] - XS[neig, :]
    return [np.inner(grads[n], XS[n, mas[n]]) for n in range(N)] 
    
from src.game import generate_random_uniform
from src.multiproc import run_distributed

game = generate_random_uniform(4, 4, 'complete', 1234)

game.init()

_ = game.solve()
alg = algorithm_main(game)
core = game.get_payoff_core()
rd = run_distributed(game)
print(alg)
print(core)
