from src.game import Game, generate_random_uniform
from src.proyection import proyect_into_linear
from src.build import build_proyection_player
from functools import partial
import numpy as np
import networkx as nx
import osqp
import time
from scipy import sparse





    

def proyect_into_set(x, n, C_, b_, ma_):
### Proyect into random polyedron

    #A_ = np.array([
    #    [1.0, 1.0],
    #    [-4.0, 1.0],
    #    [-6.0, 4.0]
    #])
    #b_ = np.array([2.0, 2.0, 1.0])
    y = x.copy()
    sol = proyect_into_linear(x[ma_], C_, b_)
    np.put(y, ma_, sol[0])
    return y

## Project into ball of radius 3
#    L = np.linalg.norm(x)
#    if L <= 3:
#        return x
#    else: #        return x / L * 3

def grad(x, n, grads_):
    
    return grads_[n]

### Simple functions
#    if n == 0:
#        g = np.array([2, 1])
#    elif n == 1:
#        g = np.array([1, -2])
#
#    return g

def alpha(k):
    return 1 / N


def get_grads(g):

    N, T = g.N, g.T
    cons = list(g._model.constraints)
    M = len(cons)
    grads = np.zeros((N, M))
    for n in range(N):
        pl = g._player_list[n]
        for i, c in enumerate(cons):
            c_ = c.split('_')
            if len(c_) == 5:
                if int(c_[3]) == n:
                    if 'cons_bat_up' in c:
                        grads[n, i] = pl._sm - pl._s0
                    elif 'cons_bat_low' in c:
                        grads[n, i] = pl._s0
                    elif 'cons_bnd_up' in c:
                        grads[n, i] = pl._ram
                    elif 'cons_bnd_low' in c:
                        grads[n, i] = pl._ram
            else:
                if 'cons_z_' in c:
                    grads[n, i] = pl._x[int(c_[-1])]
                elif 'cons_zo_' in c:
                    grads[n, i] = -pl._x[int(c_[-1])]

    assert np.allclose(g.b, grads.sum(axis=0))
    return grads

def main_dist(g):

    N = g.N
    T = g.T
    cons = list(g._model.constraints)
    M = len(cons)

    Cs, bs, mas = [], [], []
    for n in range(N):
        C_, b_, ma_ = build_proyection_player(n, g)
        Cs.append(C_)
        bs.append(b_)
        mas.append(ma_)

    C = Cs[0]
    n, m = C.shape
    b = bs[0]

    grads = get_grads(g)

    P = sparse.csc_matrix(np.eye(m))
    q = -np.ones(m) * 1.0
    A = sparse.csc_matrix(C)
    l = b * 1.0
    u = np.ones(n) * np.inf
    PROYECTIONS = [osqp.OSQP() for _ in range(N)]
    for p in PROYECTIONS:
        _ = p.setup(P, q, A, l, u=u, alpha=1.0, verbose=False, eps_abs=1e-12)


    ITERS = 10000
    itertimes = np.zeros(ITERS)
    A = nx.adj_matrix(g.G).A

    xs = np.zeros((N, M))
    vs = np.zeros((N, M))
    ap = g.alpha
    NE = [list(g.G.neighbors(n)) for n in range(N)]

    for i in range(1, ITERS):
        start_iter = time.clock()
        new_xs = []

        for n in range(N):
            
            tmp = xs[n, :].copy()
            tmp -= ap * vs[n] 
            tmp -= ap * grad(vs[n], n, grads)

            #tmp -= ap * np.dot(A[n], xs[n] - xs)
            for neig in NE[n]:
                tmp -= ap * A[n, neig] * (xs[n, :] - xs[neig, :])

#            if n != 0:
            #start_ = time.time()
            tmp_x = tmp.copy()
            tmp_ = tmp[mas[n]]
            #pr = proyect_into_set(tmp, n, C, b, mas[n])
            PROYECTIONS[n].update(q=-tmp_)
            res = PROYECTIONS[n].solve()
            np.put(tmp_x, mas[n], res.x)
            new_xs.append(tmp_x)
            #new_xs.append(pr)

        
        xs_ = np.vstack(new_xs)
        #for n in range(N):
        #    vs[n] += np.dot(A[n], xs_[n] - xs_)

        for n in range(N):
            for neig in NE[n]:
                vs[n, :] += A[n, neig] * (xs_[n, :] - xs_[neig, :])

        
        dis = np.linalg.norm(xs_ - xs, axis=1).max()
        if dis < 1e-7:
            print('Exit, ', i)
            break
        xs = xs_
        end_iter = time.clock()
        itertimes[i] = end_iter - start_iter
            
    return xs, grads, itertimes, i
