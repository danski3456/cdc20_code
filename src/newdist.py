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


def iterate_player(n_list, xs, vs, grads, mas, ap, Cs, bs, A):
    prs = []
    for n in n_list:
        tmp = xs[n, :].copy()
        tmp -= ap * vs[n] 
        tmp -= ap * grads[n]

        tmp -= ap * np.dot(A[n], xs[n] - xs)
        #for neig in range(N):
        #    if neig != n:
        #        tmp -= ap * A[n, neig] * (xs[n, :] - xs[neig, :])

    #            if n != 0:
        pr = proyect_into_set(tmp, n, Cs, bs, mas[n])
        prs.append((n, pr))

    return prs

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

    #Cs = Cs[0]
    #bs = bs[0]
    C = Cs[0]
    n, m = C.shape
    b = bs[0]




    #c_proy = np.hstack([g.c, np.zeros(M)]).copy()
    #A_proy = np.vstack([g.A.T, np.eye(M)]).copy()

    P = sparse.csc_matrix(np.eye(m))
    q = -np.ones(m) * 1.0
    A = sparse.csc_matrix(C)
    l = b * 1.0
    u = np.ones(n) * np.inf
    prob = osqp.OSQP()
    _ = prob.setup(P, q, A, l, u=u, alpha=1.0, verbose=False, eps_abs=1e-12)

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

    ITERS = 5000
    A = nx.adj_matrix(g.G).A

    xs = np.zeros((N, M))
    vs = np.zeros((N, M))
    ap = g.alpha

    #return 0, 0

    for i in range(1, ITERS):
        start_iter = time.time()
        new_xs = []

        for n in range(N):
            
            tmp = xs[n, :].copy()
            tmp -= ap * vs[n] 
            tmp -= ap * grad(vs[n], n, grads)

            tmp -= ap * np.dot(A[n], xs[n] - xs)

#            if n != 0:
            #start_ = time.time()
            #pr = proyect_into_set(tmp, n, Cs, bs, mas[n])
            tmp_x = tmp.copy()
            tmp_ = tmp[mas[n]]
            prob.update(q=-tmp_)
            res = prob.solve()
            np.put(tmp_x, mas[n], res.x)
            #end_ = time.time()
            #print('Pryection time', round((end_ - start_),5))
#            else:
#                pr = proyect_into_set(tmp, n, A_proy, c_proy, np.arange(A_proy.shape[1]))
    #        assert 1 == 0
            new_xs.append(tmp_x)

        
        xs_ = np.vstack(new_xs)
        for n in range(N):
            vs[n] += np.dot(A[n], xs_[n] - xs_)


#        costs = np.sum(xs * grads, axis=1)
#        costs_ = np.sum(xs_ * grads, axis=1)
        dis = np.linalg.norm(xs - xs_)
        #if i % 50 == 0:
        #    print(i, dis)
        if dis < 1e-5:
            print('Exit, ', i)
            break
        xs = xs_
        end_iter = time.time()
#        print('End iter', round(end_iter - start_iter, 4))
            
#    print('Round exited', i)
    return xs, grads

        #diff = np.diff(np.vstack(xs).reshape(N, -1), axis=0)
        #mdiff = diff.max()
        #if i % 100 == 0:
        #    print(i, mdiff)
        #if mdiff < 1e-10:
        #    return xs, grads
            #break
    #print(i, xs[0] - EXACT, xs[1] - EXACT)
    #print('-' * 20)

if __name__ == '__main__':
    import sys
    import time
    import pickle

    if len(sys.argv) < 5:
        sys.exit()
    N = int(sys.argv[1])
    T = int(sys.argv[2])
    G = sys.argv[3].strip()
    seed = int(sys.argv[4])
    print(N, T, G, seed)

    start = time.time()
    g = generate_random_uniform(N, T, G, seed)
    g.init()
    x, gr = main_dist(g)
    end = time.time()
    print('Elapsed time', round(end - start, 4))
    costs = np.sum(x * gr, axis=1)
    pc = g.get_payoff_core()
    print('Dis payoff', costs.round(2))
    print('True payoff', pc.round(2))
    print('Distance algorithm-core: ', np.linalg.norm(costs - pc))
    print('Distance algorithm-core rel: ', np.linalg.norm(costs - pc) /
    np.linalg.norm(costs))
    print('ASSERT CLOSE:', np.allclose(costs, pc, atol=1e-8))
    data = [g, x, gr, end - start, costs, pc]
    with open('/home/infres/dkiedanski/' + '{}_{}_{}_{}.pkl'.format(N, T, G,
    seed), 'wb') as fh:
        pickle.dump(data, fh)

#TEST = [
#    generate_random_uniform(5, 4, 'complete', 666),
#    generate_random_uniform(5, 4, 'complete', 12345),
#    generate_random_uniform(5, 4, 'complete', 1),
#    generate_random_uniform(5, 15, 'complete', 666),
#    generate_random_uniform(5, 15, 'complete', 12345),
#    generate_random_uniform(5, 15, 'complete', 1),
#    generate_random_uniform(10, 4, 'complete', 12345),
#    generate_random_uniform(10, 4, 'complete', 1),
#    generate_random_uniform(10, 15, 'complete', 12345),
#    generate_random_uniform(10, 15, 'complete', 1),
#    generate_random_uniform(10, 4, 'complete', 666),
#    generate_random_uniform(10, 15, 'complete', 666),
#    generate_random_uniform(20, 15, 'complete', 2210),
#    generate_random_uniform(20, 15, 'complete', 1312),
#]
#for t in TEST: t.init()
#
#for i, g in enumerate(TEST):
#
#    x, gr = main_dist(g)
#    costs = np.sum(x * gr, axis=1)
#    pc = g.get_payoff_core()
#    print(i, 'Distance algorithm-core: ', np.linalg.norm(costs - pc))
#    print('ASSERT CLOSE:', np.allclose(costs, pc, atol=1e-8))
