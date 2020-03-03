from src.game import Game, generate_random_uniform
from src.proyection import proyect_into_linear
from src.build import build_proyection_player
import numpy as np
import networkx as nx





    

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

    c_proy = np.hstack([g.c, np.zeros(M)]).copy()
    A_proy = np.vstack([g.A.T, np.eye(M)]).copy()

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

    ITERS = 2000
    #EXACT = np.array([-9 / np.sqrt(10), 3 / np.sqrt(10)])
    #A = get_ds_matrix(N)
    A = nx.adj_matrix(g.G).A

    xs = np.zeros((N, M))
    vs = np.zeros((N, M))
    ap = g.alpha

    for i in range(1, ITERS):

        #ap = alpha(i)
        new_xs = []
        for n in range(N):
            
            tmp = xs[n, :].copy()
            tmp -= ap * vs[n] 
            tmp -= ap * grad(vs[n], n, grads)

            tmp -= ap * np.dot(A[n], xs[n] - xs)
            #for neig in range(N):
            #    if neig != n:
            #        tmp -= ap * A[n, neig] * (xs[n, :] - xs[neig, :])

            if n != 0:
                pr = proyect_into_set(tmp, n, Cs[n], bs[n], mas[n])
            else:
                pr = proyect_into_set(tmp, n, A_proy, c_proy, np.arange(A_proy.shape[1]))
    #        assert 1 == 0
            new_xs.append(pr)

        xs_ = np.vstack(new_xs)
        for n in range(N):
            vs[n] += np.dot(A[n], xs_[n] - xs_)

        #for n in range(N):
        #    for neig in range(N):
        #        if n != neig:
        #            vs[n] += A[n, neig] * (xs[n, :] - xs[neig, :])

        #for n in range(N):
        #    vs[n] = np.zeros(M)
        #    for n2 in range(N):
        #        vs[n] += A[n, n2] * xs[n2]
        #vs = np.dot(A, xs)
        dis = np.linalg.norm(xs - xs_)
        if i % 50 == 0:
            print(i, dis)
        if dis < 1e-10:
            break
        xs = xs_
            
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


TEST = [
    generate_random_uniform(5, 4, 'complete', 666),
    generate_random_uniform(5, 4, 'complete', 12345),
    generate_random_uniform(5, 4, 'complete', 1),
    generate_random_uniform(5, 15, 'complete', 666),
    generate_random_uniform(5, 15, 'complete', 12345),
    generate_random_uniform(5, 15, 'complete', 1),
    generate_random_uniform(10, 4, 'complete', 12345),
    generate_random_uniform(10, 4, 'complete', 1),
    generate_random_uniform(10, 15, 'complete', 12345),
    generate_random_uniform(10, 15, 'complete', 1),
    generate_random_uniform(10, 4, 'complete', 666),
    generate_random_uniform(10, 15, 'complete', 666),
    generate_random_uniform(20, 15, 'complete', 2210),
    generate_random_uniform(20, 15, 'complete', 1312),
]
for t in TEST: t.init()

for i, g in enumerate(TEST):

    x, gr = main_dist(g)
    costs = np.sum(x * gr, axis=1)
    pc = g.get_payoff_core()
    print(i, 'Distance algorithm-core: ', np.linalg.norm(costs - pc))
    print('ASSERT CLOSE:', np.allclose(costs, pc, atol=1e-8))
