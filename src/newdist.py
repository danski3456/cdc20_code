from src.game import Game, generate_random_uniform
from src.proyection import proyect_into_linear
from src.build import build_proyection_player
import numpy as np


g = generate_random_uniform(5, 10, 'complete', 666)
g.init()

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
#c_proy = np.hstack([g.c, np.zeros(M)]).copy()
#A_proy = np.vstack([g.A.T, np.eye(M)]).copy()

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


def get_ds_matrix(N):

    A = np.ones((N, N))
    return A / N
    

def proyect_into_set(x, n):



### Proyect into random polyedron

    #A_ = np.array([
    #    [1.0, 1.0],
    #    [-4.0, 1.0],
    #    [-6.0, 4.0]
    #])
    #b_ = np.array([2.0, 2.0, 1.0])
    print('Start')
    tmp = x.copy()
    print(tmp)
    man = mas[n]
    sol = proyect_into_linear(x[man], Cs[n], bs[n])
    print(sol[0])
    tmp[man] = sol[0]
    print(tmp)
    print('End')
    return tmp

## Project into ball of radius 3
#    L = np.linalg.norm(x)
#    if L <= 3:
#        return x
#    else:
#        return x / L * 3

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
    if k > 0:
        return 1 / k
    else:
        return 1



#N = 2
#T = 2
ITERS = 2000
#EXACT = np.array([-9 / np.sqrt(10), 3 / np.sqrt(10)])
A = get_ds_matrix(N)

#xs = [np.zeros(M) for _ in range(N)]
xs = np.zeros((N, M))
vs = np.zeros((N, M))
#vs = [np.zeros(M) for _ in range(N)]

for i in range(1, ITERS):

    ap = alpha(i)
    for n in range(N):
        
        tmp = vs[n] - ap * grad(vs[n], n, grads)
        pr = proyect_into_set(tmp, n)
        assert 1 == 0
        xs[n] = pr

    #for n in range(N):
    #    vs[n] = np.zeros(M)
    #    for n2 in range(N):
    #        vs[n] += A[n, n2] * xs[n2]
    vs = np.dot(A, xs)

    diff = np.diff(np.vstack(xs).reshape(N, -1), axis=0)
    mdiff = diff.max()
    print(i, mdiff)
    #print(i, xs[0] - EXACT, xs[1] - EXACT)
    #print('-' * 20)

costs = np.array([np.inner(grads[n, :], xs[n]) for n in range(N)])

print('Alg', costs)
print('Core', g.get_payoff_core())
