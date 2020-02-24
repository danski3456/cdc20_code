import quadprog
import numpy as np
from src.game import Game
from src.build import build_proyection_player, to_matrix_form
from src.proyection import proyect_into_linear
import time

def build_L_Kn(N, n_vars):
    
    L = np.diag(np.ones(N))
    L -= np.ones((N, N)) / N
    L = np.kron(L, np.eye(n_vars))
    return L

    

def algorithm_init(g):

    PL = g._player_list
    N = g.N
    T = g.T
    n_var = 2 * T * N + 2 * T
    n_cons = 4 * T * N + 2 * T
    
    proy_params = []
    A, _, _ = to_matrix_form(g)
    for n in range(N):
        proy = build_proyection_player(n, A, g)
        proy_params.append(proy)

    x_ini = []
    for n, pl in enumerate(PL):
        x = np.zeros(n_cons)
        #for t in range(T):
        #    if pl._x[t] >= 0:
        #        x[t] = pl._x[t]
        #    else:
        #        x[T + t] = pl._x[t]
        x_ini.append(x)

    
    grads = []
    for n, pl in enumerate(PL):
#        gf = np.zeros(n_cons)
#        
#        gf[T * n: T * (n + 1)] = pl._sm - pl._s0
#        gf[T * N + T * n: T * N + T * (n + 1)] = pl._s0
#        gf[2 * T * N + T * n: 2 * T * N + T * (n + 1)] = pl._ram
#        gf[3 * T * N + T * n: 3 * T * N + T * (n + 1)] = pl._ram
#        gf[-2 * T: - T] = pl._x
#        gf[-T: ] = - pl._x
#
        gf = np.hstack([
            np.ones(T) * pl._sm - pl._s0,
            np.ones(T) * pl._s0,
            np.ones(2 * T) * pl._ram,
            np.ones(T) * pl._x,
            np.ones(T) * (- pl._x),
        ])
        grads.append(gf)

    return proy_params, x_ini, grads


def algorithm_main(game):

    proy, xini, grads = algorithm_init(game)
    N = game.N
    T = game.T
    n_vars = len(xini[0])

    #L = build_L_Kn(N, n_vars)
    #A, b, c = to_matrix_form(game)
    #n_r = A.shape[0]
    #A_ = np.vstack([A.T, np.eye(n_r)])
    #c_ = np.hstack([c, np.zeros(n_r)])

    #ws = L.dot(np.random.uniform(0, 3, L.shape[0]))
    #ws = [ws[i : i + n_vars] for i in range(0, L.shape[0], n_vars)]
    ws = [np.zeros(n_vars) for _ in range(N)]
    xs = [x for x in xini]
    n_iters = 1000
    aph = 1/ (N + 3)

    for i in range(n_iters):
#        start = time.time() 
        new_xs = []
        for n in range(N):
#            start_n = time.time()
            #G = proy[n][2]
            C = proy[n][0]
            b = proy[n][1]
            ma = proy[n][2]

            #tmp = (sum(xs) / N).copy()
            #tmp -= (1 / (i + 5)) * grads[n]

            tmp = xs[n].copy()
            tmp[ma] -= aph * grads[n]
            tmp -= aph * ws[n]
            for n2 in range(N):
                if n2 != n:
                    tmp -= aph * (xs[n] - xs[n2])
            
            new_x = tmp.copy()
            sol = proyect_into_linear(tmp[ma], C, b)
            new_x[ma] = sol[0]
            #sqp = quadprog.solve_qp(G, tmp, C.T, b, 0)
            #new_xs.append(sol[0].copy())
            new_xs.append(new_x)
#            print(f'End {n}: ', time.time() - start_n)
        for n in range(N):
            for n2 in range(N):
                if n != n2:
                    ws[n] += new_xs[n] - new_xs[n2]

        d1 = np.hstack(xs)
        d2 = np.hstack(new_xs)
        
        if i % 50 == 0:
            dis = np.linalg.norm(d1 - d2)
            if dis < 1e-10:
                break
            print(i, np.linalg.norm(d1 - d2))
            #print(i, [sum(w) for w in ws])
        xs = new_xs[:]
#        print('End iteration :', time.time() - start)

    for n in range(N): print(n, np.inner(grads[n], xs[n][proy[n][2]]))
    return [np.inner(grads[n], xs[n][proy[n][2]]) for n in range(N)]
