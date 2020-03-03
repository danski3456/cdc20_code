from src.game import Game
import quadprog


def proyect_step(z, alpha, C, b):
    """
    
    C >= b
    """

#    print(z, alpha, C, b)
#    print(z.shape)
#    print(C.shape)
#    print(b.shape)
    G = np.eye(C.shape[1]) * 2 / alpha

    sol = quadprog.solve_qp(G, -z, C.T, b)
    return sol
    

def main(game):

    N = game.N
    T = game.T

    const = list(game._model.constraints)

    M = len(const)

    grads = np.zeros((N, M))

    for n, pl in enumerate(game._player_list):
        for i, c in enumerate(const):
            c_ = c.split('_')
            if len(c_) == 5:
                if int(c_[3]) == 0:
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

    XS = np.zeros((N, M))
    ZS = np.zeros((N, M))
    ZS_old = np.zeros((N, M))


    A = game.A.T.copy()
    c = np.hstack([game.c, np.zeros(M)])
    A = np.vstack([A, np.eye(M)])

    ITERS = 10

    for i in range(ITERS):

        alpha = 1 / np.sqrt((i + 1))
        for n in range(N):
            
            for neig in range(N):
                if neig != n:
                    ZS[n, :] += (1 / (N - 1)) * ZS_old[neig, :]
            ZS[n, :] += grads[n, :]

            pr = proyect_step(ZS[n, :].copy(), alpha, A, c)
            XS[n, :] = pr[0]

        ZS_old = ZS.copy()

        max_diff = np.diff(XS, axis=0).max()
        print(max_diff)
        if max_diff < 1e-8:
            break

    total = 0
    for n in range(N):
        cost = np.inner(grads[n, :], XS[n, :])
        total += cost
        print(n, cost)

    print(cost)



game = generate_random_uniform(5, 5, 'complete', 101)
game.init()
main(game)
game.get_payoff_core()

