import numpy as np
import quadprog
from src.player import Player


def build_problem(players_list, pb, ps):

    N = len(players_list)
    T = players_list[0]._x.shape[0]

    ###### Build A
    ## User block

    tmp = np.hstack([
        np.tril(np.ones(T)),
        - np.tril(np.ones(T))
        ])
    tmp = np.vstack([tmp, -tmp])
    tmp = np.vstack([tmp, np.eye(2 * T)])

    ## Repeat user block for every player
    tmp = np.kron(
        np.eye(N), tmp
    )

    ## Add z block

    tmp = np.hstack([
        np.zeros((4 * T * N, 2 * T)), tmp
    ])

    ## Bottom rows

    print(T)
    last_row = np.hstack([
        np.hstack([ np.eye(T) * (1 / pl._ec), -np.eye(T) * pl._ed])
        for pl in players_list
    ])
    last_row = np.hstack([
        np.eye(T),
        -np.eye(T),
        last_row
    ])



    tmp = np.vstack([
        tmp,
        last_row,
        -last_row
    ])

        
    A = tmp

    ## Build B
    
    b = np.hstack([
        np.hstack([
            np.ones(T) * (pl._sm - pl._s0),
            np.ones(T) * pl._s0,
            np.ones(T) * pl._ram,
            np.ones(T) * pl._ram,
            ]) for pl in players_list])

    sum_load = sum(pl._x for pl in players_list)
    b = np.hstack([
        b,
        sum_load,
        -sum_load
    ])

    ## Build c

    c = np.hstack([
        -pb,
        -ps,
        np.zeros(2 * T * N)
    
    ])

    A[np.isclose(A, 0)] = 0
    b[np.isclose(b, 0)] = 0
    c[np.isclose(c, 0)] = 0


    return A, b, c



import pulp as plp
A, b, c = build_problem(player_list, buying_price, selling_price)

size = A.shape[1]

opt_model = plp.LpProblem(name="MIP Model")

x_vars = {i: plp.LpVariable(
    cat=plp.LpContinuous,
    lowBound=0,
    name=f"x{str(i)}"
) for i in range(size)}

constraints = {j : opt_model.addConstraint(
plp.LpConstraint(
             e=plp.lpSum( A[j, i] * x_vars[i] for i in range(size)),
             sense=plp.LpConstraintLE,
             rhs=b[j],
             name=f"constraint_{str(j)}"))
       for j in range(A.shape[0])}

objective = plp.lpSum(x_vars[i] * c[i] for i in range(size))

opt_model.sense = plp.LpMaximize
# for minimization
opt_model.setObjective(objective)

sol = opt_model.solve()










def extract_player(n): return A[:, 2 * T * (n+1): 2 * T * (n + 2)]

def build_player(n):
    aux = extract_player(n).T
    tmp = np.hstack([
        np.eye(T),
        - np.eye(T)
    ])
    tmp = np.vstack([
        tmp, -tmp 
    ])
    tmp = np.hstack([
        np.zeros((2 * T, 4 * T * N)),
        tmp
    ])
    B = np.vstack([tmp, aux])
    B = np.vstack([B, np.eye(4 * T * N + 2 * T)])
    return B
C = build_player(0).T


size = 4 * T * N + 2 * T
b = np.zeros(C.shape[1])
b[:2] = -3
b[2:4] = -1
G = np.eye(size)
a = np.random.rand(size)

sol = quadprog.solve_qp(G, a, C, b)

