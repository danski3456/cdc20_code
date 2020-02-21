import pulp as plp
import pandas as pd
import numpy as np

from src.player import Player

def solve_centralized(player_list, buying_price, selling_price):

    model = plp.LpProblem(name="model")

    set_N = range(len(player_list))
    set_T = range(len(player_list[0]._x))

    ## z^+_t
    zp_vars = {t:
        plp.LpVariable(
            cat=plp.LpContinuous,
            lowBound=0,
            upBound=None,
            name="zp_{0}".format(str(t))
        ) for t in set_T
    }

    ## z^-_t
    zn_vars = {t:
        plp.LpVariable(
            cat=plp.LpContinuous,
            lowBound=0,
            upBound=None,
            name="zn_{0}".format(str(t))
        ) for t in set_T
    }

    ## c^n_t

    ch_vars = {(n, t):
        plp.LpVariable(
            cat=plp.LpContinuous,
            lowBound=0,
            name="ch_{0}_{1}".format(str(n),str(t))
        ) for t in set_T for n in set_N
    }

    ## d^n_t

    dis_vars = {(n, t):
        plp.LpVariable(
            cat=plp.LpContinuous,
            lowBound=0,
            name="dis_{0}_{1}".format(str(n),str(t))
        ) for t in set_T for n in set_N
    }

    var = [zp_vars, zn_vars, ch_vars, dis_vars]


    #### Constraints

    ## Variable boudns

    cons_bnd_ub = {(n, t) :
    plp.LpConstraint(
                 e=ch_vars[(n, t)],
                 sense=plp.LpConstraintLE,
                 rhs=player_list[n]._ram,
                 name="cons_bnd_up_{0}_{1}".format(n, t))
           for t in set_T for n in set_N}

    for k in cons_bnd_ub: model.addConstraint(cons_bnd_ub[k])

    ## Battery lower bound
    cons_bnd_lb = {(n, t) :
    plp.LpConstraint(
                 e=dis_vars[(n, t)],
                 sense=plp.LpConstraintLE,
                 rhs=player_list[n]._ram,
                 name="cons_bnd_low_{0}_{1}".format(n, t))
           for t in set_T for n in set_N}

    for k in cons_bnd_lb: model.addConstraint(cons_bnd_lb[k])

    ## Battery upper bound
    cons_bat_ub = {(n, j) :
    plp.LpConstraint(
                 e=plp.lpSum(ch_vars[(n, t)] - dis_vars[(n, t)] for t in range(j +
                 1)),
                 sense=plp.LpConstraintLE,
                 rhs=player_list[n]._sm - player_list[n]._s0,
                 name="cons_bat_up_{0}_{1}".format(n, j))
           for j in set_T for n in set_N}

    for k in cons_bat_ub: model.addConstraint(cons_bat_ub[k])

    ## Battery lower bound
    cons_bat_lb = {(n, j) :
    plp.LpConstraint(
                 e=-plp.lpSum(ch_vars[(n, t)] - dis_vars[(n, t)] for t in range(j +
                 1)),
                 sense=plp.LpConstraintLE,
                 rhs=player_list[n]._s0,
                 name="cons_bat_low_{0}_{1}".format(n, j))
           for j in set_T for n in set_N}

    for k in cons_bat_lb: model.addConstraint(cons_bat_lb[k])

    ## Final equality constraint

    cons_z = {t: 
        plp.LpConstraint(
                     e=plp.lpSum(
                        zp_vars[t] - zn_vars[t]  
                        - plp.lpSum( ch_vars[(n,t)] * (1 / player_list[n]._ec) -
                        dis_vars[(n,t)] * player_list[n]._ed for n in set_N)
                     ),
                     sense=plp.LpConstraintLE,
                     rhs=sum(player_list[n]._x[t] for n in set_N),
                     name="cons_z_{0}".format(t))
                     for t in set_T}
    for k in cons_z: model.addConstraint(cons_z[k])

    cons_zo = {t: 
        plp.LpConstraint(
                     e=-plp.lpSum(
                        zp_vars[t] - zn_vars[t]  
                        - plp.lpSum( ch_vars[(n,t)] * (1 / player_list[n]._ec) -
                        dis_vars[(n,t)] * player_list[n]._ed for n in set_N)
                     ),
                     sense=plp.LpConstraintLE,
                     rhs=-sum(player_list[n]._x[t] for n in set_N),
                     name="cons_zo_{0}".format(t))
                     for t in set_T}
    for k in cons_zo: model.addConstraint(cons_zo[k])

    cons = [cons_bnd_ub, cons_bnd_lb, cons_bat_ub, cons_bat_lb, cons_z,
    cons_zo]

    objective = plp.lpSum(-zp_vars[t] * buying_price[t] + zn_vars[t] *
    selling_price[t] for t  in set_T)

    model.sense = plp.LpMaximize
    model.setObjective(objective)

    model.solve()

    opt_val = objective.value()

    df = pd.DataFrame(ch_vars.keys())
    df['ch'] = df.apply(lambda x: ch_vars[(x[0], x[1])].varValue, axis=1)
    df['dis'] = df.apply(lambda x: dis_vars[(x[0], x[1])].varValue, axis=1)
    df.columns = ['n', 't', 'ch', 'dis']
    df_cd = pd.pivot_table(df, index='n', columns='t', values=['ch', 'dis']).round(4)

    df = pd.DataFrame(zp_vars.keys())
    df['zp'] = df.apply(lambda x: zp_vars[x[0]].varValue, axis=1)
    df['zn'] = df.apply(lambda x: zn_vars[x[0]].varValue, axis=1)
    df.columns = ['t', 'zp', 'zn']
    df_z = df.copy()

    return model, var, cons, df_cd, df_z


def extract_core_payment(game):

    PL = game._player_list
    pb = game._buying_price
    ps = game._selling_price
    T = len(ps)
    N = len(PL)
    m = game.solve()
    
    cons = m.constraints
    
    payoff = np.zeros(N)
    for n, pl in enumerate(PL):
        pay = 0

        for t in range(T):
            du = cons[f"cons_bnd_up_{n}_{t}"].pi
            pay += du * pl._ram

            du = cons[f"cons_bnd_low_{n}_{t}"].pi
            pay += du * pl._ram

            du = cons[f"cons_bat_up_{n}_{t}"].pi
            pay += du * (pl._sm - pl._s0)

            du = cons[f"cons_bat_low_{n}_{t}"].pi
            pay += du * (pl._s0)

            du = cons[f"cons_z_{t}"].pi
            pay += du * pl._x[t]
            
            du = cons[f"cons_zo_{t}"].pi
            pay += - du * pl._x[t]
            
        payoff[n] = pay

    return payoff
            

def to_matrix_form(g):

    PL = g._player_list
    pb = g._buying_price
    ps = g._selling_price
    T = len(ps)
    N = len(PL)
    m = g.solve()
    cons = [v for lc in g._res[2] for (k, v) in lc.items()]
    n_var = len(m.variables())
    n_con = len(m.constraints)
    ck = [k for k in m.constraints]

    A = np.zeros((n_con, n_var))

    varn = [f'zp_{i}' for i in range(T)]
    varn += [f'zn_{i}' for i in range(T)]
    for n in range(N):
        varn += [f'ch_{n}_{t}' for t in range(T)]
        varn += [f'dis_{n}_{t}' for t in range(T)]

    b = np.zeros(n_con)
    for i, con_ in enumerate(cons):
        
        dcon = dict((k.name, v) for (k, v) in con_.items())
        tmp = [dcon.get(var, 0) for var in varn]
        A[i, :] = np.array(tmp) 
        b[i] = con_.getUb()

    c = np.hstack([
        -pb,
        -ps,
        np.zeros(2 * N * T)
    ])

    return A, b, c



def extract_player(n, A, T): return A[:, 2 * T * (n+1): 2 * T * (n + 2)]

def extract_common(A, T): return A[:, :2 * T]

def build_proyection_player(n, game):

    ps = game._selling_price
    pb = game._buying_price
    T = game.T
    N = game.N

    A, _, _ = to_matrix_form(game)

    pl_A = extract_player(n, A, T).T.copy()
    N_1 = pl_A.shape[0]
    cm_A = extract_common(A, T).T.copy()
    N_2 = cm_A.shape[0]

    tmp = np.vstack([pl_A, cm_A])
#
#    tmp = np.hstack([
#        np.eye(T),
#        - np.eye(T)
#    ])
#    tmp = np.vstack([
#        tmp, -tmp 
#    ])
#    tmp = np.hstack([
#        np.zeros((2 * T, 4 * T * N)),
#        tmp
#    ])
#    B = np.vstack([tmp, aux]) 
    A_pl = np.vstack([tmp, np.eye(4 * T * N + 2 * T)]) # variables are positive

    b = np.hstack([np.zeros(N_1), -pb, -ps, np.zeros(4 * T * N + 2 * T) ]) 
    G = np.eye(A.shape[0])
    return A_pl, b, G
        

