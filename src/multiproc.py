import ray
import time
from src.build import *
from src.proyection import proyect_into_linear
from examples.ex50 import g, N, T

ray.init()

@ray.remote
class Agent(object):
    
    def __init__(self, n, game, alpha):

        C, b, ma = build_proyection_player(n, game)
        self.n = n
        self.C = C
        self.b = b
        self.ma = ma
        self.N = game.N
        self.T = game.T
        self.alpha = alpha

        T = game.T
        pl = game._player_list[n]
        self.grad = np.hstack([
            np.ones(T) * pl._sm - pl._s0,
            np.ones(T) * pl._s0,
            np.ones(2 * T) * pl._ram,
            np.ones(T) * pl._x,
            np.ones(T) * (- pl._x),
        ])
        
    def update(self, xs, ws, xs_new):
        tmp = xs[self.n].copy()
        tmp[self.ma] -= self.alpha * self.grad
        tmp -= self.alpha * ws[self.n]
        for n_ in range(self.N):
            if n_ != self.n:
                tmp -= self.alpha * (xs[self.n] - xs[n_])

        new_x = tmp.copy()
        sol = proyect_into_linear(tmp[self.ma], self.C, self.b)
        new_x[self.ma] = sol[0]

        xs_new[self.n] = new_x 
        return new_x

    def update_w(self, xs, ws):
        n = self.n 
        for n_ in range(N):
            if n != n_:
                ws[n] += xs[n] - xs[n_]

    def update_old(self, xs_old, xs_new):
        xs_old[self.n] = xs_new[self.n]
         

ALPHA = (1 / (N + 5))
n_vars = 4 * T * N + 2 * T

agents = []
for n in range(N):
    ag = Agent.remote(n, g, ALPHA)
    agents.append(ag)
    

ws = [np.zeros(n_vars) for _ in range(N)]
xs = [np.zeros(n_vars) for _ in range(N)]
xs_new = [np.zeros(n_vars) for _ in range(N)]

xs_id = ray.put(xs)
ws_id = ray.put(ws)

for i in range(30):

    start = time.time()    
    fut = [ag.update.remote(xs_id, ws_id) for ag in agents]
    xs_ = ray.get(fut)
    d2 = np.hstack(xs_).copy()
    xs_id = ray.put(xs_)
    



    for n in range(N):
        for n_ in range(N):
            if n != n_:
                ws[n] += xs_[n] - xs_[n_]
    ws_id = ray.put(ws)
    d1 = d2
    print(i, time.time() - start, np.linalg.norm(d1 - d2))



        


