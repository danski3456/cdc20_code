import numpy as np
from src.game import * 
from src.build import build_proyection_player

N = 2
T = 2
g = generate_random_uniform(5, 5, 'regular', 1)
g.init()

C, d, _ = build_proyection_player(0, g)
n, m = C.shape

x = np.random.uniform(-2, 2, m)

C_ = C[: n - m, :]
d_ = d[: n - m]

n = n - m

x_ = np.hstack([x, np.zeros(n)])

C = np.hstack([C_, - np.eye(n)])

A = np.eye(n + m)
for i in range(m, n + m): A[i, i] = 0

A_ = 2.0 * np.dot(A.T, A)

X = np.vstack([
    np.hstack([A_, C.T]),
    np.hstack([C, np.zeros((n, n))])
])

if np.linalg.cond(X) < 1/sys.float_info.epsilon:
    I = np.linalg.inv(X)


p_1 = np.zeros(n + m)
q_1 = np.zeros(n + m)
x_1 = x_.copy()

y_1 = np.clip(x_1 + p_1, a_min=0, a_max=None).copy()
p_2 = (p_1 + x_1 - y_1).copy()

tt_1 = 2.0 * np.dot(A.T, y_1 + q_1)
B_1 = np.hstack([tt_1,  d_])
x_2 = np.dot(I, B_1)[: n + m] #[: n + m].copy()
q_2 = q_1 + y_1 - x_2


y_2 = np.clip(x_2 + p_2, a_min=0, a_max=None).copy()
p_3 = (p_2 + x_2 - y_2).copy()

tt_2 = 2.0 * np.dot(A.T, y_2 + q_2)
B_2 = np.hstack([tt_2,  d_])
x_3 = np.dot(I, B_2)[: n + m] #[: n + m].copy()
q_3 = q_2 + y_2 - x_3
