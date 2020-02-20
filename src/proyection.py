import numpy as np
import quadprog

T = 2
N = 2


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

x = np.hstack([
    np.eye(T),
    -np.eye(T)
])

tmp = np.vstack([
    tmp,
    np.hstack([x] * (N + 1)),
    -np.hstack([x] * (N + 1)),
])

        
A = tmp

def extract_player(n): return A[:, 2 * T * (n+1): 2 * T * (n + 2)].T

def build_player(n):
    aux = extract_player(n)
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

