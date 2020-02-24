import ray
import numpy as np

ray.init()


@ray.remote
class Test(object):

    def __init__(self):
        pass

    def up(self, x, n):
        return x[n]

    def mod(self, x, n, y):
        x[n] = y



X = np.arange(10)
X_id = ray.put(X)

t1, t2 = Test.remote(), Test.remote()
fut = t1.up.remote(X_id, 3)
print(ray.get(fut))
fut2 = t1.mod.remote(X_id, 3, -2)
print(ray.get(fut2))
fut3 = t2.up.remote(X_id, 3)
print(ray.get(fut3))




