import itertools
def powerset(l):
    N = len(l)
    return [x for n in range(1, N + 1) for x in itertools.combinations(l, n)]
