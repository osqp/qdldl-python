from scipy.sparse import random


def random_psd(m, n):
    M = random(m, n, density=0.8)
    return M.dot(M.T)


