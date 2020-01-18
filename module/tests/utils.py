from scipy.sparse import random


def random_psd(m, n):
    M = random(m, n, density=0.6)
    #  M = random(m, n, density=1.0)
    return M.dot(M.T)


