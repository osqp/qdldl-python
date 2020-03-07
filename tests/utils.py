from scipy.sparse import random


def random_psd(m, n, density=0.8):
    M = random(m, n, density=density)
    #  M = random(m, n, density=1.0)
    return M.dot(M.T).tocsc()


