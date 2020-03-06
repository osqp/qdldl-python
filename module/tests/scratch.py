import numpy as np
import scipy.sparse as spa
import scipy.sparse.linalg as sla
from qdldl.tests.utils import random_psd
import qdldl
from time import time
from tqdm import tqdm


np.random.seed(2)
n = 100
N = 1000


t_qdldl_vec = []
t_scipy_vec = []

for i in tqdm(range(N)):
    d = 0.6
    A = random_psd(n, n, density=d)
    B = random_psd(n, n, density=d)
    C = - random_psd(n, n, density=d)
    M = spa.bmat([[A, B.T], [B, C]], format='csc')
    b = np.random.randn(n + n)

    t_qdldl = time()
    x_qdldl = qdldl.factor(M).solve(b)
    t_qdldl = time() - t_qdldl
    t_qdldl_vec.append(t_qdldl)

    t_scipy = time()
    x_scipy = sla.spsolve(M, b)
    t_scipy = time() - t_scipy
    t_scipy_vec.append(t_scipy)


print("Time qdldl %.4e" % np.mean(t_qdldl_vec))
print("Time scipy %.4e" % np.mean(t_scipy_vec))




