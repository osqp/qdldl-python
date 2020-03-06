import numpy as np
import scipy.sparse as spa
from qdldl.tests.utils import random_psd
import qdldl


np.random.seed(2)
n = 5
A = random_psd(n, n)
B = random_psd(n, n)
C = - random_psd(n, n)
M = spa.bmat([[A, B.T], [B, C]], format='csc')
b = np.random.randn(n + n)

#  import ipdb; ipdb.set_trace()
m = qdldl.factor(M)

