import qdldl
import scipy.sparse as spa
import scipy.sparse.linalg as sla
from .utils import random_psd
import numpy as np
from multiprocessing.pool import ThreadPool

# Unit Test
import unittest
import numpy.testing as nptest
from time import time


class factors(unittest.TestCase):

    def setUp(self):
        np.random.seed(2)

    def test_basic_ls(self):
        np.random.seed(2)
        n = 5
        A = random_psd(n, n)
        B = random_psd(n, n)
        C = - random_psd(n, n)
        M = spa.bmat([[A, B.T], [B, C]], format='csc')

        #  import ipdb; ipdb.set_trace()
        m = qdldl.Solver(M)
        L, D, p = m.factors()
        L = L.toarray()
        nptest.assert_array_almost_equal(np.triu(L), np.zeros((2 * n, 2 * n)))
        assert (np.sort(p) == np.arange(2 * n)).all()

        P = np.zeros((2*n, 2*n))
        P[np.arange(2 * n), p] = 1

        L += np.eye(2 * n)
        # Assert close
        nptest.assert_array_almost_equal(P @ L @ np.diag(D) @ L.T @ P.T, M.toarray())
