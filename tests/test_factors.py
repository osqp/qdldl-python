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

        def check(matrix, m, verify_asymmetric_P=False):
            solver = qdldl.Solver(matrix)
            L, D, p = solver.factors()
            L = L.toarray()
            nptest.assert_array_almost_equal(np.triu(L), np.zeros((m, m)))
            assert (np.sort(p) == np.arange(m)).all()

            P = np.zeros((m, m))
            P[p, np.arange(m)] = 1

            if verify_asymmetric_P:
                assert np.linalg.norm(P - P.T) > 1

            L += np.eye(m)
            # Assert close
            nptest.assert_array_almost_equal(P @ L @ np.diag(D) @ L.T @ P.T,
                                             matrix.toarray())

        np.random.seed(2)
        n = 5
        A = random_psd(n, n)
        B = random_psd(n, n)
        C = - random_psd(n, n)
        M = spa.bmat([[A, B.T], [B, C]], format='csc')

        check(M, 2 * n)

        # Hardcode a matrix with an asymmetric P
        A = spa.csc_matrix(np.array([[ 1, 0, 500, 0, 1000],
                                    [ 0, 1, 0, 0, 1000],
                                    [ 500, 0, 500, 0, 1000],
                                    [ 0, 0, 0, 1., 1000],
                                    [1000, 1000, 1000, 1000, 5000]]))
        check(A, 5, True)
