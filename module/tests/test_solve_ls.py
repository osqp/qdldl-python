import qdldl
import scipy.sparse as spa
import scipy.sparse.linalg as sla
from .utils import random_psd
import numpy as np

# Unit Test
import unittest
import numpy.testing as nptest


class solve_ls(unittest.TestCase):

    def setUp(self):
        np.random.seed(1)

    def test_basic_ls(self):
        n = 5
        A = random_psd(n, n)
        B = random_psd(n, n)
        C = - random_psd(n, n)
        M = spa.bmat([[A, B.T], [B, C]], format='csc')
        b = np.random.randn(n + n)

        F = qdldl.factor(M)
        x_qdldl = F.solve(b)
        x_scipy = sla.spsolve(M, b)

        # Assert close
        nptest.assert_array_almost_equal(x_qdldl, x_scipy)

    def test_scalar_ls(self):
        M = spa.csc_matrix(np.random.randn(1, 1))
        b = np.random.randn(1)

        F = qdldl.factor(M)
        x_qdldl = F.solve(b)
        x_scipy = sla.spsolve(M, b)

        # Assert close
        nptest.assert_array_almost_equal(x_qdldl, x_scipy)
