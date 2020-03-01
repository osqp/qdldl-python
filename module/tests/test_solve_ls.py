import qdldl
import scipy.sparse as spa
import scipy.sparse.linalg as sla
from .utils import random_psd
import numpy as np
from multiprocessing.pool import ThreadPool

# Unit Test
import unittest
import numpy.testing as nptest


class solve_ls(unittest.TestCase):

    def setUp(self):
        np.random.seed(2)

    def test_basic_ls(self):
        n = 5
        A = random_psd(n, n)
        B = random_psd(n, n)
        C = - random_psd(n, n)
        M = spa.bmat([[A, B.T], [B, C]], format='csc')
        b = np.random.randn(n + n)

        import ipdb; ipdb.set_trace()
        m = qdldl.factor(M)

        #  x_qdldl = m.solve(b)
        #  x_scipy = sla.spsolve(M, b)
        #
        #  # Assert close
        #  nptest.assert_array_almost_equal(x_qdldl, x_scipy)

    def test_scalar_ls(self):
        M = spa.csc_matrix(np.random.randn(1, 1))
        b = np.random.randn(1)

        F = qdldl.factor(M)
        x_qdldl = F.solve(b)
        x_scipy = sla.spsolve(M, b)

        # Assert close
        nptest.assert_array_almost_equal(x_qdldl, x_scipy)

    def test_thread(self):

        n = 10
        N = 10

        def get_random_ls(n):
            A = random_psd(n, n)
            B = random_psd(n, n)
            C = - random_psd(n, n)
            M = spa.bmat([[A, B.T], [B, C]], format='csc')
            b = np.random.randn(n + n)
            return M, b

        ls = [get_random_ls(n) for _ in range(N)]

        # Solve in loop with scipy
        res_scipy = []
        for (M, b) in ls:
            res_scipy.append(sla.spsolve(M, b))

        # Solve with threads
        def solve_qdldl(M, b):
            return qdldl.factor(M).solve(b)

        with ThreadPool(processes=2) as pool:
            res_qdldl = pool.starmap(solve_qdldl, ls)

        # Compare
        for i in range(N):
            nptest.assert_array_almost_equal(res_scipy[i],
                                             res_qdldl[i])

    #  def test_update(self):
    #      n = 5
    #      A = random_psd(n, n)
    #      B = random_psd(n, n)
    #      C = - random_psd(n, n)
    #      M = spa.bmat([[A, B.T], [B, C]], format='csc')
    #      b = np.random.randn(n + n)
    #
    #      F = qdldl.factor(M)
    #
    #      x_first_scipy = sla.spsolve(M, b)
    #      x_first_qdldl = F.solve(b)
    #
    #
    #      # Update
    #      M.data = M.data + 0.1 * np.random.randn(M.nnz)
    #      x_second_scipy = sla.spsolve(M, b)
    #
    #      #  F.update(spa.triu(M).data)
    #      #  x_second_qdldl = F.solve(b)
    #
    #      import ipdb; ipdb.set_trace()
    #
    #
    #
    #
