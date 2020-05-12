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


class solve_ls(unittest.TestCase):

    def setUp(self):
        np.random.seed(2)

    def test_basic_ls(self):
        np.random.seed(2)
        n = 5
        A = random_psd(n, n)
        B = random_psd(n, n)
        C = - random_psd(n, n)
        M = spa.bmat([[A, B.T], [B, C]], format='csc')
        b = np.random.randn(n + n)

        #  import ipdb; ipdb.set_trace()
        m = qdldl.Solver(M)

        x_qdldl = m.solve(b)
        x_scipy = sla.spsolve(M, b)

        # Assert close
        nptest.assert_array_almost_equal(x_qdldl, x_scipy)

    def test_scalar_ls(self):
        M = spa.csc_matrix(np.random.randn(1, 1))
        b = np.random.randn(1)

        F = qdldl.Solver(M)
        x_qdldl = F.solve(b)
        x_scipy = sla.spsolve(M, b)

        # Assert close
        nptest.assert_array_almost_equal(x_qdldl, x_scipy)

    def test_thread(self):

        n = 100
        N = 400

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

        def solve_qdldl(M, b):
            return qdldl.Solver(M).solve(b)

        # Solve with qdldl serial
        t_serial = time()
        res_qdldl_serial = []
        for (M, b) in ls:
            res_qdldl_serial.append(solve_qdldl(M, b))
        t_serial = time() - t_serial


        # Solve with threads
        t_thread = time()
        with ThreadPool(processes=2) as pool:
            res_qdldl_thread = pool.starmap(solve_qdldl, ls)
        t_thread = time() - t_thread

        # Compare
        for i in range(N):
            nptest.assert_allclose(res_scipy[i],
                                   res_qdldl_thread[i],
                                   rtol=1e-05,
                                   atol=1e-05)

        #  print("Time serial %.4e s" % t_serial)
        #  print("Time thread %.4e s" % t_thread)

    def test_update(self):
        n = 5
        A = random_psd(n, n)
        B = random_psd(n, n)
        C = - random_psd(n, n)
        M = spa.bmat([[A, B.T], [B, C]], format='csc')

        b = np.random.randn(n + n)

        F = qdldl.Solver(M)

        x_first_scipy = sla.spsolve(M, b)
        x_first_qdldl = F.solve(b)

        # Update
        M.data = M.data + 0.1 * np.random.randn(M.nnz)
        # Symmetrize matrix
        M =.5 * (M + M.T)
        x_second_scipy = sla.spsolve(M, b)

        x_second_qdldl_scratch = qdldl.Solver(M).solve(b)

        #  M_triu = spa.triu(M, format='csc')
        #  M_triu.sort_indices()
        #  F.update(M_triu.data)
        F.update(M)
        x_second_qdldl = F.solve(b)

        nptest.assert_allclose(x_second_scipy,
                               x_second_qdldl,
                               rtol=1e-05,
                               atol=1e-05)

    def test_upper(self):
        np.random.seed(2)
        n = 5
        A = random_psd(n, n)
        B = random_psd(n, n)
        C = - random_psd(n, n)
        M = spa.bmat([[A, B.T], [B, C]], format='csc')
        b = np.random.randn(n + n)

        #  import ipdb; ipdb.set_trace()
        m = qdldl.Solver(M)
        x_qdldl = m.solve(b)


        M_triu = spa.triu(M, format='csc')
        m_triu = qdldl.Solver(M_triu, upper=True)
        x_qdldl_triu = m_triu.solve(b)


        nptest.assert_allclose(x_qdldl,
                               x_qdldl_triu,
                               rtol=1e-05,
                               atol=1e-05)


    def test_update_upper(self):
        n = 5
        A = random_psd(n, n)
        B = random_psd(n, n)
        C = - random_psd(n, n)
        M = spa.bmat([[A, B.T], [B, C]], format='csc')
        b = np.random.randn(n + n)

        F = qdldl.Solver(M)
        F_upper = qdldl.Solver(spa.triu(M, format='csc'), upper=True)

        x_first_qdldl = F.solve(b)
        x_first_qdldl_upper = F_upper.solve(b)

        # Update
        M.data = M.data + 0.1 * np.random.randn(M.nnz)
        # Symmetrize matrix
        M =.5 * (M + M.T)

        F.update(M)
        F_upper.update(spa.triu(M, format='csc'), upper=True)
        x_second_qdldl = F.solve(b)
        x_second_qdldl_upper = F_upper.solve(b)

        nptest.assert_allclose(x_second_qdldl,
                               x_second_qdldl_upper,
                               rtol=1e-05,
                               atol=1e-05)


