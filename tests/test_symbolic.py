import qdldl
import scipy.sparse as spa
import scipy.sparse.linalg as sla
from .utils import random_psd
import numpy as np

import unittest
import numpy.testing as nptest


class symbolic(unittest.TestCase):

    def setUp(self):
        np.random.seed(2)

    def test_basic_solve(self):
        n = 5
        A = random_psd(n, n)
        B = random_psd(n, n)
        C = - random_psd(n, n)
        M = spa.bmat([[A, B.T], [B, C]], format='csc')
        b = np.random.randn(2 * n)

        sym = qdldl.SymbolicSolver(M)
        sym.factor(M)
        x_qdldl = sym.solve(b)
        x_scipy = sla.spsolve(M, b)

        nptest.assert_array_almost_equal(x_qdldl, x_scipy)

    def test_refactor(self):
        n = 5
        A = random_psd(n, n)
        B = random_psd(n, n)
        C = - random_psd(n, n)
        M = spa.bmat([[A, B.T], [B, C]], format='csc')
        b = np.random.randn(2 * n)

        sym = qdldl.SymbolicSolver(M)

        sym.factor(M)
        x1 = sym.solve(b)
        x1_scipy = sla.spsolve(M, b)
        nptest.assert_array_almost_equal(x1, x1_scipy)

        # Change values, keep sparsity
        M.data = M.data + 0.1 * np.random.randn(M.nnz)
        M = .5 * (M + M.T)

        sym.factor(M)
        x2 = sym.solve(b)
        x2_scipy = sla.spsolve(M, b)
        nptest.assert_allclose(x2, x2_scipy, rtol=1e-05, atol=1e-05)

    def test_factors(self):
        n = 5
        A = random_psd(n, n)
        B = random_psd(n, n)
        C = - random_psd(n, n)
        M = spa.bmat([[A, B.T], [B, C]], format='csc')
        m = 2 * n

        sym = qdldl.SymbolicSolver(M)
        sym.factor(M)
        L, D, p = sym.factors()
        L = L.toarray()

        nptest.assert_array_almost_equal(np.triu(L), np.zeros((m, m)))
        assert (np.sort(p) == np.arange(m)).all()

        P = np.zeros((m, m))
        P[p, np.arange(m)] = 1

        L += np.eye(m)
        nptest.assert_array_almost_equal(P @ L @ np.diag(D) @ L.T @ P.T,
                                         M.toarray())

    def test_matches_solver(self):
        n = 5
        A = random_psd(n, n)
        B = random_psd(n, n)
        C = - random_psd(n, n)
        M = spa.bmat([[A, B.T], [B, C]], format='csc')
        b = np.random.randn(2 * n)

        solver = qdldl.Solver(M)
        sym = qdldl.SymbolicSolver(M)
        sym.factor(M)

        nptest.assert_array_almost_equal(sym.solve(b), solver.solve(b))

        L1, D1, p1 = solver.factors()
        L2, D2, p2 = sym.factors()
        nptest.assert_array_almost_equal(L1.toarray(), L2.toarray())
        nptest.assert_array_almost_equal(D1, D2)
        nptest.assert_array_equal(p1, p2)

    def test_upper(self):
        n = 5
        A = random_psd(n, n)
        B = random_psd(n, n)
        C = - random_psd(n, n)
        M = spa.bmat([[A, B.T], [B, C]], format='csc')
        b = np.random.randn(2 * n)

        M_triu = spa.triu(M, format='csc')
        sym = qdldl.SymbolicSolver(M_triu, upper=True)
        sym.factor(M_triu, upper=True)
        x_upper = sym.solve(b)

        sym2 = qdldl.SymbolicSolver(M)
        sym2.factor(M)
        x_full = sym2.solve(b)

        nptest.assert_allclose(x_upper, x_full, rtol=1e-05, atol=1e-05)

    def test_pattern_only_tuple(self):
        n = 5
        A = random_psd(n, n)
        B = random_psd(n, n)
        C = - random_psd(n, n)
        M = spa.bmat([[A, B.T], [B, C]], format='csc')
        M_triu = spa.triu(M, format='csc')
        b = np.random.randn(2 * n)

        # Construct from (indptr, indices, shape) — no data array
        sym = qdldl.SymbolicSolver(
            (M_triu.indptr, M_triu.indices, M_triu.shape), upper=True
        )
        sym.factor(M)

        sym2 = qdldl.SymbolicSolver(M)
        sym2.factor(M)

        nptest.assert_array_almost_equal(sym.solve(b), sym2.solve(b))

    def test_solve_before_factor_raises(self):
        n = 3
        M = spa.eye(n, format='csc')
        sym = qdldl.SymbolicSolver(M)
        b = np.ones(n)

        with self.assertRaises(RuntimeError):
            sym.solve(b)
