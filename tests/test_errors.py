import qdldl
import scipy.sparse as spa
import numpy as np
import unittest


class solve_ls(unittest.TestCase):

    def test_wrong_size_A(self):
        np.random.seed(2)

        A = spa.random(10, 12)

        with self.assertRaises(ValueError):
            F = qdldl.Solver(A)


    def test_wrong_size_b(self):
        np.random.seed(2)

        A = spa.eye(10)
        b = np.random.randn(8)

        F = qdldl.Solver(A)

        #  x_qdldl = F.solve(b)
        with self.assertRaises(ValueError):
            x_qdldl = F.solve(b)




