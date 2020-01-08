import numpy as np
import scipy.sparse as spa
import _qdldl

class QDLDLFactor:

    def __init__(self, Lx, Li, Lp, Dinv, P):
        self._Lx = Lx
        self._Lp = Lp
        self._Li = Li
        self._Dinv = Dinv
        self._P = P

    @property
    def L(self):
        return spa.csc_matrix((self._Lx, self._Li, self._Lp))

    @property
    def D(self):
        return spa.diags(np.reciprocal(self._Dinv))

    @property
    def P(self):
        return self._P

    def solve(self, b):
        return _qdldl.solve(b)


def factor(A):

    # Perform all the checks
    m, n = A.shape

    if m != n:
        raise ValueError("Matrix A is not square")

    if not spa.isspmatrix_csc(A):
        warn("Converting matrix A to a Sparse CSC " +
             "(compressed sparse column) matrix. (It may take a while...)")
        A = spa.csc_matrix(A)

    if A.nnz == 0:
        raise ValueError("Matrix A is empty")

    A = spa.triu(A, format='csc')

    # TODO: Factor using low level


    # Return













