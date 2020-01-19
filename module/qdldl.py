import numpy as np
import scipy.sparse as spa
from warnings import warn
import qdldl._qdldl as _qdldl

class QDLDLFactor:

    def __init__(self, n, Lx, Li, Lp, D, Dinv, P, Apermp, Apermi, Apermx, A2Aperm,
                 Lnz, etree, iwork, bwork, fwork):
        self._n = n
        self._Lx = Lx
        self._Lp = Lp
        self._Li = Li
        self._D = D
        self._Dinv = Dinv
        self._P = P
        self._Apermp = Apermp
        self._Apermi = Apermi
        self._Apermx = Apermx
        self._A2Aperm = A2Aperm
        self._Lnz = Lnz
        self._etree = etree
        self._iwork = iwork
        self._bwork = bwork
        self._fwork = fwork

    @property
    def L(self):
        return spa.csc_matrix((self._Lx, self._Li, self._Lp))

    @property
    def D(self):
        return spa.diags(self._D)

    @property
    def P(self):
        return self._P

    @property
    def A2Aperm(self):
        return self._A2Aperm

    def solve(self, b):
        return _qdldl.solve(self._n,
                            b,
                            self._Lp,
                            self._Li,
                            self._Lx,
                            self._Dinv,
                            self._P)

    def update(self, Anew_x):
        return _qdldl.update(
                self._A2Aperm,
                self._Apermp,
                self._Apermi,
                self._Apermx,
                self._Lp,
                self._Li,
                self._Lx,
                self._D,
                self._Dinv,
                self._Lnz,
                self._etree,
                self._iwork,
                self._bwork,
                self._fwork)


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


    if not A.has_sorted_indices:
        A.sort_indices()

    #  Lp, Li, Lx, D, Dinv, P, Apermp, Apermi, Apermx, A2Aperm = _qdldl.factor(A.indptr, A.indices, A.data)
    f = _qdldl.factor(A.indptr, A.indices, A.data)

    return QDLDLFactor(n, *f)













