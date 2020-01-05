import numpy as np
import scipy.sparse as spa
from warnings import warn
cimport cython


cdef extern from "amd.h":

    ctypedef long long SuiteSparse_long
    ctypedef double c_float
    int AMD_INFO

    SuiteSparse_long amd_l_order(SuiteSparse_long n,
                                 const SuiteSparse_long Ap [ ],
                                 const SuiteSparse_long Ai [ ],
                                 SuiteSparse_long P [ ],
                                 c_float Control [ ],
                                 c_float Info [ ]) nogil


cdef extern from "qdldl.h":

    ctypedef long long QDLDL_int
    ctypedef double QDLDL_float
    ctypedef unsigned char QDLDL_bool

    QDLDL_int QDLDL_etree(const QDLDL_int   n,
                       const QDLDL_int* Ap,
                       const QDLDL_int* Ai,
                       QDLDL_int* work,
                       QDLDL_int* Lnz,
                       QDLDL_int* etree) nogil

    QDLDL_int QDLDL_factor(const QDLDL_int    n,
                  const QDLDL_int*   Ap,
                  const QDLDL_int*   Ai,
                  const QDLDL_float* Ax,
                  QDLDL_int*   Lp,
                  QDLDL_int*   Li,
                  QDLDL_float* Lx,
                  QDLDL_float* D,
                  QDLDL_float* Dinv,
                  const QDLDL_int* Lnz,
                  const QDLDL_int* etree,
                  QDLDL_bool* bwork,
                  QDLDL_int* iwork,
                  QDLDL_float* fwork) nogil

    void QDLDL_solve(const QDLDL_int    n,
                     const QDLDL_int*   Lp,
                     const QDLDL_int*   Li,
                     const QDLDL_float* Lx,
                     const QDLDL_float* Dinv,
                     QDLDL_float* x) nogil

    void QDLDL_Lsolve(const QDLDL_int    n,
                      const QDLDL_int*   Lp,
                      const QDLDL_int*   Li,
                      const QDLDL_float* Lx,
                      QDLDL_float* x) nogil

    void QDLDL_Ltsolve(const QDLDL_int    n,
                       const QDLDL_int*   Lp,
                       const QDLDL_int*   Li,
                       const QDLDL_float* Lx,
                       QDLDL_float* x) nogil



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

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    def solve(self, b):

        x_vec = np.copy(b, order='C')
        cdef QDLDL_int n = b.size
        cdef QDLDL_int[::1] Li = self._Li
        cdef QDLDL_int[::1] Lp = self._Lp
        cdef QDLDL_float[::1] Lx = self._Lx
        cdef QDLDL_float[::1] Dinv = self._Dinv
        cdef QDLDL_float[::1] P = self._P
        cdef QDLDL_float[::1] x = x_vec

        # Solve Px = b
        for j in range(n):
            x[j] = b[P[j]]

        with nogil:
            QDLDL_solve(n, &Lp[0], &Li[0], &Lx[0], &Dinv[0], &x[0]);

        # Solve Px = b
        for j in range(n):
            x[P[j]] = b[j]

        return x_vec


def permute(A):

    cdef QDLDL_int[::1] Ai = np.ascontiguousarray(A.indices.astype(np.int64))
    cdef QDLDL_int[::1] Ap = np.ascontiguousarray(A.indptr.astype(np.int64))

    # Memory allocations AMD
    cdef QDLDL_float[::1] P = np.empty(n, dtype=np.double, order='C')
    cdef QDLDL_float[::1] info = np.empty(AMD_INFO, dtype=np.double, order='C')

    # Construct permutation
    with nogil:
        amd_status = amd_l_order(n, &Ap[0], &Ai[0], &P[0], NULL, &info[0]);

    if amd_status < 0:
        raise ValueError("Error in Approximate Minimum Degree ordering.")

    # Permute matrix A


    # RETURN permuted A and P

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def factor(A):

    # Perform all the checks
    cdef QDLDL_int m, n
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
    A, P = permute(A)

    # TODO: Check if arrays are contiguous before converting
    # them using np.ascontiguousarray
    cdef QDLDL_int[::1] Ai = np.ascontiguousarray(A.indices.astype(np.int64))
    cdef QDLDL_int[::1] Ap = np.ascontiguousarray(A.indptr.astype(np.int64))
    cdef QDLDL_float[::1] Ax = np.ascontiguousarray(A.data)

    # Memory allocations QDLDL
    cdef QDLDL_int[::1] etree = np.empty(n, dtype=np.int64, order='C')
    cdef QDLDL_int[::1] Lnz = np.empty(n, dtype=np.int64, order='C')
    cdef QDLDL_int[::1] Lp = np.empty(n + 1, dtype=np.int64, order='C')
    cdef QDLDL_float[::1] D = np.empty(n, dtype=np.double, order='C')
    cdef QDLDL_float[::1] Dinv = np.empty(n, dtype=np.double, order='C')
    cdef QDLDL_int[::1] iwork = np.empty(3 * n, dtype=np.int64, order='C')
    cdef QDLDL_bool[::1] bwork = np.empty(n, dtype=np.ubyte, order='C')
    cdef QDLDL_float[::1] fwork = np.empty(n, dtype=np.double, order='C')


    # Construct elimination tree
    with nogil:
        sum_Lnz = QDLDL_etree(n, &Ap[0], &Ai[0], &iwork[0], &Lnz[0], &etree[0])

    if sum_Lnz < 0:
        raise ValueError("Input matrix is not quasi-definite")

    # Allocate L memory
    cdef QDLDL_int[::1] Li = np.empty(sum_Lnz, dtype=np.int64, order='C')
    cdef QDLDL_float[::1] Lx = np.empty(sum_Lnz, dtype=np.double, order='C')

    # Factor
    with nogil:
        QDLDL_factor(n, &Ap[0], &Ai[0], &Ax[0],
                     &Lp[0], &Li[0], &Lx[0], &D[0], &Dinv[0], &Lnz[0],
                     &etree[0], &bwork[0], &iwork[0], &fwork[0])

    return QDLDLFactor(np.asarray(Lx), np.asarray(Li), np.asarray(Lp),
                       np.asarray(Dinv), P)
