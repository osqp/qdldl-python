import numpy as np
import scipy.sparse as spa
from warnings import warn


cdef extern from "qdldl.h":

    ctypedef long QDLDL_int
    ctypedef double QDLDL_float
    ctypedef unsigned char QDLDL_bool

    QDLDL_int QDLDL_etree(const QDLDL_int   n,
                       const QDLDL_int* Ap,
                       const QDLDL_int* Ai,
                       QDLDL_int* work,
                       QDLDL_int* Lnz,
                       QDLDL_int* etree)

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
                  QDLDL_float* fwork)

    void QDLDL_solve(const QDLDL_int    n,
                     const QDLDL_int*   Lp,
                     const QDLDL_int*   Li,
                     const QDLDL_float* Lx,
                     const QDLDL_float* Dinv,
                     QDLDL_float* x)

    void QDLDL_Lsolve(const QDLDL_int    n,
                      const QDLDL_int*   Lp,
                      const QDLDL_int*   Li,
                      const QDLDL_float* Lx,
                      QDLDL_float* x);

    void QDLDL_Ltsolve(const QDLDL_int    n,
                       const QDLDL_int*   Lp,
                       const QDLDL_int*   Li,
                       const QDLDL_float* Lx,
                       QDLDL_float* x)



class QDLDLFactor:

    def __init__(self, L, Dinv_vec):
        self._L = L
        self._D_inv_vec = Dinv_vec

    @property
    def L(self):
        return self._L

    @property
    def D(self):
        return spa.diags(np.reciprocal(self._D_inv_vec))

    def solve(self, rhs):
        # Implement calling QDLDL solve
        return NotImplemented



def factor(A):

    # Perform all the checks
    m, n = A.shape
    if m != n:
        raise ValueError("Matrix A is not square")

    if not spa.isspmatrix_csc(A):
        warn("Converting matrix A to a Sparse CSC " +
             "(compressed sparse column) matrix. (It may take a while...)")
        A = spa.csc_matrix(A)


    cdef QDLDL_int[::1] A_i = np.ascontiguous(A.indices)
    cdef QDLDL_int[::1] A_p = np.ascontiguous(A.indptr)
    cdef QDLDL_float[::1] A_x = np.ascontiguous(A.data)

    # Memory allocations
    #  //For the elimination tree
    #  etree = (QDLDL_int*)malloc(sizeof(QDLDL_int)*An);
    #  Lnz   = (QDLDL_int*)malloc(sizeof(QDLDL_int)*An);
    #
    #  //For the L factors.   Li and Lx are sparsity dependent
    #  //so must be done after the etree is constructed
    #  Lp    = (QDLDL_int*)malloc(sizeof(QDLDL_int)*(An+1));
    #  D     = (QDLDL_float*)malloc(sizeof(QDLDL_float)*An);
    #  Dinv  = (QDLDL_float*)malloc(sizeof(QDLDL_float)*An);
    #
    #  //Working memory.  Note that both the etree and factor
    #  //calls requires a working vector of QDLDL_int, with
    #  //the factor function requiring 3*An elements and the
    #  //etree only An elements.   Just allocate the larger
    #  //amount here and use it in both places
    #  iwork = (QDLDL_int*)malloc(sizeof(QDLDL_int)*(3*An));
    #  bwork = (QDLDL_bool*)malloc(sizeof(QDLDL_bool)*An);
    #  fwork = (QDLDL_float*)malloc(sizeof(QDLDL_float)*An);
    cdef QDLDL_int[::1] etree = np.empty(n, dtype=int, order='C')
    cdef QDLDL_int[::1] Lnz = np.empty(n, dtype=int, order='C')
    Lp_vec = np.empty(n + 1, dtype=int, order='C')
    cdef QDLDL_int[::1] Lp = Lp_vec
    cdef QDLDL_float[::1] D = np.empty(n, dtype=np.double, order='C')
    Dinv_vec = np.empty(n, dtype=np.double, order='C')
    cdef QDLDL_float[::1] Dinv = Dinv_vec
    cdef QDLDL_int[::1] iwork = np.empty(3 * n, dtype=int, order='C')
    cdef QDLDL_bool[::1] bwork = np.empty(n, dtype=bool, order='C')
    cdef QDLDL_float[::1] fwork = np.empty(n, dtype=np.double, order='C')

    # Construct elimination tree
    sum_Lnz = QDLDL_etree(n, &A_p[0], &A_i[0], &iwork[0], &Lnz[0], &etree[0])

    # Allocate L memory
    Li_vec = np.empty(sum_Lnz, dtype=int, order='C')
    cdef QDLDL_int[::1] Li = Li_vec
    Lx_vec = np.empty(sum_Lnz, dtype=np.double, order='C')
    cdef QDLDL_float[::1] Lx = Lx_vec

    # Factor
    QDLDL_factor(n, &A_p[0], &A_i[0], &A_x[0],
                 &Lp[0], &Li[0], &Lx[0], &D[0], &Dinv[0], &Lnz[0],
                 &etree[0], &bwork[0], &iwork[0], &fwork[0])

    return QDLDLFactor(spa.csc_matrix((Lx_vec, Li_vec, Lp_vec)), Dinv_vec)




