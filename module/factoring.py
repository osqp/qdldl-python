import numpy as np
import scipy.sparse as spa
from warnings import warn
from qdldl.qdldl import PySolver


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

    #  return PySolver(np.ascontiguousarray(A.indptr),
    #                  np.ascontiguousarray(A.indices),
    #                  np.ascontiguousarray(A.data))


    print("Ap ", A.indptr)
    print("Ai ", A.indices)
    print("Ax ", A.data)

    return PySolver(A.indptr, A.indices, A.data)













