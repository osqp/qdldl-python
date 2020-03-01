// Source code for QDLDL, AMD and permutations
#include "qdldl/include/qdldl.h"
#include "amd/include/amd.h"
#include "amd/include/perm.h"

// #include <stdlib.h>
// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
//
//
// namespace py = pybind11;




class QDLDLSolver {

	private:
		QDDLD_int n; // Size

		// Matrix L
		QDLDL_int * Lp;
		QDLDL_int * Li;
		QDLDL_float * Lx;

		// Matrix D
		QDLDL_float * D;
		QDLDL_float * Dinv;

		// Matrix P (permutation)
		QDLDL_int * P;

		// Workspace
		QDLDL_int * wtree;
		QDLDL_int * Lnz;
		QDLDL_int * iwork;
		QDLDL_bool * bwork;
		QDLDL_float * fwork;

		// Permuted A
		QDLDL_int * Aperm_p;
		QDLDL_int * Aperm_i;
		QDLDL_float * Aperm_x;

	public:
	// TODO: Add types
	int factor(QDLDL_int * Ap, QDLDL_int *Ai, QDLDL_float * Ax);
	int solve(QDLDL_float * b);
	int update(QDLDL_float * Ax);

};













