// Source code for QDLDL, AMD and permutations
#include "qdldl/include/qdldl.h"
#include "amd/include/amd.h"
#include "amd/include/perm.h"
#include <stdexcept>
#include <string>


namespace qdldl {

class Solver {

	private:
		// Matrix L
		QDLDL_int * Lp;
		QDLDL_int * Li;
		QDLDL_float * Lx;

		// Matrix D
		QDLDL_float * D;
		QDLDL_float * Dinv;

		// Matrix P (permutation)
		QDLDL_int * P;
		QDLDL_int * Pinv;

		// Workspace
		QDLDL_int * etree;
		QDLDL_int * Lnz;
		QDLDL_int * iwork;
		QDLDL_bool * bwork;
		QDLDL_float * fwork;

		// Permuted A
		QDLDL_int * Aperm_p;
		QDLDL_int * Aperm_i;
		QDLDL_float * Aperm_x;
		QDLDL_int * A2Aperm;

	public:
		QDLDL_int nx; // Size
		QDLDL_int nnz; // Number of nonzero elements in the matrix
		QDLDL_int sum_Lnz; // Number of nonzero elements in the factor L

		Solver(QDLDL_int n, QDLDL_int * Ap, QDLDL_int *Ai, QDLDL_float * Ax);
		QDLDL_float * solve(QDLDL_float * b);
		void update(QDLDL_float * Anew_x);

        QDLDL_float *get_D();
        QDLDL_int *get_Lp();
        QDLDL_int *get_Li();
        QDLDL_float *get_Lx();
        QDLDL_int *get_P();

		~Solver();

};

} // end namespace










