# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

#include "qdldl_types.h"

/**
 * C = A(p,p)= PAP' where A and C are symmetric the upper part stored;
 * NB. It assumes all the values are allocated
 */
void symperm(QDLDL_int n,
		     const QDLDL_int * Ap,
			 const QDLDL_int * Ai,
			 const QDLDL_float * Ax,
			 QDLDL_int * Cp,
			 QDLDL_int * Ci,
			 QDLDL_float * Cx,
			 const QDLDL_int * pinv,
			 QDLDL_int * AtoC,
			 QDLDL_int * w);


/**
 * Compute inverse of permutation matrix stored in the vector p.
 * The computed inverse is also stored in a vector.
 */
void pinv(const QDLDL_int *p, QDLDL_int * pinv, QDLDL_int        n);


/* Permute x = P*b using P */
void permute_x(QDLDL_int n, QDLDL_float * x, const QDLDL_float * b, const QDLDL_int * P);

/* Permute x = P'*b using P */
void permutet_x(QDLDL_int n, QDLDL_float * x, const QDLDL_float * b, const QDLDL_int * P);


/* Update permuted matrix A with Anewx */
void update_A(QDLDL_int Anz, QDLDL_float * Apermx, QDLDL_float * Anewx, const QDLDL_int *AtoAperm);


#ifdef __cplusplus
}
#endif
