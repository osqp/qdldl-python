#include "perm.h"

# define c_max(a, b) (((a) > (b)) ? (a) : (b))
# define c_min(a, b) (((a) < (b)) ? (a) : (b))

QDLDL_int cumsum(QDLDL_int *p, QDLDL_int *c, QDLDL_int n) {
  QDLDL_int i, nz = 0;

  if (!p || !c) return -1;  /* check inputs */

  for (i = 0; i < n; i++)
  {
    p[i] = nz;
    nz  += c[i];
    c[i] = p[i];
  }
  p[n] = nz;
  return nz; /* return sum (c [0..n-1]) */
}


void permute_x(QDLDL_int n, QDLDL_float * x, const QDLDL_float * b, const QDLDL_int * P) {
    for (QDLDL_int j = 0 ; j < n ; j++) x[j] = b[P[j]];
}


void permutet_x(QDLDL_int n, QDLDL_float * x, const QDLDL_float * b, const QDLDL_int * P) {
    for (QDLDL_int j = 0 ; j < n ; j++) x[P[j]] = b[j];
}


void pinv(QDLDL_int const *p, QDLDL_int * pinv, QDLDL_int n) {
  for (QDLDL_int k = 0; k < n; k++) pinv[p[k]] = k;  /* invert the permutation */
}


void symperm(QDLDL_int n,
		     const QDLDL_int * Ap,
			 const QDLDL_int * Ai,
			 const QDLDL_float * Ax,
			 QDLDL_int * Cp,
			 QDLDL_int * Ci,
			 QDLDL_float * Cx,
			 const QDLDL_int * pinv,
			 QDLDL_int * AtoC,
			 QDLDL_int * w){
  QDLDL_int i, j, p, q, i2, j2;

  for (j = 0; j < n; j++)    /* count entries in each column of C */
  {
    j2 = pinv ? pinv[j] : j; /* column j of A is column j2 of C */

    for (p = Ap[j]; p < Ap[j + 1]; p++) {
      i = Ai[p];

      if (i > j) continue;     /* skip lower triangular part of A */
      i2 = pinv ? pinv[i] : i; /* row i of A is row i2 of C */
      w[c_max(i2, j2)]++;      /* column count of C */
    }
  }
  cumsum(Cp, w, n);        /* compute column pointers of C */

  for (j = 0; j < n; j++) {
    j2 = pinv ? pinv[j] : j;   /* column j of A is column j2 of C */

    for (p = Ap[j]; p < Ap[j + 1]; p++) {
      i = Ai[p];

      if (i > j) continue;                             /* skip lower triangular
                                                          part of A*/
      i2 = pinv ? pinv[i] : i;                         /* row i of A is row i2
                                                          of C */
      Ci[q = w[c_max(i2, j2)]++] = c_min(i2, j2);

      if (Cx) Cx[q] = Ax[p];

	  if (AtoC) { // If vector AtoC passed, store values of the mappings
        AtoC[p] = q;
      }

    }
  }
}


void update_A(QDLDL_int Anz, QDLDL_float * Apermx,
		      QDLDL_float * Anewx, const QDLDL_int *AtoAperm) {
  for (QDLDL_int i = 0; i < Anz; i++) Apermx[AtoAperm[i]] = Anewx[i];
}
