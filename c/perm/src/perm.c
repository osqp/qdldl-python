#include "perm.h"

c_int csc_cumsum(c_int *p, c_int *c, c_int n) {
  c_int i, nz = 0;

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


void permute_x(c_int n, c_float * x, const c_float * b, const c_int * P) {
    c_int j;
    for (j = 0 ; j < n ; j++) x[j] = b[P[j]];
}


void permutet_x(c_int n, c_float * x, const c_float * b, const c_int * P) {
    c_int j;
    for (j = 0 ; j < n ; j++) x[P[j]] = b[j];
}



KKT_temp = csc_symperm((*KKT), Pinv, KtoPKPt, 1);



csc* csc_symperm(const csc *A, const c_int *pinv, c_int *AtoC, c_int values) {
  c_int i, j, p, q, i2, j2, n, *Ap, *Ai, *Cp, *Ci, *w;
  c_float *Cx, *Ax;
  csc     *C;

  n  = A->n;
  Ap = A->p;
  Ai = A->i;
  Ax = A->x;
  C  = csc_spalloc(n, n, Ap[n], values && (Ax != OSQP_NULL),
                   0);                                /* alloc result*/
  w = csc_calloc(n, sizeof(c_int));                   /* get workspace */

  if (!C || !w) return csc_done(C, w, OSQP_NULL, 0);  /* out of memory */

  Cp = C->p;
  Ci = C->i;
  Cx = C->x;

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
  csc_cumsum(Cp, w, n);        /* compute column pointers of C */

  for (j = 0; j < n; j++) {
    j2 = pinv ? pinv[j] : j;   /* column j of A is column j2 of C */

    for (p = Ap[j]; p < Ap[j + 1]; p++) {
      i = Ai[p];

      if (i > j) continue;                             /* skip lower triangular
                                                          part of A*/
      i2                         = pinv ? pinv[i] : i; /* row i of A is row i2
                                                          of C */
      Ci[q = w[c_max(i2, j2)]++] = c_min(i2, j2);

      if (Cx) Cx[q] = Ax[p];

      if (AtoC) { // If vector AtoC passed, store values of the mappings
        AtoC[p] = q;
      }
    }
  }
  return csc_done(C, w, OSQP_NULL, 1); /* success; free workspace, return C */
}
