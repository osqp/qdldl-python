typedef long long c_int; /* for indices */
typedef double c_float; /* for numerical values  */


/**
 * C = A(p,p)= PAP' where A and C are symmetric the upper part stored;
 *  NB: pinv not p!
 * @param  A      Original matrix (upper-triangular)
 * @param  pinv   Inverse of permutation vector
 * @param  AtoC   Mapping from indices of A-x to C->x
 * @param  values Are values of A allocated?
 * @return        New matrix (allocated)
 */
csc* csc_symperm(const csc   *A,
                 const c_int *pinv,
                 c_int       *AtoC,
                 c_int        values);


/**
 * p [0..n] = cumulative sum of c [0..n-1], and then copy p [0..n-1] into c
 *
 * @param  p Create cumulative sum into p
 * @param  c Vector of which we compute cumulative sum
 * @param  n Number of elements
 * @return   Exitflag
 */
c_int csc_cumsum(c_int *p,
                 c_int *c,
                 c_int  n);

/**
 * Compute inverse of permutation matrix stored in the vector p.
 * The computed inverse is also stored in a vector.
 */
c_int* csc_pinv(c_int const *p,
                c_int        n);


/* Permute x = P*b using P */
void permute_x(c_int n, c_float * x, const c_float * b, const c_int * P);

/* Permute x = P'*b using P */
void permutet_x(c_int n, c_float * x, const c_float * b, const c_int * P);
