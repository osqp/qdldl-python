// Source code for QDLDL, AMD and permutations
#include "qdldl/include/qdldl.h"
#include "amd/include/amd.h"
#include "amd/include/perm.h"

#include <stdlib.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace py = pybind11;


py::tuple py_factor(const py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast> Ap_py,
	  const py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast> Ai_py,
	  const py::array_t<QDLDL_float, py::array::c_style | py::array::forcecast> Ax_py){

    QDLDL_int n = Ap_py.size() - 1;

    // Extract arrays
    auto Ap = static_cast<QDLDL_int *>(Ap_py.request().ptr);
    auto Ai = static_cast<QDLDL_int *>(Ai_py.request().ptr);
    auto Ax = static_cast<QDLDL_float *>(Ax_py.request().ptr);

    QDLDL_int Anz = Ap[n];

	//For the elimination tree
	QDLDL_int *etree = new QDLDL_int[n];
	QDLDL_int *Lnz   = new QDLDL_int[n];

	//For the L factors.   Li and Lx are sparsity dependent
	//so must be done after the etree is constructed
	py::array_t<QDLDL_int> Lp_np = py::array_t<QDLDL_int>(n + 1);
	QDLDL_int * Lp = static_cast<QDLDL_int *>(Lp_np.request().ptr);
	py::array_t<QDLDL_float> D_np = py::array_t<QDLDL_float>(n);
	QDLDL_float * D = static_cast<QDLDL_float *>(D_np.request().ptr);
	py::array_t<QDLDL_float> Dinv_np = py::array_t<QDLDL_float>(n);
	QDLDL_float * Dinv = static_cast<QDLDL_float *>(Dinv_np.request().ptr);

	//Working memory.  Note that both the etree and factor
	//calls requires a working vector of QDLDL_int, with
	//the factor function requiring 3*An elements and the
	//etree only An elements.   Just allocate the larger
	//amount here and use it in both places
	QDLDL_int *iwork = new QDLDL_int[3 * n];
	QDLDL_bool *bwork = new QDLDL_bool[n];
	QDLDL_float *fwork = new QDLDL_float[n];

	// Permute A
	py::array_t<QDLDL_int> P_np = py::array_t<QDLDL_int>(n);
	QDLDL_int * P = static_cast<QDLDL_int *>(P_np.request().ptr);
	QDLDL_int * Pinv = new QDLDL_int[n];

	QDLDL_int amd_status = amd_l_order(n, Ap, Ai, P, NULL, NULL);
	if (amd_status < 0)
		throw py::value_error("Error in AMD computation " + std::to_string(amd_status));

	pinv(P, Pinv, n); // Compute inverse permutation

	// Compute permuted matrix
	QDLDL_int *Apermp = new QDLDL_int[n + 1];
	QDLDL_int *Apermi = new QDLDL_int[Anz];
	QDLDL_float *Apermx = new QDLDL_float[Anz];
	QDLDL_int *work_perm = new QDLDL_int[n](); // Initialize to 0

	symperm(n, Ap, Ai, Ax, Apermp, Apermi, Apermx, Pinv, work_perm);

	// Compute elimination tree
    int sum_Lnz = QDLDL_etree(n, Apermp, Apermi, iwork, Lnz, etree);

	if (sum_Lnz < 0)
		throw py::value_error("Input matrix is not quasi-definite, sum_Lnz = " + std::to_string(sum_Lnz));

	py::array_t<QDLDL_int> Li_np = py::array_t<QDLDL_int>(sum_Lnz);
	QDLDL_int * Li = static_cast<QDLDL_int *>(Li_np.request().ptr);
	py::array_t<QDLDL_float> Lx_np = py::array_t<QDLDL_float>(sum_Lnz);
	QDLDL_float* Lx = static_cast<QDLDL_float *>(Lx_np.request().ptr);

	// Compute numeric factorization
    QDLDL_factor(n, Apermp, Apermi, Apermx,
			     Lp, Li, Lx,
				 D, Dinv, Lnz,
				 etree, bwork, iwork, fwork);

    // Delete memory
	delete [] etree;
	delete [] Lnz;
	delete [] iwork;
	delete [] bwork;
	delete [] fwork;
	delete [] Apermp;
	delete [] Apermi;
	delete [] Apermx;
	delete [] Pinv;
	delete [] work_perm;

	// Return tuple of results
	py::tuple returns = py::make_tuple(Lp_np, Li_np, Lx_np, D_np, Dinv_np, P_np);

	return returns;

}

/* solves P'LDL'P x = b for x */
py::array_t<QDLDL_float> py_solve(QDLDL_int n,
const py::array_t<QDLDL_float, py::array::c_style | py::array::forcecast> b_py,
const py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast> Lp_py,
const py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast> Li_py,
const py::array_t<QDLDL_float, py::array::c_style | py::array::forcecast> Lx_py,
const py::array_t<QDLDL_float, py::array::c_style | py::array::forcecast> Dinv_py,
const py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast> P_py) {


    // Extract arrays
    auto b = static_cast<QDLDL_float *>(b_py.request().ptr);
    auto Lp = static_cast<QDLDL_int *>(Lp_py.request().ptr);
    auto Li = static_cast<QDLDL_int *>(Li_py.request().ptr);
	auto Lx = static_cast<QDLDL_float *>(Lx_py.request().ptr);
	auto Dinv = static_cast<QDLDL_float *>(Dinv_py.request().ptr);
	auto P = static_cast<QDLDL_int *>(P_py.request().ptr);
	auto work = new QDLDL_float[n];

    // Create solution vector
	py::array_t<QDLDL_float> x_np = py::array_t<QDLDL_float>(n);
	QDLDL_float * x = static_cast<QDLDL_float *>(x_np.request().ptr);

    permute_x(n, work, b, P);
    QDLDL_solve(n, Lp, Li, Lx, Dinv, work);
    permutet_x(n, x, work, P);

	delete [] work;

	return x_np;
}

void init_qdldl_wrapper(py::module &m){
  m.def("factor", &py_factor);
  m.def("solve", &py_solve);
}
