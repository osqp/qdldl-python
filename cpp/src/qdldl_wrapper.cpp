#include <stdlib.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// Source code for QDLDL, AMD and permutations
#include "qdldl.h"
#include "amd.h"
#include "perm.h"

namespace py = pybind11;


int py_etree(const py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast> Ap_py,
	  const py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast> Ai_py,
	  py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast> iwork_py,
	  py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast> Lnz_py,
	  py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast> etree_py){


    QDLDL_int n = Ap_py.size() - 1;

    auto Ap = static_cast<QDLDL_int *>(Ap_py.request().ptr);
    auto Ai = static_cast<QDLDL_int *>(Ai_py.request().ptr);
    auto iwork = static_cast<QDLDL_int *>(iwork_py.request().ptr);
    auto Lnz = static_cast<QDLDL_int *>(Lnz_py.request().ptr);
    auto etree = static_cast<QDLDL_int *>(etree_py.request().ptr);

    int sum_Lnz = QDLDL_etree(n, Ap, Ai, iwork, Lnz, etree);

    if (sum_Lnz < 0) py::value_error("Input matrix is not quasi-definite");

    return sum_Lnz;

}



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
	py::array_t<QDLDL_int> Pinv_np = py::array_t<QDLDL_int>(n);
	QDLDL_int * Pinv = static_cast<QDLDL_int *>(Pinv_np.request().ptr);
    QDLDL_float *info = new QDLDL_float[AMD_INFO];

	QDLDL_int amd_status = amd_l_order(n, Ap, Ai, P, NULL, info);
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
	delete [] info;
	delete [] Apermp;
	delete [] Apermi;
	delete [] Apermx;
	delete [] work_perm;

	// Return tuple of results
	py::tuple returns = py::make_tuple(Lp_np, Li_np, Lx_np, D_np, Dinv_np, P_np, Pinv_np);

	return returns;

}


void init_qdldl_wrapper(py::module &m){
  // m.def("etree", &py_etree);
  m.def("factor", &py_factor);
  // m.def("factor", &py_factor);
  // m.def("solve", &py_solve);
}
