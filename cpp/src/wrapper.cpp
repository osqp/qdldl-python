#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "qdldl.hpp"

namespace py = pybind11;


qdldl::Solver py_qdldl_solver(
		const py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast> Ap_py,
		const py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast> Ai_py,
		const py::array_t<QDLDL_float, py::array::c_style | py::array::forcecast> Ax_py){


	// Extract arrays
	QDLDL_int nx = Ap_py.request().size - 1;
	auto Ap = static_cast<QDLDL_int *>(Ap_py.request().ptr);
	auto Ai = static_cast<QDLDL_int *>(Ai_py.request().ptr);
	auto Ax = static_cast<QDLDL_float *>(Ax_py.request().ptr);

	return qdldl::Solver(nx, Ap, Ai, Ax);
}

py::array_t<QDLDL_float> py_solve(
		const py::array_t<QDLDL_float, py::array::c_style | py::array::forcecast> b_py){

	auto b = static_cast<QDLDL_float *>(b_py.request().ptr);
	auto x = solve(b);

    return py::array(n, x);
}


//
// py::tuple py_factor(
// const py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast> Ap_py,
// const py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast> Ai_py,
// const py::array_t<QDLDL_float, py::array::c_style | py::array::forcecast> Ax_py){
//
//     QDLDL_int n = Ap_py.request().size - 1;
//
//     py::print("n =", n);
//
//     // Extract arrays
//     auto Ap = static_cast<QDLDL_int *>(Ap_py.request().ptr);
//     auto Ai = static_cast<QDLDL_int *>(Ai_py.request().ptr);
//     auto Ax = static_cast<QDLDL_float *>(Ax_py.request().ptr);
//
//     QDLDL_int Anz = Ap[n];
//
//     py::print("Anz =", Anz);
//
//     //For the elimination tree
//     py::array_t<QDLDL_int> etree_np = py::array_t<QDLDL_int>(n);
//     QDLDL_int * etree = static_cast<QDLDL_int *>(etree_np.request().ptr);
//     py::array_t<QDLDL_int> Lnz_np = py::array_t<QDLDL_int>(n);
//     QDLDL_int * Lnz = static_cast<QDLDL_int *>(Lnz_np.request().ptr);
//
//     //For the L factors.   Li and Lx are sparsity dependent
//     //so must be done after the etree is constructed
//     py::array_t<QDLDL_int> Lp_np = py::array_t<QDLDL_int>(n + 1);
//     QDLDL_int * Lp = static_cast<QDLDL_int *>(Lp_np.request().ptr);
//     py::array_t<QDLDL_float> D_np = py::array_t<QDLDL_float>(n);
//     QDLDL_float * D = static_cast<QDLDL_float *>(D_np.request().ptr);
//     py::array_t<QDLDL_float> Dinv_np = py::array_t<QDLDL_float>(n);
//     QDLDL_float * Dinv = static_cast<QDLDL_float *>(Dinv_np.request().ptr);
//
//     //Working memory.  Note that both the etree and factor
//     //calls requires a working vector of QDLDL_int, with
//     //the factor function requiring 3*An elements and the
//     //etree only An elements.   Just allocate the larger
//     //amount here and use it in both places
//     //
//     py::array_t<QDLDL_int> iwork_np = py::array_t<QDLDL_int>(3 * n);
//     QDLDL_int * iwork = static_cast<QDLDL_int *>(iwork_np.request().ptr);
//     py::array_t<QDLDL_bool> bwork_np = py::array_t<QDLDL_bool>(n);
//     QDLDL_bool * bwork = static_cast<QDLDL_bool *>(bwork_np.request().ptr);
//     py::array_t<QDLDL_float> fwork_np = py::array_t<QDLDL_float>(n);
//     QDLDL_float * fwork = static_cast<QDLDL_float *>(fwork_np.request().ptr);
//
//     // Permute A
//     py::array_t<QDLDL_int> P_np = py::array_t<QDLDL_int>(n);
//     QDLDL_int * P = static_cast<QDLDL_int *>(P_np.request().ptr);
//     QDLDL_int * Pinv = new QDLDL_int[n];
//
//     QDLDL_int amd_status = amd_l_order(n, Ap, Ai, P, NULL, NULL);
//     if (amd_status < 0)
//         throw py::value_error("Error in AMD computation " + std::to_string(amd_status));
//
//     pinv(P, Pinv, n); // Compute inverse permutation
//
//     // Compute permuted matrix
//     py::array_t<QDLDL_int> Apermp_np = py::array_t<QDLDL_int>(n + 1);
//     QDLDL_int * Apermp = static_cast<QDLDL_int *>(Apermp_np.request().ptr);
//     py::array_t<QDLDL_int> Apermi_np = py::array_t<QDLDL_int>(Anz);
//     QDLDL_int * Apermi = static_cast<QDLDL_int *>(Apermi_np.request().ptr);
//     py::array_t<QDLDL_float> Apermx_np = py::array_t<QDLDL_float>(Anz);
//     QDLDL_float * Apermx = static_cast<QDLDL_float *>(Apermx_np.request().ptr);
//     QDLDL_int *work_perm = new QDLDL_int[n](); // Initialize to 0
//
//     py::array_t<QDLDL_int> A2Aperm_np = py::array_t<QDLDL_int>(n);
//     QDLDL_int * A2Aperm = static_cast<QDLDL_int *>(A2Aperm_np.request().ptr);
//
//     symperm(n, Ap, Ai, Ax, Apermp, Apermi, Apermx, Pinv, A2Aperm, work_perm);
//
//     // Compute elimination tree
//     int sum_Lnz = QDLDL_etree(n, Apermp, Apermi, iwork, Lnz, etree);
//
//     if (sum_Lnz < 0)
//         throw py::value_error("Input matrix is not quasi-definite, sum_Lnz = " + std::to_string(sum_Lnz));
//
//     py::print("sum_Lnz = ", sum_Lnz);
//
//     py::array_t<QDLDL_int> Li_np = py::array_t<QDLDL_int>(sum_Lnz);
//     QDLDL_int * Li = static_cast<QDLDL_int *>(Li_np.request().ptr);
//     py::array_t<QDLDL_float> Lx_np = py::array_t<QDLDL_float>(sum_Lnz);
//     QDLDL_float* Lx = static_cast<QDLDL_float *>(Lx_np.request().ptr);
//
//     // Compute numeric factorization
//     QDLDL_factor(n, Apermp, Apermi, Apermx,
//                  Lp, Li, Lx,
//                  D, Dinv, Lnz,
//                  etree, bwork, iwork, fwork);
//
//     // Delete memory
//     delete [] Pinv;
//     delete [] work_perm;
//
//     // Return tuple of results
//     py::tuple returns = py::make_tuple(Lp_np, Li_np, Lx_np,
//                                        D_np, Dinv_np, P_np,
//                                        Apermp_np, Apermi_np, Apermx_np, A2Aperm_np,
//                                        Lnz_np, etree_np, iwork_np, bwork_np, fwork_np);
//
//     return returns;
//
// }
//
// [> solves P'LDL'P x = b for x <]
// py::array_t<QDLDL_float> py_solve(QDLDL_int n,
// const py::array_t<QDLDL_float, py::array::c_style | py::array::forcecast> b_py,
// const py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast> Lp_py,
// const py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast> Li_py,
// const py::array_t<QDLDL_float, py::array::c_style | py::array::forcecast> Lx_py,
// const py::array_t<QDLDL_float, py::array::c_style | py::array::forcecast> Dinv_py,
// const py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast> P_py) {
//
//
//     // Extract arrays
//     auto b = static_cast<QDLDL_float *>(b_py.request().ptr);
//     auto Lp = static_cast<QDLDL_int *>(Lp_py.request().ptr);
//     auto Li = static_cast<QDLDL_int *>(Li_py.request().ptr);
//     auto Lx = static_cast<QDLDL_float *>(Lx_py.request().ptr);
//     auto Dinv = static_cast<QDLDL_float *>(Dinv_py.request().ptr);
//     auto P = static_cast<QDLDL_int *>(P_py.request().ptr);
//     auto work = new QDLDL_float[n];
//
//     // Create solution vector
//     py::array_t<QDLDL_float> x_np = py::array_t<QDLDL_float>(n);
//     QDLDL_float * x = static_cast<QDLDL_float *>(x_np.request().ptr);
//
//     permute_x(n, work, b, P);
//     QDLDL_solve(n, Lp, Li, Lx, Dinv, work);
//     permutet_x(n, x, work, P);
//
//     delete [] work;
//
//     return x_np;
// }
//
//
// void py_update(
// QDLDL_float * Anewx,
// const py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast> A2Apermp_py,
// const py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast> Apermp_py,
// const py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast> Apermi_py,
// py::array_t<QDLDL_float, py::array::c_style | py::array::forcecast> Apermx_py,
// const py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast> Lp_py,
// const py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast> Li_py,
// py::array_t<QDLDL_float, py::array::c_style | py::array::forcecast> Lx_py,
// py::array_t<QDLDL_float, py::array::c_style | py::array::forcecast> D_py,
// py::array_t<QDLDL_float, py::array::c_style | py::array::forcecast> Dinv_py,
// const py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast> Lnz_py,
// const py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast> etree_py,
// py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast> iwork_py,
// py::array_t<QDLDL_bool, py::array::c_style | py::array::forcecast> bwork_py,
// py::array_t<QDLDL_float, py::array::c_style | py::array::forcecast> fwork_py
// ){
//
//     QDLDL_int n = Apermp_py.size() - 1;
//     QDLDL_int Anz = Apermx_py.size();
//
//     auto A2Aperm = static_cast<QDLDL_int *>(A2Apermp_py.request().ptr);
//     auto Apermp = static_cast<QDLDL_int *>(Apermp_py.request().ptr);
//     auto Apermi = static_cast<QDLDL_int *>(Apermi_py.request().ptr);
//     auto Apermx = static_cast<QDLDL_float *>(Apermx_py.request().ptr);
//     auto Lp = static_cast<QDLDL_int *>(Lp_py.request().ptr);
//     auto Li = static_cast<QDLDL_int *>(Li_py.request().ptr);
//     auto Lx = static_cast<QDLDL_float *>(Lx_py.request().ptr);
//     auto D = static_cast<QDLDL_float *>(D_py.request().ptr);
//     auto Dinv = static_cast<QDLDL_float *>(Dinv_py.request().ptr);
//     auto Lnz = static_cast<QDLDL_int *>(Lnz_py.request().ptr);
//     auto etree = static_cast<QDLDL_int *>(etree_py.request().ptr);
//     auto iwork = static_cast<QDLDL_int *>(iwork_py.request().ptr);
//     auto bwork = static_cast<QDLDL_bool *>(bwork_py.request().ptr);
//     auto fwork = static_cast<QDLDL_float *>(fwork_py.request().ptr);
//
//     // Update matrix
//     update_A(Anz, Apermx, Anewx, A2Aperm);
//
//     // Compute numeric factorization
//     QDLDL_factor(n, Apermp, Apermi, Apermx,
//                  Lp, Li, Lx,
//                  D, Dinv, Lnz,
//                  etree, bwork, iwork, fwork);
//
// }





PYBIND11_MODULE(_qdldl, m) {
  m.doc() = "QDLDL low level wrapper";
  py::class_<qdldl::Solver>(m, "Solver")
	  .def(py::init(&py_qdldl_solver))
	  .def("solve", &py_qdldl_solve);


  // m.def("factor", &py_factor
  //         // , py::call_guard<py::gil_scoped_release>()
  //         );
  // m.def("solve", &py_solve);
  // m.def("update", &py_update);
}
