#include <pybind11/pybind11.h>
// #include <pybind11/factory.h>
#include <pybind11/numpy.h>
#include "qdldl.hpp"

namespace py = pybind11;
using namespace py::literals; // to bring in the `_a` literal


class PySolver{
	public:
		PySolver(py::object A, const bool upper);
		py::array solve(py::array_t<QDLDL_float, py::array::c_style | py::array::forcecast> b_py);
		void update(py::object Anew_py, const bool upper);
		py::tuple factors();

	private:
		std::unique_ptr<qdldl::Solver> s;

};


py::array PySolver::solve(
		const py::array_t<QDLDL_float, py::array::c_style | py::array::forcecast> b_py){

	auto b = (QDLDL_float *)b_py.data();

	if ((QDLDL_int)b_py.size() != this->s->nx)
		throw py::value_error("Length of b does not match size of A");

	py::gil_scoped_release release;
	auto x = s->solve(b);
    py::gil_scoped_acquire acquire;

	py::array x_py = py::array(s->nx, x);

	delete [] x;

    return x_py;
}

void PySolver::update(py::object Anew, const bool upper=false){

	py::object spa = py::module::import("scipy.sparse");

	if (!spa.attr("isspmatrix_csc")(Anew)) Anew = spa.attr("csc_matrix")(Anew);

	py::object Anew_triu;
	if (upper){
		Anew_triu = Anew;
	} else {
		Anew_triu = spa.attr("triu")(Anew, "format"_a="csc");
	}

	auto Anew_x_py = Anew_triu.attr("data").cast<py::array_t<QDLDL_float>>();

	auto Anew_x = (QDLDL_float *)Anew_x_py.data();

	py::gil_scoped_release release;
	s->update(Anew_x);
    py::gil_scoped_acquire acquire;
}

py::tuple PySolver::factors() {

	py::object spa = py::module::import("scipy.sparse");

    QDLDL_int n = s->nx;
    QDLDL_int Lnz = s->sum_Lnz;
    py::array Lp = py::array(n+1, s->get_Lp());
    py::array Li = py::array(Lnz, s->get_Li());
    py::array Lx = py::array(Lnz, s->get_Lx());

    auto L = spa.attr("csc_matrix")(
         py::make_tuple(Lx, Li, Lp), py::make_tuple(n, n)
    );

    py::array D = py::array(n, s->get_D());
    py::array P = py::array(n, s->get_P());

    return py::make_tuple(L, D, P);
}


PySolver::PySolver(py::object A, const bool upper=false){

	// Use scipy to convert to upper triangular and get data
	py::object spa = py::module::import("scipy.sparse");

	// Check dimensions
	py::tuple dim = A.attr("shape");
	int m = dim[0].cast<int>();
	int n = dim[1].cast<int>();

	if (m != n) throw py::value_error("Matrix A is not square");

	if (!spa.attr("isspmatrix_csc")(A)) A = spa.attr("csc_matrix")(A);

	if (A.attr("nnz").cast<int>() == 0) throw py::value_error("Matrix A is empty");

    py::object A_triu;
	if (upper){
		A_triu = A;  // Already in upper-triangular format
	} else {
		A_triu = spa.attr("triu")(A, "format"_a="csc");
	}

	auto Ap_py = A_triu.attr("indptr").cast<py::array_t<QDLDL_int, py::array::c_style>>();
	auto Ai_py = A_triu.attr("indices").cast<py::array_t<QDLDL_int, py::array::c_style>>();
	auto Ax_py = A_triu.attr("data").cast<py::array_t<QDLDL_float, py::array::c_style>>();

	QDLDL_int nx = Ap_py.request().size - 1;
	QDLDL_int * Ap = (QDLDL_int *)Ap_py.data();
	QDLDL_int * Ai = (QDLDL_int *)Ai_py.data();
	QDLDL_float * Ax = (QDLDL_float *)Ax_py.data();

	py::gil_scoped_release release;
	// TODO: Replace this line with the make_unique line below in the future.
	// It needs C++14 but manylinux does not support it yet
	this->s = std::unique_ptr<qdldl::Solver>(new qdldl::Solver(nx, Ap, Ai, Ax));
	// s = std::make_unique<qdldl::Solver>(nx, Ap, Ai, Ax);
	py::gil_scoped_acquire acquire;
}




PYBIND11_MODULE(qdldl, m) {
  m.doc() = "QDLDL wrapper";
  py::class_<PySolver>(m, "Solver")
	  .def(py::init<py::object, bool>(), py::arg("A"), py::arg("upper") = false)
	  .def("solve", &PySolver::solve)
	  .def("update", &PySolver::update, py::arg("Anew"), py::arg("upper") = false)
	  .def("factors", &PySolver::factors, R"delim(
            factors returns a sparse n x n matrix L, a n-array d and a list of
            indexes p that represent the decomposition of A.

            Specifically,
            A == P @ (spa.eye(n) + L) @ spa.diags(d)  @ (spa.eye(n) + L).T @ P.T
            where P is the matrix given by
            P = spa.dok_matrix((n, n))
            P[p, np.arange(n)] = 1.0
            P = P.tocsr()
)delim");
}
