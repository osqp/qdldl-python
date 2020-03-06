#include <pybind11/pybind11.h>
// #include <pybind11/factory.h>
#include <pybind11/numpy.h>
#include "qdldl.hpp"

namespace py = pybind11;
// using namespace py::literals; // to bring in the `_a` literal


class PySolver{
	public:
		PySolver(py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast> Ap_py,
				 py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast> Ai_py,
			     py::array_t<QDLDL_float, py::array::c_style | py::array::forcecast> Ax_py);
		py::array solve(py::array_t<QDLDL_float, py::array::c_style | py::array::forcecast> b_py);
		void update(py::array_t<QDLDL_float, py::array::c_style | py::array::forcecast> Anew_x_py);

	private:
		std::unique_ptr<qdldl::Solver> s;

};



PySolver::PySolver(
		py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast> Ap_py,
		py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast> Ai_py,
		py::array_t<QDLDL_float, py::array::c_style | py::array::forcecast> Ax_py){

	QDLDL_int nx = Ap_py.request().size - 1;
	QDLDL_int * Ap = (QDLDL_int *)Ap_py.request().ptr;
	QDLDL_int * Ai = (QDLDL_int *)Ai_py.request().ptr;
	QDLDL_float * Ax = (QDLDL_float *)Ax_py.request().ptr;

	s = std::make_unique<qdldl::Solver>(nx, Ap, Ai, Ax);

}


py::array PySolver::solve(
		const py::array_t<QDLDL_float, py::array::c_style | py::array::forcecast> b_py){

	auto b = (QDLDL_float *)b_py.request().ptr;
	auto x = s->solve(b);

    return py::array(s->nx, x);
}

void PySolver::update(
		const py::array_t<QDLDL_float, py::array::c_style | py::array::forcecast> Anew_x_py){

	auto Anew_x = (QDLDL_float *)Anew_x_py.request().ptr;
	s->update(Anew_x);
}





PYBIND11_MODULE(qdldl, m) {
  m.doc() = "QDLDL wrapper";
  py::class_<PySolver>(m, "PySolver")
	  .def(py::init<py::array_t<QDLDL_int>, py::array_t<QDLDL_int>, py::array_t<QDLDL_float>>())
	  .def("solve", &PySolver::solve)
	  .def("update", &PySolver::update);
  // m.def("factor", &py_factor
  //         // , py::call_guard<py::gil_scoped_release>()
  //         );
}
