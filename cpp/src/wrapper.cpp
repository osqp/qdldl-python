#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

void init_qdldl_wrapper(py::module &m);

PYBIND11_MODULE(_qdldl, m) {
  m.doc() = "QDLDL low level wrapper";
  init_qdldl_wrapper(m);
}
