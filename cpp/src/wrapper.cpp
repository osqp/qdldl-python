#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "qdldl_wrapper.h"

namespace py = pybind11;


PYBIND11_MODULE(_qdldl, m) {
  m.doc() = "QDLDL low level wrapper";
  m.def("etree", &etree);
  // m.def("factor", &py_factor);
  // m.def("solve", &py_solve);
}
