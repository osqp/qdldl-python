#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "qdldl.h"

namespace py = pybind11;

int etree(const QDLDL_int n,
     const py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast> Ap,
     const py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast> Ai,
     py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast> iwork,
     py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast> Lnz,
     py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast> etree);




