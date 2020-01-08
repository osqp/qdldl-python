#include "qdldl_wrapper.h"


int etree(const py::array_t<QDLDL_int, py::array::c_style | py::array::forcecast> Ap_py,
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

    // TODO: Raise error
    // if (sum_Lnz < 0) raise ValueError("Input matrix is not quasi-definite");

    return sum_Lnz;



}

