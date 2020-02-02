# qdldl-python

[![image](https://travis-ci.org/oxfordcontrol/qdldl-python.svg?branch=master)](https://travis-ci.org/oxfordcontrol/qdldl-python)

Python interface to the [QDLDL](https://github.com/oxfordcontrol/qdldl/)
free LDL factorization routine for quasi-definite linear systems: `Ax =
b`.

## Installation

This package can be directly installed via pip,

```
pip install qdldl
```

## Usage

Initialize the factorization with

```
import qdldl
F = qdldl.factor(A)
```

where `A` must be a square quasi-definite matrix in [scipy sparse CSC
format](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html/).

To solve the linear system for a right-hand side `b`, just write

```
x = F.solve(b)
```

##  TODO

- [ ] Implement C++ object storing factorization
- [ ] Wrap object in pybind11
- [ ] Test functions + updates
- [ ] Test GIL release
