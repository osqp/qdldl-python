from __future__ import print_function
import distutils.sysconfig as sysconfig
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from shutil import copyfile, copy
from glob import glob
import shutil as sh
from subprocess import call, check_output
from platform import system
import os
import sys
from Cython.Build import cythonize

# Add parameters to cmake_args and define_macros
cmake_args = ["-DUNITTESTS=OFF"]
cmake_build_flags = []
lib_subdir = []

# Check if windows linux or mac to pass flag
if system() == 'Windows':
    if sys.version_info.major == 3:
        cmake_args += ['-G', 'Visual Studio 14 2015']
    else:
        cmake_args += ['-G', 'Visual Studio 9 2008']
    # Differentiate between 32-bit and 64-bit
    if sys.maxsize // 2 ** 32 > 0:
        cmake_args[-1] += ' Win64'
    cmake_build_flags += ['--config', 'Release']
    lib_name = 'qdldl.lib'
    lib_subdir = ['Release']

else:  # Linux or Mac
    cmake_args += ['-G', 'Unix Makefiles']
    lib_name = 'libqdldl.a'


# Define qdldl directories
current_dir = os.getcwd()
qdldl_dir = os.path.join(current_dir, 'qdldl')
qdldl_build_dir = os.path.join(qdldl_dir, 'build')

# Interface files
include_dirs = [os.path.join(qdldl_dir,  "include")]

# Set optimizer flag
if system() != 'Windows':
    compile_args = ["-O3"]
else:
    compile_args = []

# Add qdldl compiled library
extra_objects = [os.path.join('module', lib_name)]

# List with OSQP configure files
configure_files = [os.path.join(qdldl_dir, 'qdldl_sources', 'configure', 'qdldl_types.h.in')]

class build_ext_qdldl(build_ext):
    def build_extensions(self):
        # Compile QDLDL using CMake

        # Create build directory
        if os.path.exists(qdldl_build_dir):
            sh.rmtree(qdldl_build_dir)
        os.makedirs(qdldl_build_dir)
        os.chdir(qdldl_build_dir)

        try:
            check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build qdldl")

        # Compile static library with CMake
        call(['cmake'] + cmake_args + ['..'])
        call(['cmake', '--build', '.', '--target', 'qdldlstatic'] +
             cmake_build_flags)

        # Change directory back to the python interface
        os.chdir(current_dir)

        # Copy static library to src folder
        lib_origin = [qdldl_build_dir, 'out'] + lib_subdir + [lib_name]
        lib_origin = os.path.join(*lib_origin)
        print(lib_origin)
        print(os.path.join('module',  lib_name))
        copyfile(lib_origin, os.path.join('module',  lib_name))

        # Run extension
        build_ext.build_extensions(self)


_qdldl = Extension('qdldl._qdldl',
        include_dirs=include_dirs,
        extra_objects=extra_objects,
        sources=['module/_qdldl.pyx'],
        extra_compile_args=compile_args)

_qdldl.cython_directives = {'language_level': "3"} #all are Python-3

packages = ['qdldl',
            'qdldl.tests']


# Read README.rst file
def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='qdldl',
      version='0.1.3',
      author='Bartolomeo Stellato, Paul Goulart, Goran Banjac',
      author_email='bartolomeo.stellato@gmail.com',
      description='QDLDL, a free LDL factorization routine.',
      long_description=readme(),
      package_dir={'qdldl': 'module'},
      include_package_data=True,  # Include package data from MANIFEST.in
      setup_requires=["numpy >= 1.7"],
      install_requires=["numpy >= 1.7", "scipy >= 0.13.2"],
      license='Apache 2.0',
      url="https://github.com/oxfordcontrol/qdldlpy/",
      cmdclass={'build_ext': build_ext_qdldl},
      packages=packages,
      ext_modules=cythonize([_qdldl]),
      zip_safe=False,
      )
