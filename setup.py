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
import distutils
import platform


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        try:
            import pybind11
            pybind11
        except ImportError:
            if call([sys.executable, '-m', 'pip', 'install', 'pybind11']):
                raise RuntimeError('pybind11 install failed.')
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


cmake_args = []
# What variables from the environment do we wish to pass on to cmake as variables?
cmake_env_vars = ('CMAKE_GENERATOR', 'CMAKE_GENERATOR_PLATFORM', 'CMAKE_OSX_ARCHITECTURES')
for cmake_env_var in cmake_env_vars:
    cmake_var = os.environ.get(cmake_env_var)
    if cmake_var:
        cmake_args.extend([f'-D{cmake_env_var}={cmake_var}'])

# Add parameters to cmake_args and define_macros
cmake_args += ["-DQDLDL_UNITTESTS=OFF"]
cmake_build_flags = []
lib_subdir = []

# Check if windows linux or mac to pass flag
if system() == 'Windows':
    cmake_build_flags += ['--config', 'Release']
    lib_name = 'qdldlamd.lib'
    lib_subdir = ['Release']

else:  # Linux or Mac
    lib_name = 'libqdldlamd.a'

# Set optimizer flag
if system() != 'Windows':
    compile_args = ["-O3"]
else:
    compile_args = []

# Compile QDLDL using CMake
current_dir = os.getcwd()
qdldl_dir = os.path.join(current_dir, 'c',)
qdldl_build_dir = os.path.join(qdldl_dir, 'build')
qdldl_lib = [qdldl_build_dir, 'out'] + lib_subdir + [lib_name]
qdldl_lib = os.path.join(*qdldl_lib)


class build_ext_qdldl(build_ext):
    def build_extensions(self):

        # Create build directory
        if not os.path.exists(qdldl_build_dir):
            os.makedirs(qdldl_build_dir)
        os.chdir(qdldl_build_dir)

        try:
            check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build qdldl")

        # Compile static library with CMake
        call(['cmake'] + cmake_args + ['..'])
        call(['cmake', '--build', '.', '--target', 'qdldlamd'] +
             cmake_build_flags)

        # Change directory back to the python interface
        os.chdir(current_dir)

        # Run extension
        build_ext.build_extensions(self)


qdldl = Extension('qdldl',
                  sources=glob(os.path.join('cpp', '*.cpp')),
                  include_dirs=[os.path.join('c'),
                                os.path.join('c', 'qdldl', 'include'),
                                get_pybind_include(),
                                get_pybind_include(user=False)],
                  language='c++',
                  extra_compile_args=compile_args + ['-std=c++11'],
                  extra_objects=[qdldl_lib])


def readme():
    with open('README.md') as f:
        return f.read()


setup(name='qdldl',
      version='0.1.7.post4',
      author='Bartolomeo Stellato, Paul Goulart, Goran Banjac',
      author_email='bartolomeo.stellato@gmail.com',
      description='QDLDL, a free LDL factorization routine.',
      long_description=readme(),
      long_description_content_type='text/markdown',
      package_dir={'qdldl': 'module'},
      include_package_data=True,  # Include package data from MANIFEST.in
      setup_requires=["setuptools>=18.0", "pybind11"],
      install_requires=["numpy >= 1.7", "scipy >= 0.13.2"],
      license='Apache 2.0',
      url="https://github.com/oxfordcontrol/qdldl-python/",
      cmdclass={'build_ext': build_ext_qdldl},
      ext_modules=[qdldl],
      )
