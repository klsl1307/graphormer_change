from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy  # Add this import

ext_modules = [
    Extension("algos", ["algos.pyx"], include_dirs=[numpy.get_include()])  # Specify numpy include path
]

setup(
    ext_modules=cythonize(ext_modules)
)
