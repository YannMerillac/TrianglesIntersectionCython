import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

ext = [
    Extension(
        "triangle_intersection",
        sources=["./triangle_intersection.pyx"],
        include_dirs=[np.get_include(), "./eigen-3.4.0"],
        extra_compile_args=["-O3", "-mavx", "-ffast-math"],
        language="c++",
    )
]

setup(ext_modules=cythonize(ext, compiler_directives={"language_level": 3, "profile": False}))