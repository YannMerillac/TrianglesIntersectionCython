import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

ext = [
    Extension(
        "aabb_interface",
        sources=["./aabb_interface.pyx"],
        include_dirs=[np.get_include(), "./eigen-3.4.0"],
        extra_compile_args=["-g"],
        language="c++",
    )
]

setup(ext_modules=cythonize(ext, compiler_directives={"language_level": 3, "profile": False}))