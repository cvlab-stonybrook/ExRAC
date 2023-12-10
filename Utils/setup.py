from setuptools import setup
from Cython.Build import cythonize
from setuptools.command.build_ext import build_ext

class BuildExt(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)
        self.build_lib = '../Models/Feature/'

setup(
    cmdclass={'build_ext': BuildExt},
    ext_modules=cythonize("ExemplarDetect.pyx"),
)