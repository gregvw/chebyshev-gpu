# setup.py

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import os

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

setup( name = 'chebtran',
       version = '0.1',
       author = 'Greg von Winckel',
       author_email = 'greg.von.winckel@gmail.com',
       url = 'http://www.scientificpython.net',
       description = 'Fast Chebyshev Transform on the GPU using VexCL',
       ext_modules=[Extension("chebtran_cl",
                        sources=["chebtran.pyx","chebtran_cl.cpp"],
                        extra_compile_args=["-std=c++11"],
                        extra_link_args=["-lOpenCL","-lboost_system"],
                        language="c++",
                        include_dirs=[numpy.get_include()],
                        library_dirs=["/opt/local/lib"])
       ],
       cmdclass = {'build_ext': build_ext},
)       
