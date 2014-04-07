# distutils: language = c++
# distutils: sources = chebtran_cl.cpp

import cython
import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libcpp.vector cimport vector



cdef extern from "chebtran_cl.hpp":
    cdef cppclass ChebyshevTransform:
        ChebyshevTransform(int)
        
        string get_device_name()

        vector[double] coeff_to_nodal(const vector[double] &a)
        vector[double] nodal_to_coeff(const vector[double] &b)



cdef class ChebTran:
    cdef ChebyshevTransform *thisptr
    
#    def get_device_name

    def __init__(self,int n):
        self.thisptr = new ChebyshevTransform(n)

    def __dealloc__(self):
        del self.thisptr

    def get_device_name(self):
        return self.thisptr.get_device_name().split('\n')[0]

    def coeff_to_nodal(self,const vector[double] &a):
        return np.array(self.thisptr.coeff_to_nodal(a))

    def nodal_to_coeff(self,const vector[double] &b):
        return np.array(self.thisptr.nodal_to_coeff(b))

        

