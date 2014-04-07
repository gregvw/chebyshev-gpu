#include <iostream>
#include <vector>
#include <vexcl/vexcl.hpp>
#include <string>

typedef std::vector<double> dvec;
typedef std::vector<cl_double> clvec;

class ChebyshevTransform {

    public: 
        ChebyshevTransform(int);
        std::string get_device_name(); 

        dvec coeff_to_nodal(const dvec &a);

        dvec nodal_to_coeff(const dvec &b);


    private:
        int N;
        int M;

        std::shared_ptr< vex::Context > ctx;
        std::shared_ptr< vex::FFT<double, cl_double> > fft;
        std::shared_ptr< vex::FFT<cl_double,double> > ifft;
   

};

