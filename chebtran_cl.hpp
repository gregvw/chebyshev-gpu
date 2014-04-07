#include <iostream>
#include <vector>
#include <string>

// Throw if there are no compute devices.
#define VEXCL_THROW_ON_EMPTY_CONTEXT
#include <vexcl/vexcl.hpp>

typedef std::vector<double>    dvec;
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

        vex::Context ctx;
        vex::FFT<double, cl_double> fft;
        vex::FFT<cl_double, double> ifft;

        vex::vector<double> catrev(const dvec &a);
};

