#include <iostream>
#include <vector>
#include <memory>
#include <string>

// Throw if there are no compute devices.
#define VEXCL_THROW_ON_EMPTY_CONTEXT
#include <vexcl/vexcl.hpp>

typedef vex::vector<double> dev_dvec;
typedef std::vector<double> host_dvec;

class Chebyshev {

    public:
        Chebyshev(int);

        std::string get_device_name();

        void coeff_to_nodal(const dev_dvec &a, dev_dvec &b);
        host_dvec coeff_to_nodal(const host_dvec &a);

        void nodal_to_coeff(const dev_dvec &b, dev_dvec &a);
        host_dvec nodal_to_coeff(const host_dvec &b);

        void nodal_diff(const dev_dvec &u, dev_dvec &v);
        host_dvec nodal_diff(const host_dvec &u);

        void coeff_int(const dev_dvec &v, dev_dvec &u);

        void coeff_diff(const dev_dvec &u, dev_dvec &v);
         
    private:
        int N;
        int M;

        vex::Context ctx;

        vex::FFT<double, double> fft;
        vex::FFT<double, double> ifft;
        vex::FFT<cl_double2, cl_double2> cplx_ifft;
        vex::Reductor<double,vex::SUM> sum;
        vex::slicer<1> slice;

        dev_dvec X2;

        dev_dvec w0;
        dev_dvec wi;
        dev_dvec wN;

        void catrev(const dev_dvec &a, dev_dvec &X2);

};
