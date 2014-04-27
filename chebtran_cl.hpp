#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <boost/math/constants/constants.hpp>

// Throw if there are no compute devices.
#define VEXCL_THROW_ON_EMPTY_CONTEXT
#include <vexcl/vexcl.hpp>

typedef vex::vector<double> dev_dvec;
typedef std::vector<double> host_dvec;

const double pi = boost::math::constants::pi<double>();

class Chebyshev {

    public:
        Chebyshev(int);

        std::string get_device_name();

        dev_dvec coeff_to_nodal(const dev_dvec &a);
        host_dvec coeff_to_nodal(const host_dvec &a);

        dev_dvec nodal_to_coeff(const dev_dvec &b);
        host_dvec nodal_to_coeff(const host_dvec &b);

        dev_dvec nodal_diff(const dev_dvec &u);
        host_dvec nodal_diff(const host_dvec &u);


    private:
        int N;
        int M;

        vex::Context ctx;

        vex::slicer<1> slice;
        vex::Reductor<double,vex::SUM> sum;
        vex::FFT<double, double> fft;
        vex::FFT<double, double> ifft;
        vex::FFT<cl_double2, cl_double2> cplx_ifft;

        dev_dvec X2;

        dev_dvec kkrev;

        dev_dvec w0;
        dev_dvec wi;
        dev_dvec wN;

        void catrev(const dev_dvec &a, dev_dvec &X2);

        // Copy the first N elements of a to b
        void copy_subvector(const dev_dvec &a, dev_dvec &b, int start, int stop);


};
