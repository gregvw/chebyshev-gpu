
// g++ -std=c++11 -DSTANDALONE_TEST -L/opt/local/lib cheb_cl.cpp -lOpenCL -lboost_system
//#include <cstdlib>
#include "chebtran_cl.hpp"

// Print elements of vector to stdout
template<class T>
void printvector(const T &v){
    for(int i=0;i<v.size();++i) {
        std::cout << v[i] << std::endl;
    }
    std::cout << std::endl;
}


Chebyshev::Chebyshev(int n)
    : N(n), M(2 * N - 2),
      ctx(
            vex::Filter::Type(CL_DEVICE_TYPE_GPU) &&
            vex::Filter::DoublePrecision &&
            vex::Filter::Count(1)
         ),
      fft(ctx, M), ifft(ctx, M, vex::fft::inverse),
      cplx_ifft(ctx, M, vex::fft::inverse),
      sum(ctx),
      slice(vex::extents[M]),
      X2(ctx, M),
      w0(ctx, N),
      wi(ctx, N-2),
      wN(ctx, N)
{
    dev_dvec k2(ctx, N);

    auto i = vex::tag<1>(vex::element_index());

    k2 = 2 * i * i;

    // 2,8,18,...,2*(N-2)^2,(N-1)^2
    k2[N - 1] = (N - 1) * (N - 1);

    w0 = coeff_to_nodal(k2);
    w0 = w0 / (N - 1);

    w0[0]     = 0.5 * w0[0];
    w0[N - 1] = 0.5 * w0[N - 1];

    wN = -vex::permutation(N - 1 - i)(w0);
    wi = 1 / ( sin(vex::constants::pi() * (i + 1) / (N - 1)) );
}




std::string Chebyshev::get_device_name() {

    std::ostringstream os;

    // Write device name to output string stream
    os << ctx.queue(0);

    // extract string
    std::string devName = os.str();

    return devName;
}


void Chebyshev::copy_subvector(const dev_dvec &a, dev_dvec &b, int start, int stop) {
    b = slice[vex::range(start,stop)](a);
}


void Chebyshev::catrev(const dev_dvec &a, dev_dvec &A2) {

    // First half of A2 holds a:
    slice[vex::range(0,N)](A2) = a;

    // Second half of A2 holds reversed copy of a (with endpoints removed):
    slice[vex::range(N, M)](A2) = vex::permutation( N - 2 - vex::element_index() )(A2);
}



host_dvec Chebyshev::coeff_to_nodal(const host_dvec &a) {

    dev_dvec A(ctx, a);
    host_dvec b(N);
    auto B = coeff_to_nodal(A);
    vex::copy(B,b);
    return b;
}



dev_dvec Chebyshev::coeff_to_nodal(const dev_dvec &a) {

    catrev(a,X2);

    X2[0]   = 2 * a[0];
    X2[N-1] = 2 * a[N-1];

    X2 = fft(X2) / 2;

    dev_dvec b(ctx, N);

    copy_subvector(X2,b,0,N);

    return b;
}



// Compute Chebyshev expansion coefficient from grid values
dev_dvec Chebyshev::nodal_to_coeff(const dev_dvec &b){

    catrev(b, X2);

    X2 = ifft(X2) * 2;

    dev_dvec a(ctx, N);

    copy_subvector(X2,a,0,N);

    a[0]   = 0.5*a[0];
    a[N-1] = 0.5*a[N-1];

    return a;
}

host_dvec Chebyshev::nodal_to_coeff(const host_dvec &b) {

    dev_dvec B(ctx, b);
    host_dvec a(N);
    auto A = nodal_to_coeff(B);
    vex::copy(A,a);
    return a;
}


// Differentiate a function on the grid
dev_dvec Chebyshev::nodal_diff(const dev_dvec &u){

    dev_dvec v(ctx, N);

    catrev(u,X2);

    // 0,1,...,N-1,-(N-1),-(N-2),...,-1
    VEX_FUNCTION(double, kkrev, (ptrdiff_t, N)(ptrdiff_t, i),
            if (i < N) return i;
            return -2 * N + i + 2;
            );

    X2 = fft(X2) * kkrev(N, vex::element_index());
    X2[N-1] = 0;

    // Extract imaginary part of vector
    VEX_FUNCTION(double, imag, (cl_double2, c),
            return c.y;
            );

    X2 = imag(cplx_ifft(X2));

    v[0] = sum(u*w0);

    slice[vex::range(1,N-1)](v) = slice[vex::range(1,N-1)](X2) * wi;

    v[N-1] = sum(wN*u);

    return v;
}

host_dvec Chebyshev::nodal_diff(const host_dvec &u) {

    dev_dvec U(ctx, u);

    host_dvec v(N);

    auto V = nodal_diff(U);

    vex::copy(V,v);
    return v;
}




#ifdef STANDALONE_TEST
int main(int argc, char* argv[]) {

    int N = argc > 1 ? atoi(argv[1]) : 16;
    int k = argc > 2 ? atoi(argv[2]) : 0;

    try {
        Chebyshev cheb(N);
        std::cout << cheb.get_device_name() << std::endl;

        host_dvec a(N,0);
        a[k] = 1;

        dev_dvec A(a);

        auto b = cheb.coeff_to_nodal(a);
        auto bx = cheb.nodal_diff(b);
        auto c = cheb.nodal_to_coeff(bx);

        printvector(b);
        printvector(bx);

    } catch (const cl::Error &e) {
        std::cerr << "OpenCL error: " << e << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

}
#endif
