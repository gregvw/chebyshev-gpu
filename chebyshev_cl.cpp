
// g++ -std=c++11 -DSTANDALONE_TEST -L/opt/local/lib chebyshev_cl.cpp -lOpenCL -lboost_system
//#include <cstdlib>
#include "chebyshev_cl.hpp"

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

    coeff_to_nodal(k2,w0);
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




void Chebyshev::catrev(const dev_dvec &a, dev_dvec &A2) {

    // First half of A2 holds a:
    slice[vex::range(0,N)](A2) = a;

    // Second half of A2 holds reversed copy of a (with endpoints removed):
    slice[vex::range(N, M)](A2) = vex::permutation( N - 2 - vex::element_index() )(A2);
}



host_dvec Chebyshev::coeff_to_nodal(const host_dvec &a) {

    dev_dvec A(ctx, a);
    dev_dvec B(ctx, N);
    host_dvec b(N);
    coeff_to_nodal(A,B);
    vex::copy(B,b);
    return b;
}



void Chebyshev::coeff_to_nodal(const dev_dvec &a, dev_dvec &b) {

    catrev(a,X2);

    X2[0]   = 2 * a[0];
    X2[N-1] = 2 * a[N-1];

    X2 = fft(X2) / 2;

    b = slice[vex::range(0,N)](X2);
}



// Compute Chebyshev expansion coefficient from grid values
void Chebyshev::nodal_to_coeff(const dev_dvec &b, dev_dvec &a){

    catrev(b, X2);

    X2 = ifft(X2) * 2;

    a = slice[vex::range(0,N)](X2);

    a[0]   = 0.5*a[0];
    a[N-1] = 0.5*a[N-1];

}

host_dvec Chebyshev::nodal_to_coeff(const host_dvec &b) {

    dev_dvec B(ctx, b);
    dev_dvec A(ctx,N);
    host_dvec a(N);
    nodal_to_coeff(B,A);
    vex::copy(A,a);
    return a;
}


// Differentiate a function on the grid
void Chebyshev::nodal_diff(const dev_dvec &u, dev_dvec &v){

    catrev(u,X2);

    // 0,1,...,N-1,-(N-2),...,-1
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
}


host_dvec Chebyshev::nodal_diff(const host_dvec &u) {

    dev_dvec U(ctx, u);
    dev_dvec V(ctx, N);

    host_dvec v(N);

    nodal_diff(U,V);

    vex::copy(V,v);
    return v;
}


void Chebyshev::coeff_int(const dev_dvec &v, dev_dvec &u) {
    VEX_FUNCTION(double, scaled_diff, (size_t, N)(size_t, i)(const double*, v),
            if (i == 0) {
                return 0;
            } else if (i == 1) {
                return v[0] - 0.5 * v[2];
            } else if (i + 1 >= N){
                return 0.5 * v[i - 1] / i;
            } else {
                return 0.5 * (v[i - 1] - v[i + 1]) / i;
            }
            );

    u = scaled_diff(N, vex::element_index(), vex::raw_pointer(v));

}


void Chebyshev::coeff_diff(const dev_dvec &u, dev_dvec &v) {
    host_dvec w(N);
    vex::copy(u, w);
    for(size_t i = 0; i < N; ++i) w[i] *= 2 * i;

    auto V = v.map(0);

    V[N-1] = 0;
    V[N-2] = u[N-1];

    for(int l = 0; l < N-2 ; l += 2) {
        int j = N-l-3;
        V[j] = V[j+2] + w[j+1];
    }

    for(int l = 1; l < N-2 ; l += 2) {
        int j = N-l-3;
        V[j] = V[j+2] + w[j+1];
    }

    V[0] = 0.5*(V[2]+w[1]);

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

//        auto b = cheb.coeff_to_nodal(a);
//        auto bx = cheb.nodal_diff(b);
//        auto c = cheb.nodal_to_coeff(bx);

//        printvector(b);
//        printvector(bx);
        dev_dvec A(a);
        dev_dvec B(N);

//        cheb.coeff_int(A,B);
      cheb.coeff_diff(A,B);
      if (N <= 32) printvector(B);


    } catch (const cl::Error &e) {
        std::cerr << "OpenCL error: " << e << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

}
#endif
