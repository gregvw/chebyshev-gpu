// g++ -std=c++11 -DSTANDALONE_TEST -L/opt/local/lib chebtran_cl.cpp -lOpenCL -lboost_system
//#include <cstdlib>
#include "chebtran_cl.hpp"


ChebyshevTransform::ChebyshevTransform(int n)
    : N(n), M(2 * N - 2),
      ctx(
            vex::Filter::Type(CL_DEVICE_TYPE_GPU) &&
            vex::Filter::DoublePrecision &&
            vex::Filter::Count(1)
         ),
      fft(ctx, M), ifft(ctx, M, vex::fft::inverse),
      X2(ctx, M)
{ }



std::string ChebyshevTransform::get_device_name() {

    std::ostringstream os;

    // Write device name to output string stream
    os << ctx.queue(0);

    // extract string
    std::string devName = os.str();

    return devName;
}

// Concatenate input vector with its reversed self,
// copy the result to compute device
void ChebyshevTransform::catrev(const dvec &a, vex::vector<double> &A2) {
    // First half of A2 holds a:
    vex::copy(a.begin(), a.end(), A2.begin());

    // Second half of A2 holds reversed copy of a (with endpoints removed):
    vex::slicer<1> slice(vex::extents[M]);
    slice[vex::range(N, M)](A2) = vex::permutation( N - 2 - vex::element_index() )(A2);
}

// Evaluate real-valued Chebyshev expansions on the grid
dvec ChebyshevTransform::coeff_to_nodal(const dvec &a) {

    catrev(a, X2);

    X2[0]   = 2 * a[0];
    X2[N-1] = 2 * a[N-1];

    X2 = fft(X2) / 2;

    dvec b(N);

    vex::copy(X2.begin(), X2.begin() + N, b.begin());

    return b;
}


// Compute Chebyshev expansion coefficient from grid values
dvec ChebyshevTransform::nodal_to_coeff(const dvec &b){

    catrev(b, X2);

    X2 = ifft(X2) * 2;

    dvec a(N);

    vex::copy(X2.begin(), X2.begin() + N, a.begin());

    a[0]   *= 0.5;
    a[N-1] *= 0.5;

    return a;
}



#ifdef STANDALONE_TEST
int main(int argc, char* argv[]) {

    int N = argc > 1 ? atoi(argv[1]) : 16;
    int k = argc > 2 ? atoi(argv[2]) : 0;

    try {
        ChebyshevTransform chebtran(N);
        std::cout << chebtran.get_device_name() << std::endl;

        dvec a(N,0);
        a[k] = 1;

        auto b = chebtran.coeff_to_nodal(a);
        auto c = chebtran.nodal_to_coeff(b);

        for(int i=0;i<c.size();++i) {
            std::cout << c[i] << std::endl;
        }
    } catch (const cl::Error &e) {
        std::cerr << "OpenCL error: " << e << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

}
#endif

