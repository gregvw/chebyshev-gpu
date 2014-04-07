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
      fft(ctx, M), ifft(ctx, M, vex::fft::inverse)
{ }



std::string ChebyshevTransform::get_device_name() {

    std::ostringstream os;

    // Write device name to output string stream
    os << ctx.queue(0);

    // extract string
    std::string devName = os.str();

    return devName;
}


// Evaluate real-valued Chebyshev expansions on the grid
dvec ChebyshevTransform::coeff_to_nodal(const dvec &a) {

    // Create a reversed copy of the input coefficients
    dvec arev(a);
    std::reverse(arev.begin(),arev.end());

    dvec a2;
    a2.reserve(M);

    a2.insert(a2.begin(),a.begin(),a.end());
    a2.insert(a2.begin()+N,arev.begin()+1,arev.end()-1);

    a2[0] *= 2;
    a2[N-1] *= 2;

    vex::vector<double> A2(ctx,a2);
    vex::vector<cl_double> B2(ctx,M);

    B2 = fft(A2);
    B2 /= 2;

    dvec b(N);

    vex::copy(B2.begin(),B2.begin()+N,b.begin());

    return b;
}


// Compute Chebyshev expansion coefficient from grid values
dvec ChebyshevTransform::nodal_to_coeff(const dvec &b){

    // Create a reversed copy of the input coefficients
    dvec brev(b);
    std::reverse(brev.begin(),brev.end());

    dvec b2;
    b2.reserve(M);

    b2.insert(b2.begin(),b.begin(),b.end());
    b2.insert(b2.begin()+N,brev.begin()+1,brev.end()-1);

    vex::vector<cl_double> B2(ctx,b2);
    vex::vector<double> A2(ctx,M);

    A2 = ifft(B2);
    A2 *= 2;

    dvec a(N);

    vex::copy(A2.begin(),A2.begin()+N,a.begin());

    a[0] *= 0.5;
    a[N-1] *= 0.5;

    return a;
}



#ifdef STANDALONE_TEST
int main(int argc, char* argv[]) {

    int N = argc > 1 ? atoi(argv[1]) : 16;
    int k = argc > 2 ? atoi(argv[2]) : 0;

    ChebyshevTransform chebtran(N);
    std::cout << chebtran.get_device_name() << std::endl;

    dvec a(N,0);
    a[k] = 1;

    auto b = chebtran.coeff_to_nodal(a);
    auto c = chebtran.nodal_to_coeff(b);

    for(int i=0;i<c.size();++i) {
        std::cout << c[i] << std::endl;
    }

}
#endif

