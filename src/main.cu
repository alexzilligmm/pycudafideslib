// main.cu – same CLI interface as the original Go version.
//
// Usage (same flags):
//   ./cuda_cachemir -test Ops     -logN 12
//   ./cuda_cachemir -test Decoder -logN 12 -hidDim 256 -ffDim 1024
//   ./cuda_cachemir -test Model   -logN 16 -hidDim 4096 -ffDim 16384 -seqLen 512

#include "llama.h"
#include <cmath>
#include <iostream>
#include <string>
#include <stdexcept>

struct Flags {
    int logN=12; std::string test="Decoder"; int level=5; int btpLevel=15;
    int hidDim=256; int ffDim=1024; int seqLen=512; int numHeads=32;
    bool parallel=true;
};

static Flags parse(int argc, char** argv) {
    Flags f;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto nxt = [&]() { if (++i >= argc) throw std::runtime_error("Missing value for "+s); return std::string(argv[i]); };
        if      (s=="-logN"||s=="--logN")         f.logN     = std::stoi(nxt());
        else if (s=="-test"||s=="--test")          f.test     = nxt();
        else if (s=="-level"||s=="--level")        f.level    = std::stoi(nxt());
        else if (s=="-btpLevel"||s=="--btpLevel")  f.btpLevel = std::stoi(nxt());
        else if (s=="-hidDim"||s=="--hidDim")      f.hidDim   = std::stoi(nxt());
        else if (s=="-ffDim"||s=="--ffDim"||s=="-expDim"||s=="--expDim") f.ffDim = std::stoi(nxt());
        else if (s=="-seqLen"||s=="--seqLen")      f.seqLen   = std::stoi(nxt());
        else if (s=="-numHeads"||s=="--numHeads")  f.numHeads = std::stoi(nxt());
        else if (s=="-parallel"||s=="--parallel")  f.parallel = (nxt()!="false");
    }
    return f;
}

static double mse(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0; for (size_t i = 0; i < a.size(); ++i) { double d=a[i]-b[i]; s+=d*d; }
    return std::sqrt(s/a.size());
}
static std::vector<double> ref_silu(const std::vector<double>& x) {
    std::vector<double> y(x.size());
    for (size_t i=0;i<x.size();++i) y[i]=x[i]/(std::exp(-x[i])+1);
    return y;
}
static std::vector<double> ref_softmax(const std::vector<double>& x) {
    std::vector<double> y(x.size()), ex(x.size()); std::vector<double> s(8,0);
    for (size_t i=0;i<x.size();++i){ex[i]=std::exp(x[i]);s[i%8]+=ex[i];}
    for (size_t i=0;i<x.size();++i) y[i]=ex[i]/s[i%8];
    return y;
}
static std::vector<double> ref_norm(const std::vector<double>& x, int hD) {
    int n=x.size(); std::vector<double> y(n);
    for (int g=0;g<n/hD;++g){
        double m=0,v=0;
        for (int j=0;j<hD;++j) m+=x[g*hD+j]; m/=hD;
        for (int j=0;j<hD;++j){double d=x[g*hD+j]-m;v+=d*d;} v/=hD;
        double s=1.0/std::sqrt(v+1e-8);
        for (int j=0;j<hD;++j) y[g*hD+j]=(x[g*hD+j]-m)*s;
    }
    return y;
}

int main(int argc, char** argv) {
    Flags f = parse(argc, argv);

    Inference llama = make_llama(f.logN, f.hidDim, f.ffDim,
                                       f.seqLen, f.numHeads, f.parallel);
    std::cout << "Initialization finished! slots=" << llama.slots << "\n";

    // Random test input
    std::vector<double> msg_in(llama.slots);
    for (int i=0;i<llama.slots;++i) msg_in[i]=-2.0+4.0*i/llama.slots;
    Ptx pt_in = llama.cc()->MakeCKKSPackedPlaintext(msg_in);
    Ctx x = llama.cc()->Encrypt(llama.fhe->pk(), pt_in);

    const std::string& T = f.test;

    std::cout << "No tests for now " << "\n";
    return 0;
}
