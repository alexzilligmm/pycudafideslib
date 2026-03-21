// main.cu – same CLI interface as the original Go version.
//
// Usage (same flags):
//   ./cuda_cachemir -test Ops     -logN 12
//   ./cuda_cachemir -test Decoder -logN 12 -hidDim 256 -expDim 1024
//   ./cuda_cachemir -test Model   -logN 16 -hidDim 4096 -expDim 16384 -seqLen 512

#include "llama.h"
#include <cmath>
#include <iostream>
#include <string>
#include <stdexcept>

Ctx bootstrap_to(const LlamaInference&, const Ctx&, uint32_t);

struct Flags {
    int logN=12; std::string test="Decoder"; int level=5; int btpLevel=15;
    int hidDim=256; int expDim=1024; int seqLen=512; int numHeads=32;
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
        else if (s=="-expDim"||s=="--expDim")      f.expDim   = std::stoi(nxt());
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

    LlamaInference llama = make_llama(f.logN, f.hidDim, f.expDim,
                                       f.seqLen, f.numHeads, f.parallel);
    std::cout << "Initialization finished! slots=" << llama.slots << "\n";

    // Random test input
    std::vector<double> msg_in(llama.slots);
    for (int i=0;i<llama.slots;++i) msg_in[i]=-2.0+4.0*i/llama.slots;
    Ctx x = llama.cc()->Encrypt(llama.fhe->pk(),
                                 llama.cc()->MakeCKKSPackedPlaintext(msg_in));

    const std::string& T = f.test;

    if (T=="QKV") {
        prepare_weights(llama,{"q","k","v"});
        Timer t; qkv_q(llama,x); qkv_k(llama,x); qkv_v(llama,x);
        std::cout<<"QKV: "<<t.elapsed_s()<<" s\n";
    } else if (T=="RoPE") {
        prepare_weights(llama,{"RoPE"}); Timer t; rope(llama,x,x);
        std::cout<<"RoPE: "<<t.elapsed_s()<<" s\n";
    } else if (T=="Cache") {
        prepare_cache(llama,{"k","v","mask"}); Timer t; cache_kv(llama,x,x);
        std::cout<<"Cache: "<<t.elapsed_s()<<" s\n";
    } else if (T=="QK_T") {
        prepare_cache(llama,{"k"}); Timer t; qk_transpose(llama,x);
        std::cout<<"QK_T: "<<t.elapsed_s()<<" s\n";
    } else if (T=="AttnV") {
        prepare_cache(llama,{"v"}); Timer t; attn_v(llama,x);
        std::cout<<"AttnV: "<<t.elapsed_s()<<" s\n";
    } else if (T=="Out") {
        prepare_weights(llama,{"out"}); Timer t; out_proj(llama,x);
        std::cout<<"Out: "<<t.elapsed_s()<<" s\n";
    } else if (T=="UpGate") {
        prepare_weights(llama,{"up","gate"}); Timer t; up_gate(llama,x);
        std::cout<<"UpGate: "<<t.elapsed_s()<<" s\n";
    } else if (T=="Down") {
        prepare_weights(llama,{"down"}); Timer t; down_proj(llama,x);
        std::cout<<"Down: "<<t.elapsed_s()<<" s\n";
    } else if (T=="SiLU") {
        auto ref=ref_silu(msg_in); Timer t;
        auto r=decrypt(llama.cc(),silu(llama,x),llama.fhe->sk());
        std::cout<<"SiLU: "<<t.elapsed_s()<<" s  MSE="<<mse(r,ref)<<"\n";
    } else if (T=="Softmax") {
        auto ref=ref_softmax(msg_in); Timer t;
        auto r=decrypt(llama.cc(),softmax(llama,x,f.btpLevel,0),llama.fhe->sk());
        std::cout<<"Softmax: "<<t.elapsed_s()<<" s  MSE="<<mse(r,ref)<<"\n";
    } else if (T=="Norm") {
        auto ref=ref_norm(msg_in,f.hidDim); Timer t;
        auto r=decrypt(llama.cc(),norm(llama,x,f.btpLevel),llama.fhe->sk());
        std::cout<<"Norm: "<<t.elapsed_s()<<" s  MSE="<<mse(r,ref)<<"\n";
    } else if (T=="Argmax") {
        Timer t; argmax(llama,x);
        std::cout<<"Argmax: "<<t.elapsed_s()<<" s\n";
    } else if (T=="CtMult") {
        const int N=100; Timer t;
        for (int i=0;i<N;++i) llama.cc()->EvalMult(x,x);
        std::cout<<"CtMult avg: "<<t.elapsed_s()/N<<" s (level "<<f.level<<")\n";
    } else if (T=="Ops") {
        const int N=100;
        std::vector<Ctx> cts(N); for(auto& c:cts) c=x;
        std::vector<Ptx> pts(N);
        for(auto& p:pts) p=llama.cc()->MakeCKKSPackedPlaintext(msg_in);
        { Timer t; for(int i=0;i<N;++i) llama.cc()->EvalAdd(cts[i],cts[i]);
          std::cout<<"Add: "<<t.elapsed_s()/N<<" s\n"; }
        { Timer t; for(int i=0;i<N;++i) llama.cc()->EvalMult(cts[i],cts[i]);
          std::cout<<"CtMul: "<<t.elapsed_s()/N<<" s\n"; }
        { Timer t; for(int i=0;i<N;++i) llama.cc()->EvalMult(cts[i],pts[i]);
          std::cout<<"CtPtMul: "<<t.elapsed_s()/N<<" s\n"; }
        { Timer t; for(int i=0;i<N;++i) llama.cc()->EvalRotate(cts[i],5);
          std::cout<<"Rotate: "<<t.elapsed_s()/N<<" s\n"; }
        { Timer t; llama.cc()->EvalBootstrap(x);
          std::cout<<"Bootstrap: "<<t.elapsed_s()<<" s\n"; }
    } else if (T=="Decoder") {
        prepare_weights(llama,{"q","k","v","out","up","gate","down","RoPE"});
        prepare_cache  (llama,{"k","v","mask"});
        Timer t; decoder(llama,x);
        std::cout<<"Decoder: "<<t.elapsed_s()<<" s\n";
    } else if (T=="Model") {
        prepare_weights(llama,{"q","k","v","out","up","gate","down","RoPE"});
        prepare_cache  (llama,{"k","v","mask"});
        model(llama,x);
    } else {
        std::cerr<<"Unknown test: "<<T<<"\n"; return 1;
    }
    return 0;
}
