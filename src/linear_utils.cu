#include "gpt2.h"
#include <fstream>
#include <stdexcept>

int interleave_idx(int m, int d, int dim) {
    int a = (dim > d) ? (dim / d) : 1;
    return (m / a + (m % a) * d) % dim;
}

CacheMirParams compute_cm_params(int N, int d_in, int d_out) {
    CacheMirParams p;
    p.is_up  = (d_in <= d_out);
    p.d      = p.is_up ? d_in : d_out;
    p.alpha  = std::max(d_in, d_out) / p.d;
    p.t      = N / p.d;
    p.tp     = N / (p.alpha * p.d);
    p.tp_in  = p.is_up ? p.t  : p.tp;
    p.tp_out = p.is_up ? p.tp : p.t;
    int d_   = p.is_up ? p.d : p.alpha * p.d;
    p.n_pt   = d_ / p.tp_out;
    p.r_i    = std::max(1, p.d * p.d / N);
    p.r_i    = std::min(p.r_i, p.n_pt);
    p.r_o    = p.n_pt / p.r_i;
    return p;
}

std::vector<double> load_matrix_txt(const std::string& path, int d_in, int d_out) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);
    std::vector<double> vals;
    vals.reserve(d_in * d_out);
    double v;
    while (f >> v) vals.push_back(v);
    if ((int)vals.size() != d_in * d_out)
        throw std::runtime_error(path + ": got " + std::to_string(vals.size())
                                 + ", expected " + std::to_string(d_in * d_out));
    return vals;
}

Ctx encode_linear_input(Inference& inf, const std::vector<double>& x, int d_in, int d_out) {
    auto p  = compute_cm_params(inf.slots, d_in, d_out);
    int N   = inf.slots;
    int d_x = p.is_up ? p.d : p.alpha * p.d;
    int M   = N / p.tp;
    std::vector<double> ptx(N, 0.0);
    if (p.is_up)
        for (int i = 0; i < p.d; ++i) ptx[i * p.t] = x[i];
    else
        for (int m = 0; m < M; ++m) ptx[m * p.tp] = x[interleave_idx(m, p.d, d_x)];
    return encrypt(inf.cc(), inf.cc()->MakeCKKSPackedPlaintext(ptx), inf.fhe->pk());
}

std::vector<Ptx> encode_weight_matrix(Inference& inf, const std::vector<double>& vals,
                                       int d_in, int d_out) {
    int N     = inf.slots;
    auto p    = compute_cm_params(N, d_in, d_out);
    int M_out         = N / p.tp_out;
    int cascade_shift = (p.t * p.tp) / p.tp_out;

    std::vector<std::vector<double>> pt(p.n_pt, std::vector<double>(N, 0.0));
    for (int j = 0; j < p.r_i; ++j)
        for (int k = 0; k < p.r_o; ++k)
            for (int i = 0; i < N; ++i) {
                int row = ((i / p.t + j * p.t + i % p.tp_in) % p.d)
                        + ((i % p.t) / p.tp_in) * p.d;
                int ms  = ((i / p.tp_out - k * cascade_shift) % M_out + M_out) % M_out;
                pt[j * p.r_o + k][i] = vals[row * d_out + interleave_idx(ms, p.d, d_out)];
            }

    std::vector<Ptx> result(p.n_pt);
    for (int i = 0; i < p.n_pt; ++i)
        result[i] = inf.cc()->MakeCKKSPackedPlaintext(pt[i]);
    return result;
}

std::vector<Ptx> load_weight_txt(Inference& inf, const std::string& path,
                                  int d_in, int d_out) {
    return encode_weight_matrix(inf, load_matrix_txt(path, d_in, d_out), d_in, d_out);
}
