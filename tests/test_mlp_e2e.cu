#include "gpt2.h"
#include "nonlinear.h"
#include "ckks_primitives.h"
#include <iostream>
#include <random>
#include <cmath>
#include <stdexcept>

#define LOG(msg) do { std::cout << msg << std::endl; } while (0)

static std::vector<double> ref_matmul(const std::vector<double>& x,
                                       const std::vector<std::vector<double>>& W) {
    int d_in  = (int)x.size();
    int d_out = (int)W[0].size();
    std::vector<double> y(d_out, 0.0);
    for (int j = 0; j < d_out; ++j)
        for (int i = 0; i < d_in; ++i)
            y[j] += x[i] * W[i][j];
    return y;
}

static std::vector<double> ref_layernorm(const std::vector<double>& x, double eps = 1e-5) {
    int d = (int)x.size();
    double mean = 0.0;
    for (auto v : x) mean += v;
    mean /= d;
    double var = 0.0;
    for (auto v : x) var += (v - mean) * (v - mean);
    var /= d;
    double inv_std = 1.0 / std::sqrt(var + eps);
    std::vector<double> y(d);
    for (int i = 0; i < d; ++i)
        y[i] = (x[i] - mean) * inv_std;
    return y;
}

static std::vector<double> ref_gelu(const std::vector<double>& x) {
    std::vector<double> y(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        y[i] = 0.5 * x[i] * (1.0 + std::erf(x[i] / std::sqrt(2.0)));
    return y;
}

static double safe_check_stride(Inference& inf, const Ctx& ct,
                                 const std::vector<double>& ref,
                                 int stride, int n, const char* label) {
    try {
        auto dec = decrypt(inf.cc(), ct, inf.fhe->sk());
        double max_err = 0.0;
        for (int r = 0; r < n; ++r)
            max_err = std::max(max_err, std::abs(dec[r * stride] - ref[r]));
        LOG("  " << label << " max_err=" << std::scientific << max_err);
        return max_err;
    } catch (const std::exception& e) {
        LOG("  " << label << " DECRYPT FAILED: " << e.what());
        return -1.0;
    }
}

static double safe_check_interleaved(Inference& inf, const Ctx& ct,
                                      const std::vector<double>& ref,
                                      int N, int d, int d_exp, const char* label) {
    try {
        auto dec = decrypt(inf.cc(), ct, inf.fhe->sk());
        auto p = compute_cm_params(N, d, d_exp);
        double max_err = 0.0;
        for (int m = 0; m < d_exp; ++m) {
            int slot    = m * p.tp;
            int ref_idx = interleave_idx(m, d, d_exp);
            max_err = std::max(max_err, std::abs(dec[slot] - ref[ref_idx]));
        }
        LOG("  " << label << " max_err=" << std::scientific << max_err);
        return max_err;
    } catch (const std::exception& e) {
        LOG("  " << label << " DECRYPT FAILED: " << e.what());
        return -1.0;
    }
}

int main() {
    const int logN = 16, d = 1024, d_exp = 4096, H = 16;
    const int S = 1 << (logN - 1), t = S / d;

    LOG("=== MLP E2E Test ===");
    LOG("  S=" << S << " d=" << d << " d_exp=" << d_exp << " H=" << H << " t=" << t);

    auto extra_rots = compute_gpt2_rot_indices(S, d, d_exp, H, 1);

    Inference inf;
    inf.fhe = make_ckks_context(logN, 13, 41, 0, true, 50, 53, {4,3}, 0, 192, 3, 15, extra_rots);
    inf.slots = S;  inf.total_depth = 28;  inf.bench_mode = false;
    inf.size.hidDim = d;  inf.size.dim = d;  inf.size.expDim = d_exp;  inf.size.numHeads = H;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> xdist(-1.0, 1.0);
    auto rand_mat = [&](int r, int c) {
        double sc = 1.0 / std::sqrt((double)r);
        std::vector<std::vector<double>> W(r, std::vector<double>(c));
        for (auto& row : W) for (auto& v : row) v = xdist(rng) * sc;
        return W;
    };

    auto W_up = rand_mat(d, d_exp), W_down = rand_mat(d_exp, d);
    inf.w["up"]   = encode_weight_matrix(inf, W_up,   d, d_exp);
    inf.w["down"] = encode_weight_matrix(inf, W_down, d_exp, d);

    std::vector<double> x(d);
    for (auto& v : x) v = xdist(rng);
    Ctx x_enc = encode_linear_input(inf, x, d, d);

    auto x_n = ref_layernorm(x);
    auto up  = ref_matmul(x_n, W_up);
    auto gl  = ref_gelu(up);
    auto dn  = ref_matmul(gl, W_down);
    std::vector<double> mlp_ref(d);
    for (int i = 0; i < d; ++i) mlp_ref[i] = dn[i] + x[i];

    Ctx h = norm(inf, x_enc, 14, NORM_ENCLLM_GPT2);
    LOG("  post-norm level=" << level_of(h));
    safe_check_stride(inf, h, x_n, t, d, "norm");

    h = linear(inf, h, "up", d, d_exp);
    LOG("  post-up level=" << level_of(h));
    safe_check_interleaved(inf, h, up, S, d, d_exp, "up_proj");

    h = bootstrap_to(inf, h, 13);
    LOG("  post-btp level=" << level_of(h));

    h = gelu(inf, h, GELU_ENCLLM_GPT2);
    LOG("  post-gelu level=" << level_of(h));
    safe_check_interleaved(inf, h, gl, S, d, d_exp, "gelu");

    h = bootstrap_to(inf, h, 9);

    h = linear(inf, h, "down", d_exp, d);
    LOG("  post-down level=" << level_of(h));
    safe_check_stride(inf, h, dn, t, d, "down_proj");

    Ctx res = inf.cc()->EvalAdd(h, x_enc);
    double err = safe_check_stride(inf, res, mlp_ref, t, d, "residual");

    LOG("");
    LOG("  mlp max_err=" << std::scientific << err
        << "  " << (err < 5.0 ? "PASS" : "FAIL"));

    return (err < 5.0) ? 0 : 1;
}