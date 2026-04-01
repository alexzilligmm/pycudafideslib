#include "gpt2.h"
#include <iostream>
#include <random>
#include <cmath>

static std::vector<double> ref_matmul(const std::vector<double>& x,
                                       const std::vector<double>& W,
                                       int d_in, int d_out) {
    std::vector<double> y(d_out, 0.0);
    for (int j = 0; j < d_out; ++j)
        for (int i = 0; i < d_in; ++i)
            y[j] += x[i] * W[i * d_out + j];
    return y;
}

static std::vector<double> decode_output(const std::vector<double>& cy,
                                          int N, int d_in, int d_out) {
    auto p = compute_cm_params(N, d_in, d_out);
    int M = N / p.tp;
    std::vector<double> y(d_out, 0.0);

    if (p.is_up && p.alpha > 1) {
        for (int m = 0; m < M; ++m) {
            int idx = interleave_idx(m, p.d, d_out);
            if (idx < d_out)
                y[idx] = cy[m * p.tp];
        }
    } else {
        for (int i = 0; i < d_out; ++i)
            y[i] = cy[i * p.t];
    }
    return y;
}

static bool run_test(const std::string& label, Inference& inf,
                     const std::vector<double>& W, int d_in, int d_out,
                     double tol = 1e-3) {
    std::cout << "[" << label << "] " << d_in << "x" << d_out << " ... " << std::flush;

    inf.w[label] = encode_weight_matrix(inf, W, d_in, d_out);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-0.1, 0.1);
    std::vector<double> x(d_in);
    for (auto& v : x) v = dist(rng);

    auto raw = decrypt(inf.cc(),
                       linear(inf, encode_linear_input(inf, x, d_in, d_out),
                              label, d_in, d_out),
                       inf.fhe->sk());
    auto y_fhe = decode_output(raw, inf.slots, d_in, d_out);
    auto y_ref = ref_matmul(x, W, d_in, d_out);

    double max_err = 0.0;
    for (int i = 0; i < d_out; ++i)
        max_err = std::max(max_err, std::abs(y_ref[i] - y_fhe[i]));

    bool ok = max_err < tol;
    std::cout << (ok ? "PASS" : "FAIL") << "  max_err=" << max_err << "\n";
    return ok;
}

int main() {
    const int logN  = 16;
    const int d     = 1024;
    const int d_exp = 4096;

    int S = 1 << (logN - 1);
    auto extra_rots = compute_gpt2_rot_indices(S, d, d_exp, /*numHeads=*/1, /*seqLen=*/1);

    Inference inf;
    inf.fhe        = make_ckks_context(logN, /*depth=*/13, /*scale_bits=*/41,
                                        0, /*bootstrap=*/true,
                                        /*btp_scale_bits=*/50, /*first_mod_bits=*/53,
                                        /*level_budget_in=*/{4, 3},
                                        /*batch_size=*/0,
                                        /*h_weight=*/192,
                                        /*num_large_digits=*/3,
                                        /*btp_depth_overhead=*/15,
                                        /*extra_rot_steps=*/extra_rots);
    inf.slots      = S;
    inf.bench_mode = false;

    std::mt19937 rng(0);
    std::uniform_real_distribution<double> dist(-0.01, 0.01);
    auto rand_mat = [&](int r, int c) {
        std::vector<double> W(r * c);
        for (auto& v : W) v = dist(rng);
        return W;
    };

    bool ok = true;
    ok &= run_test("square", inf, rand_mat(d,     d),     d,     d);
    ok &= run_test("up",     inf, rand_mat(d,     d_exp), d,     d_exp);
    ok &= run_test("down",   inf, rand_mat(d_exp, d),     d_exp, d);
    return ok ? 0 : 1;
}
