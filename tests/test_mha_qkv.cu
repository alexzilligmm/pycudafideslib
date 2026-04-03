#include "gpt2.h"
#include "attention.h"
#include <iostream>
#include <random>
#include <cmath>

// Encode ground-truth output in CacheMIR square-matrix N-slot format:
// For square d->d with alpha=1, is_up=true: y[i] sits at slot i*t
static std::vector<double> encode_gt_square(const std::vector<double>& y, int N, int d) {
    int t = N / d;
    std::vector<double> ct(N, 0.0);
    for (int i = 0; i < d; ++i)
        ct[i * t] = y[i];
    return ct;
}

// Reference matmul x @ W
static std::vector<double> ref_matmul(const std::vector<double>& x,
                                       const std::vector<std::vector<double>>& W,
                                       int d_in, int d_out) {
    std::vector<double> y(d_out, 0.0);
    for (int j = 0; j < d_out; ++j)
        for (int i = 0; i < d_in; ++i)
            y[j] += x[i] * W[i][j];
    return y;
}

int main() {
    const int logN    = 16;
    const int d       = 1024;
    const int d_exp   = 4096;
    const int H       = 16;
    const int d_head  = d / H;

    int S = 1 << (logN - 1);
    auto extra_rots = compute_gpt2_rot_indices(S, d, d_exp, H, 1);

    std::cout << "Creating CKKS context (logN=" << logN << ")...\n";
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
    inf.size.hidDim   = d;
    inf.size.numHeads = H;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-0.01, 0.01);

    // Random weight matrix W_Q (d x d)
    std::vector<std::vector<double>> W_Q(d, std::vector<double>(d));
    for (auto& row : W_Q) for (auto& v : row) v = dist(rng);

    // Random input x (d,), normalized
    std::vector<double> x(d);
    for (auto& v : x) v = dist(rng);
    double norm = 0.0;
    for (auto v : x) norm += v * v;
    norm = std::sqrt(norm);
    for (auto& v : x) v /= norm;

    // ── Test 1: rearrange_qkv_weights correctness (pure CPU) ──
    std::cout << "\n[rearrange] Testing column permutation ... " << std::flush;
    auto W_rearranged = rearrange_qkv_weights(W_Q, H);

    // Verify: rearranged[:, r] == W_Q[:, h*d_head + ld] where h=r%H, ld=r//H
    double max_err = 0.0;
    for (int r = 0; r < d; ++r) {
        int h  = r % H;
        int ld = r / H;
        for (int i = 0; i < d; ++i)
            max_err = std::max(max_err,
                std::abs(W_rearranged[i][r] - W_Q[i][h * d_head + ld]));
    }
    bool ok_rearrange = max_err < 1e-15;
    std::cout << (ok_rearrange ? "PASS" : "FAIL")
              << "  max_err=" << max_err << "\n";

    // ── Test 2: linear(rearranged W_Q) vs encoded ground truth ──
    std::cout << "[qkv_proj] Testing rearranged projection in N-slot space ... " << std::flush;

    // Ground truth: x @ W_rearranged (result is already in interleaved order)
    auto y_gt = ref_matmul(x, W_rearranged, d, d);
    // Encode ground truth into N-slot format
    auto gt_encoded = encode_gt_square(y_gt, S, d);

    // FHE path: encode rearranged weights, project
    inf.w["q"] = encode_weight_matrix(inf, W_rearranged, d, d);
    Ctx x_enc = encode_linear_input(inf, x, d, d);
    Ctx q_enc = linear(inf, x_enc, "q", d, d);
    auto q_raw = decrypt(inf.cc(), q_enc, inf.fhe->sk());

    // Compare raw decrypted slots against encoded ground truth
    max_err = 0.0;
    for (int i = 0; i < S; ++i)
        max_err = std::max(max_err, std::abs(q_raw[i] - gt_encoded[i]));

    bool ok_proj = max_err < 1e-3;
    std::cout << (ok_proj ? "PASS" : "FAIL")
              << "  max_err=" << max_err << "\n";

    bool ok = ok_rearrange && ok_proj;
    std::cout << "\n" << (ok ? "ALL PASSED" : "SOME FAILURES") << "\n";
    return ok ? 0 : 1;
}
