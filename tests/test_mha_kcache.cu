#include "gpt2.h"
#include "attention.h"
#include <iostream>
#include <random>
#include <cmath>

// Reference matmul x @ W
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

// Encode ground-truth QK^T into head-interleaved N-slot format
// Matches Python gt_qkt_repack: for each query q, key k, head h:
//   slot[k//t * tH + h*t + k%t] = qkt[q][k][h]
static std::vector<double> encode_gt_qkt(const std::vector<std::vector<std::vector<double>>>& gt_qkt,
                                          int N, int d, int H) {
    int t  = N / d;
    int tH = t * H;
    int nq = (int)gt_qkt.size();
    int nk = (int)gt_qkt[0].size();
    // For single-query decoding test, use first query
    std::vector<double> encoded(N, 0.0);
    for (int k = 0; k < nk; ++k)
        for (int h = 0; h < H; ++h)
            encoded[k / t * tH + h * t + k % t] = gt_qkt[0][k][h];
    return encoded;
}

// Compute ground-truth QK^T: Q(nq,H,d_head) x K(nk,H,d_head) -> (nq,nk,H)
// Scaled by 1/sqrt(d_head)
static std::vector<std::vector<std::vector<double>>> compute_gt_qkt(
        const std::vector<std::vector<double>>& Q_heads,  // (nq, d) in interleaved order
        const std::vector<std::vector<double>>& K_heads,  // (nk, d) in interleaved order
        int d, int H) {
    int nq = (int)Q_heads.size();
    int nk = (int)K_heads.size();
    int d_head = d / H;
    double scale = 1.0 / std::sqrt((double)d_head);

    // gt_qkt[q][k][h] = sum_ld Q[q,h,ld] * K[k,h,ld] * scale
    // In interleaved order: index r -> h=r%H, ld=r//H
    std::vector<std::vector<std::vector<double>>> gt(nq,
        std::vector<std::vector<double>>(nk, std::vector<double>(H, 0.0)));

    for (int q = 0; q < nq; ++q)
        for (int k = 0; k < nk; ++k)
            for (int r = 0; r < d; ++r) {
                int h = r % H;
                gt[q][k][h] += Q_heads[q][r] * K_heads[k][r];
            }

    for (int q = 0; q < nq; ++q)
        for (int k = 0; k < nk; ++k)
            for (int h = 0; h < H; ++h)
                gt[q][k][h] *= scale;

    return gt;
}

int main() {
    const int logN   = 16;
    const int d      = 1024;
    const int d_exp  = 4096;
    const int H      = 16;
    const int nk     = 5;  // number of past keys

    int S = 1 << (logN - 1);
    auto extra_rots = compute_gpt2_rot_indices(S, d, d_exp, H, 1);

    std::cout << "Creating CKKS context (logN=" << logN << ")...\n";
    Inference inf;
    inf.fhe = make_ckks_context(logN, /*depth=*/13, /*scale_bits=*/41,
                                 0, /*bootstrap=*/true,
                                 /*btp_scale_bits=*/50, /*first_mod_bits=*/53,
                                 /*level_budget_in=*/{4, 3},
                                 /*batch_size=*/0,
                                 /*h_weight=*/192,
                                 /*num_large_digits=*/3,
                                 /*btp_depth_overhead=*/15,
                                 /*extra_rot_steps=*/extra_rots);
    inf.slots         = S;
    inf.bench_mode    = false;
    inf.size.hidDim   = d;
    inf.size.numHeads = H;

    prepare_mha_masks(inf);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-0.01, 0.01);

    // Random weights
    std::vector<std::vector<double>> W_Q(d, std::vector<double>(d));
    std::vector<std::vector<double>> W_K(d, std::vector<double>(d));
    for (auto& row : W_Q) for (auto& v : row) v = dist(rng);
    for (auto& row : W_K) for (auto& v : row) v = dist(rng);

    auto W_Q_r = rearrange_qkv_weights(W_Q, H);
    auto W_K_r = rearrange_qkv_weights(W_K, H);

    inf.w["q"] = encode_weight_matrix(inf, W_Q_r, d, d);
    inf.w["k"] = encode_weight_matrix(inf, W_K_r, d, d);

    // Random past tokens (nk) + current query token (1), normalized
    auto make_normalized_vec = [&]() {
        std::vector<double> v(d);
        for (auto& x : v) x = dist(rng);
        double norm = 0.0;
        for (auto x : v) norm += x * x;
        norm = std::sqrt(norm);
        for (auto& x : v) x /= norm;
        return v;
    };

    std::vector<std::vector<double>> old_X(nk);
    for (auto& x : old_X) x = make_normalized_vec();
    auto curr_x = make_normalized_vec();

    // ── Build K cache incrementally ──
    std::cout << "\n[kcache] Pushing " << nk << " keys ... " << std::flush;

    // Collect ground-truth K vectors in interleaved order
    std::vector<std::vector<double>> K_interleaved(nk);
    for (int i = 0; i < nk; ++i) {
        Ctx x_enc = encode_linear_input(inf, old_X[i], d, d);
        Ctx k_enc = linear(inf, x_enc, "k", d, d);
        cache_k_push(inf, k_enc);

        // Also compute plaintext reference: old_X[i] @ W_K_rearranged
        K_interleaved[i] = ref_matmul(old_X[i], W_K_r);
    }
    std::cout << "done (seqLen=" << inf.k_count << ")\n";

    // ── Compute QKT ──
    std::cout << "[qkt] Computing QK^T ... " << std::flush;
    Ctx q_enc = encode_linear_input(inf, curr_x, d, d);
    q_enc = linear(inf, q_enc, "q", d, d);
    Ctx qkt_enc = qkt(inf, q_enc);
    auto qkt_raw = decrypt(inf.cc(), qkt_enc, inf.fhe->sk());

    // Ground truth: Q in interleaved order
    auto Q_interleaved = ref_matmul(curr_x, W_Q_r);
    auto gt_qkt = compute_gt_qkt({Q_interleaved}, K_interleaved, d, H);
    auto gt_encoded = encode_gt_qkt(gt_qkt, S, d, H);

    // Compare
    double max_err = 0.0;
    for (int i = 0; i < S; ++i)
        max_err = std::max(max_err, std::abs(qkt_raw[i] - gt_encoded[i]));

    bool ok = max_err < 1e-3;
    std::cout << (ok ? "PASS" : "FAIL") << "  max_err=" << max_err << "\n";

    std::cout << "\n" << (ok ? "ALL PASSED" : "SOME FAILURES") << "\n";
    return ok ? 0 : 1;
}
