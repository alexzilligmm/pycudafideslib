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

// Compute ground-truth softmax in N-slot format
// Python: ninf + exp(scores - max) / sum(exp)
static std::vector<double> ref_attention_softmax(
        const std::vector<std::vector<double>>& qkt,  // (nk, H)
        int N, int d, int H, double given_max) {
    int t  = N / d;
    int nk = (int)qkt.size();

    // Per-head softmax
    // softmax[k][h] = exp(qkt[k][h] - max) / sum_k'(exp(qkt[k'][h] - max))
    std::vector<std::vector<double>> soft(nk, std::vector<double>(H, 0.0));
    for (int h = 0; h < H; ++h) {
        double sum = 0.0;
        for (int k = 0; k < nk; ++k) {
            soft[k][h] = std::exp(qkt[k][h] - given_max);
            sum += soft[k][h];
        }
        for (int k = 0; k < nk; ++k)
            soft[k][h] /= sum;
    }

    // Encode into N-slot format
    std::vector<double> encoded(N, 0.0);
    for (int k = 0; k < nk; ++k)
        for (int h = 0; h < H; ++h)
            encoded[k / t * t * H + h * t + k % t] = soft[k][h];
    return encoded;
}

int main() {
    const int logN   = 16;
    const int d      = 1024;
    const int d_exp  = 4096;
    const int H      = 16;
    const int nk     = 5;

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

    // Build K cache
    std::vector<std::vector<double>> K_interleaved(nk);
    for (int i = 0; i < nk; ++i) {
        Ctx x_enc = encode_linear_input(inf, old_X[i], d, d);
        Ctx k_enc = linear(inf, x_enc, "k", d, d);
        cache_k_push(inf, k_enc);
        K_interleaved[i] = ref_matmul(old_X[i], W_K_r);
    }

    // Compute QKT
    auto Q_interleaved = ref_matmul(curr_x, W_Q_r);
    Ctx q_enc = encode_linear_input(inf, curr_x, d, d);
    q_enc = linear(inf, q_enc, "q", d, d);
    Ctx qkt_enc = qkt(inf, q_enc);

    // Ground-truth QKT (nk, H)
    int d_head = d / H;
    std::vector<std::vector<double>> gt_qkt(nk, std::vector<double>(H, 0.0));
    double scale = 1.0 / std::sqrt((double)d_head);
    for (int k = 0; k < nk; ++k)
        for (int r = 0; r < d; ++r) {
            int h = r % H;
            gt_qkt[k][h] += Q_interleaved[r] * K_interleaved[k][r];
        }
    for (int k = 0; k < nk; ++k)
        for (int h = 0; h < H; ++h)
            gt_qkt[k][h] *= scale;

    // Use max of gt_qkt as oracle max
    double given_max = 0.0;
    for (int k = 0; k < nk; ++k)
        for (int h = 0; h < H; ++h)
            given_max = std::max(given_max, std::abs(gt_qkt[k][h]));
    given_max *= 1.5; // some headroom

    // ── Test: attention_softmax ──
    std::cout << "\n[softmax] Testing attention softmax ... " << std::flush;
    Ctx soft_enc = attention_softmax(inf, qkt_enc, nk, given_max);
    auto soft_raw = decrypt(inf.cc(), soft_enc, inf.fhe->sk());

    auto gt_soft = ref_attention_softmax(gt_qkt, S, d, H, given_max);

    // Compare only at valid token positions
    int t = S / d;
    double max_err = 0.0;
    for (int k = 0; k < nk; ++k)
        for (int h = 0; h < H; ++h) {
            int idx = k / t * t * H + h * t + k % t;
            max_err = std::max(max_err, std::abs(soft_raw[idx] - gt_soft[idx]));
        }

    bool ok = max_err < 1e-2; // softmax has more error from exp + goldschmidt
    std::cout << (ok ? "PASS" : "FAIL") << "  max_err=" << max_err << "\n";

    std::cout << "\n" << (ok ? "ALL PASSED" : "SOME FAILURES") << "\n";
    return ok ? 0 : 1;
}
