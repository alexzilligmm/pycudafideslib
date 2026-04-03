#include "gpt2.h"
#include "attention.h"
#include <iostream>
#include <random>
#include <cmath>
#include <iomanip>

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

static int pos(int ld, int h, int tok, int t, int H) {
    return ld * t * H + h * t + tok;
}

static void print_slots(const std::vector<double>& ct, int N, int d, int H,
                        const char* label, int max_slots = 32) {
    int t = N / d;
    int tH = t * H;
    std::cout << "  " << label << ":\n    ";
    int count = 0;
    for (int i = 0; i < N && count < max_slots; ++i) {
        if (std::abs(ct[i]) > 1e-8) {
            int ld  = i / tH;
            int h   = (i % tH) / t;
            int tok = i % t;
            std::cout << "[" << i << "](ld=" << ld << ",h=" << h << ",tok=" << tok
                      << ")=" << std::fixed << std::setprecision(4) << ct[i] << " ";
            count++;
        }
    }
    std::cout << "\n";
}

// Full ground-truth multi-head attention in interleaved order
// Returns output encoded at tok=0 positions in N-slot format
static std::vector<double> ref_full_attention(
        const std::vector<double>& Q,
        const std::vector<std::vector<double>>& K,
        const std::vector<std::vector<double>>& V,
        int N, int d, int H, double given_max) {
    int nk = (int)K.size();
    int d_head = d / H;
    int t = N / d;
    double scale = 1.0 / std::sqrt((double)d_head);

    // QKT per head
    std::vector<std::vector<double>> qkt(nk, std::vector<double>(H, 0.0));
    for (int k = 0; k < nk; ++k)
        for (int r = 0; r < d; ++r)
            qkt[k][r % H] += Q[r] * K[k][r];
    for (int k = 0; k < nk; ++k)
        for (int h = 0; h < H; ++h)
            qkt[k][h] *= scale;

    // Softmax per head
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

    // Attention output
    std::vector<double> out(d, 0.0);
    for (int r = 0; r < d; ++r) {
        int h = r % H;
        for (int k = 0; k < nk; ++k)
            out[r] += soft[k][h] * V[k][r];
    }

    // Encode at tok=0
    std::vector<double> encoded(N, 0.0);
    for (int r = 0; r < d; ++r) {
        int h  = r % H;
        int ld = r / H;
        encoded[pos(ld, h, 0, t, H)] = out[r];
    }
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

    std::cout << "=== MHA End-to-End Test ===\n";
    std::cout << "  N=" << S << " d=" << d << " H=" << H << " nk=" << nk << "\n";
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
    prepare_vcache(inf);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-0.01, 0.01);

    // ── Step 1: Random weights, rearrange, encode ──
    std::cout << "\n[1] Preparing weights ...\n";
    auto rand_mat = [&](int r, int c) {
        std::vector<std::vector<double>> W(r, std::vector<double>(c));
        for (auto& row : W) for (auto& v : row) v = dist(rng);
        return W;
    };
    auto W_Q = rand_mat(d, d), W_K = rand_mat(d, d), W_V = rand_mat(d, d);
    auto W_Q_r = rearrange_qkv_weights(W_Q, H);
    auto W_K_r = rearrange_qkv_weights(W_K, H);
    auto W_V_r = rearrange_qkv_weights(W_V, H);

    inf.w["q"] = encode_weight_matrix(inf, W_Q_r, d, d);
    inf.w["k"] = encode_weight_matrix(inf, W_K_r, d, d);
    inf.w["v"] = encode_weight_matrix(inf, W_V_r, d, d);

    // ── Step 2: Generate normalized input vectors ──
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

    // ── Step 3: Build KV caches from past tokens ──
    std::cout << "[2] Building KV caches (" << nk << " tokens) ...\n";
    std::vector<std::vector<double>> K_ref(nk), V_ref(nk);
    for (int i = 0; i < nk; ++i) {
        Ctx x_enc = encode_linear_input(inf, old_X[i], d, d);
        Ctx k_enc = linear(inf, x_enc, "k", d, d);
        Ctx v_enc = linear(inf, x_enc, "v", d, d);
        cache_k_push(inf, k_enc);
        cache_v_push(inf, v_enc);

        K_ref[i] = ref_matmul(old_X[i], W_K_r);
        V_ref[i] = ref_matmul(old_X[i], W_V_r);
    }
    std::cout << "  k_count=" << inf.k_count << " v_count=" << inf.v_count << "\n";

    // ── Step 4: Query projection ──
    std::cout << "[3] Query projection ...\n";
    auto Q_ref = ref_matmul(curr_x, W_Q_r);
    Ctx q_enc = encode_linear_input(inf, curr_x, d, d);
    q_enc = linear(inf, q_enc, "q", d, d);

    // ── Step 5: QKT ──
    std::cout << "[4] QK^T ...\n";
    Ctx qkt_enc = qkt(inf, q_enc);

    // Oracle max
    int d_head = d / H;
    double given_max = 0.0;
    {
        double scale = 1.0 / std::sqrt((double)d_head);
        for (int k = 0; k < nk; ++k)
            for (int h = 0; h < H; ++h) {
                double dot = 0.0;
                for (int r = 0; r < d; ++r)
                    if (r % H == h) dot += Q_ref[r] * K_ref[k][r];
                given_max = std::max(given_max, std::abs(dot * scale));
            }
        given_max *= 1.5;
    }

    // ── Step 6: Softmax ──
    std::cout << "[5] Attention softmax (max=" << given_max << ") ...\n";
    Ctx soft_enc = attention_softmax(inf, qkt_enc, nk, given_max);

    // ── Step 7: V matmul ──
    std::cout << "[6] Softmax @ V ...\n";
    Ctx attn_out = softmax_v(inf, soft_enc);
    auto attn_raw = decrypt(inf.cc(), attn_out, inf.fhe->sk());

    // ── Compare ──
    auto gt = ref_full_attention(Q_ref, K_ref, V_ref, S, d, H, given_max);

    std::cout << "\n[result] Interleaving check:\n";
    print_slots(attn_raw, S, d, H, "FHE output");
    print_slots(gt, S, d, H, "GT  output");

    int t = S / d;
    double max_err = 0.0;
    for (int ld = 0; ld < d_head; ++ld)
        for (int h = 0; h < H; ++h) {
            int idx = pos(ld, h, 0, t, H);
            max_err = std::max(max_err, std::abs(attn_raw[idx] - gt[idx]));
        }

    bool ok = max_err < 1e-2;
    std::cout << "\n[e2e] " << (ok ? "PASS" : "FAIL")
              << "  max_err=" << max_err << "\n";
    std::cout << (ok ? "ALL PASSED" : "SOME FAILURES") << "\n";
    return ok ? 0 : 1;
}
