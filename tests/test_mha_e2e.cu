#include "gpt2.h"
#include "attention.h"
#include "nonlinear.h"
#include "ckks_primitives.h"
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

// Ground truth full attention output in N-slot format
static std::vector<double> ref_full_attention(
        const std::vector<double>& Q,
        const std::vector<std::vector<double>>& K,
        const std::vector<std::vector<double>>& V,
        int N, int d, int H, double given_max) {
    int nk = (int)K.size(), d_head = d / H, t = N / d;
    double scale = 1.0 / std::sqrt((double)d_head);

    // QK^T per head
    std::vector<std::vector<double>> qkt(nk, std::vector<double>(H, 0.0));
    for (int k = 0; k < nk; ++k)
        for (int r = 0; r < d; ++r)
            qkt[k][r % H] += Q[r] * K[k][r];
    for (auto& row : qkt) for (auto& v : row) v *= scale;

    // Softmax per head
    std::vector<std::vector<double>> soft(nk, std::vector<double>(H, 0.0));
    for (int h = 0; h < H; ++h) {
        double sum = 0.0;
        for (int k = 0; k < nk; ++k) { soft[k][h] = std::exp(qkt[k][h] - given_max); sum += soft[k][h]; }
        for (int k = 0; k < nk; ++k) soft[k][h] /= sum;
    }

    // Weighted V sum
    std::vector<double> out(d, 0.0);
    for (int r = 0; r < d; ++r)
        for (int k = 0; k < nk; ++k)
            out[r] += soft[k][r % H] * V[k][r];

    // Encode into N-slot format (tok=0 only)
    std::vector<double> encoded(N, 0.0);
    for (int r = 0; r < d; ++r)
        encoded[pos(r / H, r % H, 0, t, H)] = out[r];
    return encoded;
}

// Oracle max: max |QK^T score| across all heads, scaled by 1.5
static double oracle_max(const std::vector<double>& Q,
                         const std::vector<std::vector<double>>& K,
                         int d, int H) {
    int nk = (int)K.size(), d_head = d / H;
    double scale = 1.0 / std::sqrt((double)d_head);
    double mx = 0.0;
    for (int k = 0; k < nk; ++k)
        for (int h = 0; h < H; ++h) {
            double dot = 0.0;
            for (int ld = 0; ld < d_head; ++ld)
                dot += Q[h + ld * H] * K[k][h + ld * H]; // rearranged layout
            mx = std::max(mx, std::abs(dot * scale));
        }
    return mx * 1.5;
}

int main() {
    const int logN = 16, d = 1024, d_exp = 4096, H = 16, nk = 5;
    int S = 1 << (logN - 1);
    int t = S / d, d_head = d / H;

    auto extra_rots = compute_gpt2_rot_indices(S, d, d_exp, H, 1);
    std::cout << "=== MHA E2E Minimal Test ===\n"
              << "  N=" << S << " d=" << d << " H=" << H << " t=" << t
              << " d_head=" << d_head << " nk=" << nk << "\n"
              << "Creating CKKS context ...\n";

    Inference inf;
    inf.fhe = make_ckks_context(logN, 13, 41, 0, true, 50, 53, {4,3}, 0, 192, 3, 15, extra_rots);
    inf.slots = S;  inf.total_depth = 28;  inf.bench_mode = false;
    inf.size.hidDim = d;  inf.size.numHeads = H;
    prepare_mha_masks(inf);
    prepare_vcache(inf);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-0.01, 0.01);

    // ── Weights ──
    std::cout << "\n[1] Weights ...\n";
    auto rand_mat = [&](int r, int c) {
        std::vector<std::vector<double>> W(r, std::vector<double>(c));
        for (auto& row : W) for (auto& v : row) v = dist(rng);
        return W;
    };
    auto W_Q = rand_mat(d,d), W_K = rand_mat(d,d), W_V = rand_mat(d,d);
    auto W_Qr = rearrange_qkv_weights(W_Q, H);
    auto W_Kr = rearrange_qkv_weights(W_K, H);
    auto W_Vr = rearrange_qkv_weights(W_V, H);
    inf.w["q"] = encode_weight_matrix(inf, W_Qr, d, d);
    inf.w["k"] = encode_weight_matrix(inf, W_Kr, d, d);
    inf.w["v"] = encode_weight_matrix(inf, W_Vr, d, d);

    auto make_vec = [&]() {
        std::vector<double> v(d);
        for (auto& x : v) x = dist(rng);
        double n = 0; for (auto x : v) n += x*x; n = std::sqrt(n);
        for (auto& x : v) x /= n;
        return v;
    };

    // ── KV caches ──
    std::cout << "\n[2] Building KV caches ...\n";
    std::vector<std::vector<double>> K_ref(nk), V_ref(nk);
    for (int i = 0; i < nk; ++i) {
        auto x = make_vec();
        cache_k_push(inf, linear(inf, encode_linear_input(inf, x, d, d), "k", d, d));
        cache_v_push(inf, linear(inf, encode_linear_input(inf, x, d, d), "v", d, d));
        K_ref[i] = ref_matmul(x, W_Kr);
        V_ref[i] = ref_matmul(x, W_Vr);
    }
    std::cout << "  k_count=" << inf.k_count << " v_count=" << inf.v_count << "\n";

    // ── Q projection ──
    std::cout << "\n[3] Q projection ...\n";
    auto cx = make_vec();
    auto Q_ref = ref_matmul(cx, W_Qr);
    Ctx q_enc = linear(inf, encode_linear_input(inf, cx, d, d), "q", d, d);
    std::cout << "  level=" << level_of(q_enc) << "\n";

    // ── Oracle max from plaintext scores ──
    double given_max = oracle_max(Q_ref, K_ref, d, H);
    const SoftmaxConfig& cfg = SOFTMAX_ATTN_GPT2;
    double mask_max = std::max(given_max, std::pow(2.0, cfg.exp_r - 1));
    std::cout << "\n[4] Oracle max=" << std::scientific << given_max
              << "  mask_max=" << mask_max << "\n";

    // ── Black-box: QK^T → softmax → V ──
    std::cout << "\n[5] Black-box: qkt → attention_softmax → softmax_v ...\n";
    Ctx qkt_enc = qkt(inf, q_enc);
    std::cout << "  qkt level=" << level_of(qkt_enc) << "\n";

    Ctx soft_enc = attention_softmax(inf, qkt_enc, nk, given_max);
    std::cout << "  softmax level=" << level_of(soft_enc) << "\n";

    Ctx attn_out = softmax_v(inf, soft_enc);
    std::cout << "  attn_out level=" << level_of(attn_out) << "\n";

    // ── Compare against ground truth ──
    auto gt = ref_full_attention(Q_ref, K_ref, V_ref, S, d, H, given_max);
    auto dec = decrypt(inf.cc(), attn_out, inf.fhe->sk());

    double max_err = 0.0;
    for (int ld = 0; ld < d_head; ++ld)
        for (int h = 0; h < H; ++h) {
            int idx = pos(ld, h, 0, t, H);
            max_err = std::max(max_err, std::abs(dec[idx] - gt[idx]));
        }

    std::cout << "\n════════════════════════════════════════\n"
              << "  max_err=" << std::scientific << max_err
              << "  " << (max_err < 1e-2 ? "PASS" : "FAIL") << "\n"
              << "════════════════════════════════════════\n";

    return (max_err < 1e-2) ? 0 : 1;
}
