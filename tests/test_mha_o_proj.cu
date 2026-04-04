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

// Ground truth full attention output (d-dimensional, flat interleaved order)
static std::vector<double> ref_full_attention_flat(
        const std::vector<double>& Q,
        const std::vector<std::vector<double>>& K,
        const std::vector<std::vector<double>>& V,
        int d, int H, double given_max) {
    int nk = (int)K.size(), d_head = d / H;
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

    // Weighted V sum: out[r] in interleaved order
    std::vector<double> out(d, 0.0);
    for (int r = 0; r < d; ++r)
        for (int k = 0; k < nk; ++k)
            out[r] += soft[k][r % H] * V[k][r];
    return out;
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
                dot += Q[h + ld * H] * K[k][h + ld * H];
            mx = std::max(mx, std::abs(dot * scale));
        }
    return mx * 1.5;
}

int main() {
    const int logN = 16, d = 1024, d_exp = 4096, H = 16, nk = 5;
    int S = 1 << (logN - 1);
    int t = S / d, d_head = d / H;

    auto extra_rots = compute_gpt2_rot_indices(S, d, d_exp, H, 1);
    std::cout << "=== MHA + Out-Proj Test ===\n"
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

    auto rand_mat = [&](int r, int c) {
        std::vector<std::vector<double>> W(r, std::vector<double>(c));
        for (auto& row : W) for (auto& v : row) v = dist(rng);
        return W;
    };
    auto make_vec = [&]() {
        std::vector<double> v(d);
        for (auto& x : v) x = dist(rng);
        double n = 0; for (auto x : v) n += x*x; n = std::sqrt(n);
        for (auto& x : v) x /= n;
        return v;
    };

    auto W_Q = rand_mat(d,d), W_K = rand_mat(d,d), W_V = rand_mat(d,d);
    auto W_O = rand_mat(d,d);
    auto W_Qr = rearrange_qkv_weights(W_Q, H);
    auto W_Kr = rearrange_qkv_weights(W_K, H);
    auto W_Vr = rearrange_qkv_weights(W_V, H);
    auto W_Or = rearrange_wo_weights(W_O, H);
    inf.w["q"]   = encode_weight_matrix(inf, W_Qr, d, d);
    inf.w["k"]   = encode_weight_matrix(inf, W_Kr, d, d);
    inf.w["v"]   = encode_weight_matrix(inf, W_Vr, d, d);
    inf.w["out"] = encode_weight_matrix(inf, W_Or, d, d);

    std::vector<std::vector<double>> K_ref(nk), V_ref(nk);
    for (int i = 0; i < nk; ++i) {
        auto x = make_vec();
        cache_k_push(inf, linear(inf, encode_linear_input(inf, x, d, d), "k", d, d));
        cache_v_push(inf, linear(inf, encode_linear_input(inf, x, d, d), "v", d, d));
        K_ref[i] = ref_matmul(x, W_Kr);
        V_ref[i] = ref_matmul(x, W_Vr);
    }
    auto cx = make_vec();
    auto Q_ref = ref_matmul(cx, W_Qr);
    Ctx cx_enc = encode_linear_input(inf, cx, d, d);
    Ctx q_enc = linear(inf, cx_enc, "q", d, d);

    double given_max = oracle_max(Q_ref, K_ref, d, H);
    Ctx qkt_enc = qkt(inf, q_enc);

    Ctx soft_enc = attention_softmax(inf, qkt_enc, nk, given_max, SOFTMAX_ATTN_GPT2);

    Ctx attn_out = softmax_v(inf, soft_enc);

    auto attn_dec = decrypt(inf.cc(), attn_out, inf.fhe->sk());
    auto attn_ref_flat = ref_full_attention_flat(Q_ref, K_ref, V_ref, d, H, given_max);

    Ctx out_enc = out_proj(inf, attn_out);
    std::cout << "  out_proj level=" << level_of(out_enc) << "\n";
    auto out_ref = ref_matmul(attn_ref_flat, W_Or);

    auto out_dec = decrypt(inf.cc(), out_enc, inf.fhe->sk());
    double out_max_err = 0.0;
    std::cout << "\n[8] Out-proj output check (first 8 dims):\n";
    for (int r = 0; r < d; ++r) {
        int slot = r * t;  // d→d linear output: slot i*t = dim i
        double err = std::abs(out_dec[slot] - out_ref[r]);
        out_max_err = std::max(out_max_err, err);
        if (r < 8)
            std::cout << "  dim " << r << " (slot " << slot
                      << "): dec=" << std::scientific << out_dec[slot]
                      << "  ref=" << out_ref[r]
                      << "  err=" << err << "\n";
    }

    Ctx res_enc = inf.cc()->EvalAdd(out_enc, cx_enc);
    std::cout << "  residual level=" << level_of(res_enc) << "\n";

    std::vector<double> res_ref(d);
    for (int r = 0; r < d; ++r) res_ref[r] = out_ref[r] + cx[r];

    auto res_dec = decrypt(inf.cc(), res_enc, inf.fhe->sk());
    double res_max_err = 0.0;
    std::cout << "\n[10] Residual output check (first 8 dims):\n";
    for (int r = 0; r < d; ++r) {
        int slot = r * t;
        double err = std::abs(res_dec[slot] - res_ref[r]);
        res_max_err = std::max(res_max_err, err);
        if (r < 8)
            std::cout << "  dim " << r << " (slot " << slot
                      << "): dec=" << std::scientific << res_dec[slot]
                      << "  ref=" << res_ref[r]
                      << "  err=" << err << "\n";
    }

    std::cout << "\n════════════════════════════════════════\n"
              << "  oproj max_err=" << std::scientific << out_max_err
              << "  " << (out_max_err < 1e-2 ? "PASS" : "FAIL") << "\n"
              << "  resid max_err=" << std::scientific << res_max_err
              << "  " << (res_max_err < 1e-2 ? "PASS" : "FAIL") << "\n"
              << "════════════════════════════════════════\n";

    return (out_max_err < 1e-2 && res_max_err < 1e-2) ? 0 : 1;
}
