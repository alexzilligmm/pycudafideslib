#include "gpt2.h"
#include "attention.h"
#include <iostream>
#include <random>
#include <cmath>
#include <iomanip>

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

// _pos(ld, h, tok, t, H) = ld * t * H + h * t + tok
static int pos(int ld, int h, int tok, int t, int H) {
    return ld * t * H + h * t + tok;
}

// Print non-zero slots in a range to inspect interleaving
static void print_slots(const std::vector<double>& ct, int N, int d, int H,
                        const char* label, int max_slots = 64) {
    int t = N / d;
    int tH = t * H;
    std::cout << "  " << label << " (first " << max_slots << " slots, t=" << t
              << " tH=" << tH << "):\n    ";
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

// Ground-truth multi-head attention output in N-slot format
// Q(1,d_interleaved), K(nk,d_interleaved), V(nk,d_interleaved)
// Returns attention output encoded in N-slot tok=0 positions
static std::vector<double> ref_attention_output(
        const std::vector<double>& Q,
        const std::vector<std::vector<double>>& K,
        const std::vector<std::vector<double>>& V,
        int N, int d, int H, double given_max) {
    int nk = (int)K.size();
    int d_head = d / H;
    int t = N / d;
    double scale = 1.0 / std::sqrt((double)d_head);

    // QKT per head: qkt[k][h]
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

    // Attention output: for each head h, dim ld:
    //   out[h*d_head + ld] = sum_k soft[k][h] * V[k][h + ld*H]
    // (interleaved order: index r -> h=r%H, ld=r//H)
    std::vector<double> out(d, 0.0);
    for (int r = 0; r < d; ++r) {
        int h = r % H;
        for (int k = 0; k < nk; ++k)
            out[r] += soft[k][h] * V[k][r];
    }

    // Encode at tok=0 positions
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

    // Random weights
    std::vector<std::vector<double>> W_Q(d, std::vector<double>(d));
    std::vector<std::vector<double>> W_K(d, std::vector<double>(d));
    std::vector<std::vector<double>> W_V(d, std::vector<double>(d));
    for (auto& row : W_Q) for (auto& v : row) v = dist(rng);
    for (auto& row : W_K) for (auto& v : row) v = dist(rng);
    for (auto& row : W_V) for (auto& v : row) v = dist(rng);

    auto W_Q_r = rearrange_qkv_weights(W_Q, H);
    auto W_K_r = rearrange_qkv_weights(W_K, H);
    auto W_V_r = rearrange_qkv_weights(W_V, H);

    inf.w["q"] = encode_weight_matrix(inf, W_Q_r, d, d);
    inf.w["k"] = encode_weight_matrix(inf, W_K_r, d, d);
    inf.w["v"] = encode_weight_matrix(inf, W_V_r, d, d);

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

    // Build K and V caches, collect plaintext references
    std::cout << "\n[cache] Pushing " << nk << " keys+values ... " << std::flush;
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
    std::cout << "done (k_count=" << inf.k_count << ", v_count=" << inf.v_count << ")\n";

    // ── Print VCache metaciphertext layout ──
    int d_head = d / H;
    std::cout << "\n[vcache] Metaciphertext lanes: " << inf.cache["v"].size()
              << " (d_head=" << d_head << ")\n";
    for (int lane = 0; lane < std::min(3, d_head); ++lane) {
        auto raw = decrypt(inf.cc(), inf.cache["v"][lane], inf.fhe->sk());
        std::string label = "lane " + std::to_string(lane);
        print_slots(raw, S, d, H, label.c_str(), 32);
    }

    // Compute QKT + softmax
    auto Q_ref = ref_matmul(curr_x, W_Q_r);
    Ctx q_enc = encode_linear_input(inf, curr_x, d, d);
    q_enc = linear(inf, q_enc, "q", d, d);
    Ctx qkt_enc = qkt(inf, q_enc);

    // Oracle max
    double given_max = 0.0;
    {
        int d_h = d / H;
        double scale = 1.0 / std::sqrt((double)d_h);
        for (int k = 0; k < nk; ++k)
            for (int h = 0; h < H; ++h) {
                double dot = 0.0;
                for (int r = 0; r < d; ++r)
                    if (r % H == h) dot += Q_ref[r] * K_ref[k][r];
                given_max = std::max(given_max, std::abs(dot * scale));
            }
        given_max *= 1.5;
    }

    Ctx soft_enc = attention_softmax(inf, qkt_enc, nk, given_max);

    // ── Test: softmax_v ──
    std::cout << "\n[softmax_v] Computing attention output ... " << std::flush;
    Ctx attn_out = softmax_v(inf, soft_enc);
    auto attn_raw = decrypt(inf.cc(), attn_out, inf.fhe->sk());

    // Print output interleaving
    print_slots(attn_raw, S, d, H, "attn_output", 32);

    // Ground truth
    auto gt = ref_attention_output(Q_ref, K_ref, V_ref, S, d, H, given_max);
    print_slots(gt, S, d, H, "gt_output", 32);

    // Compare at tok=0 positions (where output lives)
    int t = S / d;
    double max_err = 0.0;
    for (int ld = 0; ld < d_head; ++ld)
        for (int h = 0; h < H; ++h) {
            int idx = pos(ld, h, 0, t, H);
            max_err = std::max(max_err, std::abs(attn_raw[idx] - gt[idx]));
        }

    bool ok = max_err < 1e-2;
    std::cout << (ok ? "PASS" : "FAIL") << "  max_err=" << max_err << "\n";

    std::cout << "\n" << (ok ? "ALL PASSED" : "SOME FAILURES") << "\n";
    return ok ? 0 : 1;
}
