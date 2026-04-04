#include "gpt2.h"
#include "attention.h"
#include "nonlinear.h"
#include "ckks_primitives.h"
#include <iostream>
#include <random>
#include <cmath>
#include <iomanip>
#include <numeric>

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

// ── Slot classification ────────────────────────────────────────────────
// Returns a bitmask: true = valid (nk,H) position, false = garbage
static std::vector<bool> valid_mask(int S, int nk, int H, int t) {
    int tH = t * H;
    std::vector<bool> m(S, false);
    for (int k = 0; k < nk; ++k)
        for (int h = 0; h < H; ++h)
            m[k / t * tH + h * t + k % t] = true;
    return m;
}

// ── Error report: valid vs garbage, max + avg ──────────────────────────
struct ErrorReport {
    double valid_max, valid_avg, garbage_max, garbage_avg;
    int valid_count, garbage_count;
};

static ErrorReport compute_errors(const std::vector<double>& got,
                                   const std::vector<double>& gt,
                                   const std::vector<bool>& vmask) {
    ErrorReport r{};
    double vsum = 0.0, gsum = 0.0;
    for (int i = 0; i < (int)got.size(); ++i) {
        double err = std::abs(got[i] - gt[i]);
        if (vmask[i]) {
            r.valid_max = std::max(r.valid_max, err);
            vsum += err;
            r.valid_count++;
        } else {
            r.garbage_max = std::max(r.garbage_max, err);
            gsum += err;
            r.garbage_count++;
        }
    }
    r.valid_avg   = r.valid_count   > 0 ? vsum / r.valid_count   : 0.0;
    r.garbage_avg = r.garbage_count > 0 ? gsum / r.garbage_count : 0.0;
    return r;
}

static void print_report(const char* label, int level, const ErrorReport& r) {
    std::cout << "  [" << label << "] level=" << level << "\n"
              << "    valid:   max=" << std::scientific << std::setprecision(4) << r.valid_max
              << "  avg=" << r.valid_avg
              << "  (n=" << r.valid_count << ")\n"
              << "    garbage: max=" << r.garbage_max
              << "  avg=" << r.garbage_avg
              << "  (n=" << r.garbage_count << ")\n";
}

// ── Ground truth attention output in N-slot format ─────────────────────
static std::vector<double> ref_full_attention(
        const std::vector<double>& Q,
        const std::vector<std::vector<double>>& K,
        const std::vector<std::vector<double>>& V,
        int N, int d, int H, double given_max) {
    int nk = (int)K.size(), d_head = d / H, t = N / d;
    double scale = 1.0 / std::sqrt((double)d_head);

    std::vector<std::vector<double>> qkt(nk, std::vector<double>(H, 0.0));
    for (int k = 0; k < nk; ++k)
        for (int r = 0; r < d; ++r)
            qkt[k][r % H] += Q[r] * K[k][r];
    for (auto& row : qkt) for (auto& v : row) v *= scale;

    std::vector<std::vector<double>> soft(nk, std::vector<double>(H, 0.0));
    for (int h = 0; h < H; ++h) {
        double sum = 0.0;
        for (int k = 0; k < nk; ++k) { soft[k][h] = std::exp(qkt[k][h] - given_max); sum += soft[k][h]; }
        for (int k = 0; k < nk; ++k) soft[k][h] /= sum;
    }

    std::vector<double> out(d, 0.0);
    for (int r = 0; r < d; ++r)
        for (int k = 0; k < nk; ++k)
            out[r] += soft[k][r % H] * V[k][r];

    std::vector<double> encoded(N, 0.0);
    for (int r = 0; r < d; ++r)
        encoded[pos(r / H, r % H, 0, t, H)] = out[r];
    return encoded;
}

int main() {
    const int logN = 16, d = 1024, d_exp = 4096, H = 16, nk = 5;
    int S = 1 << (logN - 1);
    int t = S / d, tH = t * H, d_head = d / H;

    auto extra_rots = compute_gpt2_rot_indices(S, d, d_exp, H, 1);
    std::cout << "=== MHA E2E Diagnostic Test ===\n";
    std::cout << "  N=" << S << " d=" << d << " H=" << H << " t=" << t
              << " d_head=" << d_head << " nk=" << nk << "\n";
    std::cout << "Creating CKKS context ...\n";

    Inference inf;
    inf.fhe = make_ckks_context(logN, 13, 41, 0, true, 50, 53, {4,3}, 0, 192, 3, 15, extra_rots);
    inf.slots = S;  inf.total_depth = 28;  inf.bench_mode = false;
    inf.size.hidDim = d;  inf.size.numHeads = H;
    prepare_mha_masks(inf);
    prepare_vcache(inf);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-0.01, 0.01);
    const SoftmaxConfig& cfg = SOFTMAX_ATTN_GPT2;

    auto vmask = valid_mask(S, nk, H, t);

    // ════════════════════════════════════════════════════════════════════
    // STEP 1: Weights
    // ════════════════════════════════════════════════════════════════════
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

    // ════════════════════════════════════════════════════════════════════
    // STEP 2: KV caches (separate encodings for K and V)
    // ════════════════════════════════════════════════════════════════════
    std::cout << "\n[2] Building KV caches ...\n";
    std::vector<std::vector<double>> K_ref(nk), V_ref(nk);
    for (int i = 0; i < nk; ++i) {
        auto x = make_vec();
        Ctx xe_k = encode_linear_input(inf, x, d, d);
        Ctx xe_v = encode_linear_input(inf, x, d, d);
        cache_k_push(inf, linear(inf, xe_k, "k", d, d));
        cache_v_push(inf, linear(inf, xe_v, "v", d, d));
        K_ref[i] = ref_matmul(x, W_Kr);
        V_ref[i] = ref_matmul(x, W_Vr);
    }
    std::cout << "  k_count=" << inf.k_count << " v_count=" << inf.v_count << "\n";

    // ════════════════════════════════════════════════════════════════════
    // STEP 3: Q projection
    // ════════════════════════════════════════════════════════════════════
    std::cout << "\n[3] Q projection ...\n";
    auto cx = make_vec();
    auto Q_ref = ref_matmul(cx, W_Qr);
    Ctx q_enc = encode_linear_input(inf, cx, d, d);
    q_enc = linear(inf, q_enc, "q", d, d);
    std::cout << "  level=" << level_of(q_enc) << "\n";

    // ════════════════════════════════════════════════════════════════════
    // STEP 4: QK^T
    // ════════════════════════════════════════════════════════════════════
    std::cout << "\n[4] QK^T ...\n";
    Ctx qkt_enc = qkt(inf, q_enc);

    // Ground-truth QKT (nk, H)
    std::vector<std::vector<double>> gt_qkt(nk, std::vector<double>(H, 0.0));
    double scale = 1.0 / std::sqrt((double)d_head);
    for (int k = 0; k < nk; ++k)
        for (int r = 0; r < d; ++r)
            gt_qkt[k][r % H] += Q_ref[r] * K_ref[k][r];
    for (auto& row : gt_qkt) for (auto& v : row) v *= scale;

    // Build full N-slot GT for comparison
    std::vector<double> gt_qkt_full(S, 0.0);
    for (int k = 0; k < nk; ++k)
        for (int h = 0; h < H; ++h)
            gt_qkt_full[k / t * tH + h * t + k % t] = gt_qkt[k][h];

    {
        auto dec = decrypt(inf.cc(), qkt_enc, inf.fhe->sk());
        auto r = compute_errors(dec, gt_qkt_full, vmask);
        print_report("QKT", level_of(qkt_enc), r);
    }

    // ════════════════════════════════════════════════════════════════════
    // STEP 5: Oracle max
    // ════════════════════════════════════════════════════════════════════
    double given_max = 0.0;
    for (auto& row : gt_qkt) for (auto v : row) given_max = std::max(given_max, std::abs(v));
    given_max *= 1.5;
    double mask_max = std::max(given_max, std::pow(2.0, cfg.exp_r - 1));

    std::cout << "\n[5] Softmax params:\n"
              << "  given_max=" << std::scientific << given_max
              << "  mask_max=" << mask_max << "\n"
              << "  btp_min=" << cfg.btp_min_remaining
              << "  gs_iters=" << cfg.gs_inv_iters
              << "  alpha=" << 1.0/nk << "\n";

    // alpha*sum diagnostic
    std::cout << "  alpha*sum: ";
    for (int h = 0; h < 4; ++h) {
        double sum = 0.0;
        for (int k = 0; k < nk; ++k)
            sum += std::exp(gt_qkt[k][h] - given_max);
        std::cout << "h" << h << "=" << std::fixed << std::setprecision(4) << sum/nk << " ";
    }
    std::cout << "\n";

    // ════════════════════════════════════════════════════════════════════
    // STEP 6: Softmax step-by-step (mirrors attention_softmax exactly)
    // ════════════════════════════════════════════════════════════════════
    std::cout << "\n[6] Softmax pipeline ...\n";

    auto log_depth = [&](const Ctx& ct, const char* tag) {
        uint32_t consumed = level_of(ct);
        uint32_t remaining = (uint32_t)inf.total_depth > consumed
                           ? (uint32_t)inf.total_depth - consumed : 0u;
        std::cout << "    depth: " << tag << " consumed=" << consumed
                  << " remaining=" << remaining << "\n";
    };

    auto maybe_btp = [&](Ctx& ct, const char* tag) {
        if (cfg.btp_min_remaining == 0) return;
        uint32_t consumed  = level_of(ct);
        uint32_t remaining = ((uint32_t)inf.total_depth > consumed)
                             ? ((uint32_t)inf.total_depth - consumed) : 0u;
        if (remaining < cfg.btp_min_remaining) {
            ct = bootstrap_to(inf, ct, cfg.btp_min_remaining);
            std::cout << "    [BTP] " << tag << " → level=" << level_of(ct) << "\n";
        }
    };

    // ── 6a: Combined mask + shift ──
    std::cout << "  [6a] mask + shift\n";
    std::vector<double> mask(S, -mask_max - given_max);
    for (int h = 0; h < H; ++h)
        for (int tok = 0; tok < nk; ++tok)
            mask[tok / t * tH + h * t + tok % t] = -given_max;
    Ptx mask_pt = inf.cc()->MakeCKKSPackedPlaintext(mask);
    Ctx ct = inf.cc()->EvalAdd(qkt_enc, mask_pt);

    std::vector<double> gt_masked(S);
    for (int i = 0; i < S; ++i) gt_masked[i] = gt_qkt_full[i] + mask[i];

    {
        auto dec = decrypt(inf.cc(), ct, inf.fhe->sk());
        auto r = compute_errors(dec, gt_masked, vmask);
        print_report("mask+shift", level_of(ct), r);
    }

    // ── 6b: exp_approx ──
    std::cout << "  [6b] exp_approx\n";
    log_depth(ct, "pre-exp input");
    maybe_btp(ct, "pre-exp");
    Ctx e = exp_approx(inf.cc(), ct, cfg.exp_r);

    std::vector<double> gt_exp(S);
    for (int i = 0; i < S; ++i) gt_exp[i] = std::exp(gt_masked[i]);

    {
        auto dec = decrypt(inf.cc(), e, inf.fhe->sk());
        auto r = compute_errors(dec, gt_exp, vmask);
        print_report("exp_approx", level_of(e), r);

        // Print head 0 values
        std::cout << "    head0: ";
        for (int k = 0; k < nk; ++k) {
            int idx = k % t;
            std::cout << "k" << k << "=" << std::scientific << std::setprecision(3)
                      << dec[idx] << "/" << gt_exp[idx] << " ";
        }
        std::cout << "\n";
    }

    // ── 6c: head_reduce_sum ──
    std::cout << "  [6c] head_reduce_sum\n";
    DepthGuard hrs_dg;
    if (!cfg.gs_btp_schedule.empty() || cfg.gs_btp_min_remaining > 0) {
        hrs_dg.refresh = [&](const Ctx& c) {
            return bootstrap_to(inf, c, (uint32_t)cfg.btp_target_level);
        };
        hrs_dg.total_depth   = (uint32_t)inf.total_depth;
        hrs_dg.min_remaining = cfg.gs_btp_min_remaining;
    }
    log_depth(e, "pre-HRS input");
    maybe_btp(e, "pre-head-reduce");
    Ctx s = head_reduce_sum(inf, e, hrs_dg);

    // GT: per-head sums
    double gt_head_sums[16];
    for (int h = 0; h < H; ++h) {
        gt_head_sums[h] = 0.0;
        for (int k = 0; k < nk; ++k)
            gt_head_sums[h] += gt_exp[k / t * tH + h * t + k % t];
    }

    {
        auto dec = decrypt(inf.cc(), s, inf.fhe->sk());
        // HRS replicates sum across all tok slots within each head
        // Check at tok=0 per head
        double max_err = 0.0, sum_err = 0.0;
        for (int h = 0; h < H; ++h) {
            int idx = h * t;
            double err = std::abs(dec[idx] - gt_head_sums[h]);
            max_err = std::max(max_err, err);
            sum_err += err;
        }
        std::cout << "  [head_reduce_sum] level=" << level_of(s) << "\n"
                  << "    per-head: max=" << std::scientific << max_err
                  << "  avg=" << sum_err / H << "\n";
        std::cout << "    sums: ";
        for (int h = 0; h < 4; ++h)
            std::cout << "h" << h << "=" << std::fixed << std::setprecision(4)
                      << dec[h * t] << "/" << gt_head_sums[h] << " ";
        std::cout << "...\n";
    }

    // ── 6d: goldschmidt_inv ──
    std::cout << "  [6d] goldschmidt_inv\n";
    double alpha = 1.0 / (double)nk;
    Ctx inv_init = encrypt_const(inf.cc(), alpha, (size_t)S, inf.fhe->pk());
    log_depth(s, "pre-inv input");
    maybe_btp(s, "pre-inv");

    DepthGuard gs_dg;
    if (!cfg.gs_btp_schedule.empty() || cfg.gs_btp_min_remaining > 0) {
        gs_dg.refresh = [&](const Ctx& c) {
            return bootstrap_to(inf, c, (uint32_t)cfg.btp_target_level);
        };
        gs_dg.schedule      = cfg.gs_btp_schedule;
        gs_dg.total_depth   = (uint32_t)inf.total_depth;
        gs_dg.min_remaining = cfg.gs_btp_min_remaining;
    }
    Ctx inv_s = goldschmidt_inv(inf.cc(), s, inv_init, cfg.gs_inv_iters, gs_dg);

    {
        auto dec = decrypt(inf.cc(), inv_s, inf.fhe->sk());
        double max_err = 0.0, sum_err = 0.0;
        for (int h = 0; h < H; ++h) {
            int idx = h * t;
            double gt_inv = 1.0 / gt_head_sums[h];
            double err = std::abs(dec[idx] - gt_inv);
            max_err = std::max(max_err, err);
            sum_err += err;
        }
        std::cout << "  [goldschmidt_inv] level=" << level_of(inv_s) << "\n"
                  << "    per-head: max=" << std::scientific << max_err
                  << "  avg=" << sum_err / H << "\n";
    }

    // ── 6e: final multiply (e * inv_s) → softmax ──
    std::cout << "  [6e] final multiply\n";
    log_depth(e, "numerator (e)");
    maybe_btp(e, "pre-final-mult");
    Ctx soft_result = inf.cc()->EvalMult(e, inv_s);

    // GT softmax
    std::vector<double> gt_soft(S, 0.0);
    for (int k = 0; k < nk; ++k)
        for (int h = 0; h < H; ++h) {
            int idx = k / t * tH + h * t + k % t;
            gt_soft[idx] = gt_exp[idx] / gt_head_sums[h];
        }

    {
        auto dec = decrypt(inf.cc(), soft_result, inf.fhe->sk());
        auto r = compute_errors(dec, gt_soft, vmask);
        print_report("softmax", level_of(soft_result), r);

        // Sum-to-1 check
        std::cout << "    sum-to-1: ";
        for (int h = 0; h < 4; ++h) {
            double s_val = 0.0;
            for (int k = 0; k < nk; ++k)
                s_val += dec[k / t * tH + h * t + k % t];
            std::cout << "h" << h << "=" << std::fixed << std::setprecision(4) << s_val << " ";
        }
        std::cout << "\n";
    }

    // ════════════════════════════════════════════════════════════════════
    // STEP 7: softmax @ V
    // ════════════════════════════════════════════════════════════════════
    std::cout << "\n[7] softmax @ V ...\n";
    log_depth(soft_result, "softmax input to V");
    Ctx attn_out = softmax_v(inf, soft_result);
    auto attn_raw = decrypt(inf.cc(), attn_out, inf.fhe->sk());

    auto gt = ref_full_attention(Q_ref, K_ref, V_ref, S, d, H, given_max);

    // tok=0 mask for output comparison
    std::vector<bool> tok0_mask(S, false);
    for (int ld = 0; ld < d_head; ++ld)
        for (int h = 0; h < H; ++h)
            tok0_mask[pos(ld, h, 0, t, H)] = true;

    {
        auto r = compute_errors(attn_raw, gt, tok0_mask);
        print_report("attn_output", level_of(attn_out), r);
    }

    // ════════════════════════════════════════════════════════════════════
    // STEP 8: Black-box comparison
    // ════════════════════════════════════════════════════════════════════
    std::cout << "\n[8] Black-box attention_softmax + softmax_v ...\n";
    Ctx bb_soft = attention_softmax(inf, qkt_enc, nk, given_max);
    Ctx bb_out  = softmax_v(inf, bb_soft);
    auto bb_raw = decrypt(inf.cc(), bb_out, inf.fhe->sk());

    {
        auto r = compute_errors(bb_raw, gt, tok0_mask);
        print_report("black-box", level_of(bb_out), r);
    }

    // ════════════════════════════════════════════════════════════════════
    // FINAL VERDICT
    // ════════════════════════════════════════════════════════════════════
    double max_err = 0.0;
    for (int ld = 0; ld < d_head; ++ld)
        for (int h = 0; h < H; ++h) {
            int idx = pos(ld, h, 0, t, H);
            max_err = std::max(max_err, std::abs(attn_raw[idx] - gt[idx]));
        }

    double bb_max_err = 0.0;
    for (int ld = 0; ld < d_head; ++ld)
        for (int h = 0; h < H; ++h) {
            int idx = pos(ld, h, 0, t, H);
            bb_max_err = std::max(bb_max_err, std::abs(bb_raw[idx] - gt[idx]));
        }

    std::cout << "\n════════════════════════════════════════\n";
    std::cout << "  step-by-step: max_err=" << std::scientific << max_err
              << "  " << (max_err < 1e-2 ? "PASS" : "FAIL") << "\n";
    std::cout << "  black-box:    max_err=" << bb_max_err
              << "  " << (bb_max_err < 1e-2 ? "PASS" : "FAIL") << "\n";
    std::cout << "════════════════════════════════════════\n";

    return (max_err < 1e-2 && bb_max_err < 1e-2) ? 0 : 1;
}
