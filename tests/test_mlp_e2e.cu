#include "gpt2.h"
#include "nonlinear.h"
#include "ckks_primitives.h"
#include <iostream>
#include <random>
#include <cmath>

// ── Plaintext references ────────────────────────────────────────────────────

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

// Check FHE output vs reference at stride-t positions, return max error
static double check_stage(const std::vector<double>& dec,
                           const std::vector<double>& ref,
                           int t, int d_check, const char* label,
                           int print_dims = 4) {
    double max_err = 0.0;
    for (int r = 0; r < d_check; ++r) {
        int slot = r * t;
        double err = std::abs(dec[slot] - ref[r]);
        max_err = std::max(max_err, err);
        if (r < print_dims)
            std::cout << "    dim " << r << ": dec=" << std::scientific << dec[slot]
                      << "  ref=" << ref[r] << "  err=" << err << "\n";
    }
    std::cout << "  " << label << " max_err=" << std::scientific << max_err << "\n";
    return max_err;
}

// For up-proj output (d_exp dims at interleaved positions)
// CacheMir up output: slot at position interleave_idx(m, d, d_exp) * tp
// But we can also check at stride tp positions against interleave-permuted ref.
static double check_stage_up(const std::vector<double>& dec,
                              const std::vector<double>& ref,
                              int N, int d, int d_exp,
                              const char* label, int print_dims = 4) {
    auto p = compute_cm_params(N, d, d_exp);
    double max_err = 0.0;
    for (int m = 0; m < d_exp; ++m) {
        int slot = m * p.tp;
        int ref_idx = interleave_idx(m, d, d_exp);
        double err = std::abs(dec[slot] - ref[ref_idx]);
        max_err = std::max(max_err, err);
        if (m < print_dims)
            std::cout << "    m=" << m << " (ref_dim=" << ref_idx
                      << "): dec=" << std::scientific << dec[slot]
                      << "  ref=" << ref[ref_idx] << "  err=" << err << "\n";
    }
    std::cout << "  " << label << " max_err=" << std::scientific << max_err << "\n";
    return max_err;
}

// ── Test ────────────────────────────────────────────────────────────────────

int main() {
    const int logN = 16, d = 1024, d_exp = 4096, H = 16;
    int S = 1 << (logN - 1);
    int t = S / d;

    auto extra_rots = compute_gpt2_rot_indices(S, d, d_exp, H, 1);
    std::cout << "=== MLP E2E Test (dissected) ===\n"
              << "  N=" << S << " d=" << d << " d_exp=" << d_exp
              << " t=" << t << "\n"
              << "Creating CKKS context ...\n";

    Inference inf;
    inf.fhe = make_ckks_context(logN, 13, 41, 0, true, 50, 53, {4,3}, 0, 192, 3, 15, extra_rots);
    inf.slots = S;  inf.total_depth = 28;  inf.bench_mode = false;
    inf.size.hidDim = d;  inf.size.expDim = d_exp;  inf.size.numHeads = H;
    inf.size.dim = d;  // getRealHidDim() == hidDim, no padding

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> xdist(-1.0, 1.0);

    auto rand_mat = [&](int r, int c) {
        // Xavier-scale init so matmul outputs stay bounded
        double scale = 1.0 / std::sqrt((double)r);
        std::vector<std::vector<double>> W(r, std::vector<double>(c));
        for (auto& row : W) for (auto& v : row) v = xdist(rng) * scale;
        return W;
    };

    // ── [1] Weights ──
    std::cout << "\n[1] Encoding weights ...\n";
    auto W_up   = rand_mat(d, d_exp);
    auto W_down = rand_mat(d_exp, d);
    inf.w["up"]   = encode_weight_matrix(inf, W_up,   d, d_exp);
    inf.w["down"] = encode_weight_matrix(inf, W_down, d_exp, d);
    std::cout << "  up:   " << inf.w["up"].size()   << " plaintexts\n";
    std::cout << "  down: " << inf.w["down"].size() << " plaintexts\n";

    // ── [2] Input ──
    std::cout << "\n[2] Encoding input ...\n";
    // variance ≈ 1/3 ≈ 0.33, compatible with NORM_ENCLLM_GPT2 taylor_z0=0.5
    std::vector<double> x(d);
    for (auto& v : x) v = dist(rng);

    Ctx x_enc = encode_linear_input(inf, x, d, d);
    std::cout << "  input level=" << level_of(x_enc) << "\n";

    // ── [3] Plaintext reference (all stages) ──
    std::cout << "\n[3] Computing plaintext reference ...\n";
    auto x_norm_ref = ref_layernorm(x);
    auto up_ref     = ref_matmul(x_norm_ref, W_up);
    auto gelu_ref   = ref_gelu(up_ref);
    auto down_ref   = ref_matmul(gelu_ref, W_down);
    std::vector<double> mlp_ref(d);
    for (int i = 0; i < d; ++i) mlp_ref[i] = down_ref[i] + x[i];

    // ── [4] FHE: Norm ──
    std::cout << "\n[4] Norm ...\n";
    NormConfig norm_cfg = NORM_ENCLLM_GPT2;
    Ctx norm_enc = norm(inf, x_enc, /*target_level=*/14, norm_cfg);
    std::cout << "  post-norm level=" << level_of(norm_enc) << "\n";
    {
        auto dec = decrypt(inf.cc(), norm_enc, inf.fhe->sk());
        check_stage(dec, x_norm_ref, t, d, "norm", 4);
    }

    // ── [5] FHE: Up projection ──
    std::cout << "\n[5] Up projection (d=" << d << " -> d_exp=" << d_exp << ") ...\n";
    Ctx up_enc = linear(inf, norm_enc, "up", d, d_exp);
    std::cout << "  post-up level=" << level_of(up_enc) << "\n";
    {
        auto dec = decrypt(inf.cc(), up_enc, inf.fhe->sk());
        check_stage_up(dec, up_ref, S, d, d_exp, "up_proj", 4);
    }

    // ── [6] Bootstrap before GeLU ──
    std::cout << "\n[6] Bootstrap before GeLU ...\n";
    up_enc = bootstrap_to(inf, up_enc, 13);
    std::cout << "  post-btp level=" << level_of(up_enc) << "\n";

    // ── [7] FHE: GeLU ──
    std::cout << "\n[7] GeLU ...\n";
    GeluConfig gelu_cfg = GELU_ENCLLM_GPT2;
    Ctx gelu_enc = gelu(inf, up_enc, gelu_cfg);
    std::cout << "  post-gelu level=" << level_of(gelu_enc) << "\n";
    {
        auto dec = decrypt(inf.cc(), gelu_enc, inf.fhe->sk());
        check_stage_up(dec, gelu_ref, S, d, d_exp, "gelu", 4);
    }

    // ── [8] Bootstrap before down proj ──
    std::cout << "\n[8] Bootstrap before down proj ...\n";
    gelu_enc = bootstrap_to(inf, gelu_enc, 9);
    std::cout << "  post-btp level=" << level_of(gelu_enc) << "\n";

    // ── [9] FHE: Down projection ──
    std::cout << "\n[9] Down projection (d_exp=" << d_exp << " -> d=" << d << ") ...\n";
    Ctx down_enc = linear(inf, gelu_enc, "down", d_exp, d);
    std::cout << "  post-down level=" << level_of(down_enc) << "\n";
    {
        auto dec = decrypt(inf.cc(), down_enc, inf.fhe->sk());
        check_stage(dec, down_ref, t, d, "down_proj", 4);
    }

    // ── [10] Residual ──
    std::cout << "\n[10] Residual connection ...\n";
    Ctx res_enc = inf.cc()->EvalAdd(down_enc, x_enc);
    std::cout << "  post-residual level=" << level_of(res_enc) << "\n";
    double res_err;
    {
        auto dec = decrypt(inf.cc(), res_enc, inf.fhe->sk());
        res_err = check_stage(dec, mlp_ref, t, d, "residual", 8);
    }

    std::cout << "\n════════════════════════════════════════\n"
              << "  mlp max_err=" << std::scientific << res_err
              << "  " << (res_err < 1.0 ? "PASS" : "FAIL") << "\n"
              << "════════════════════════════════════════\n";

    return (res_err < 1.0) ? 0 : 1;
}
