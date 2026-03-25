#include <gtest/gtest.h>
#include "test_non_linear.h"
#include <cmath>
#include <vector>

std::vector<double> linspace(double start, double end, size_t num) {
    std::vector<double> result(num);
    double step = (end - start) / (num - 1);
    for (size_t i = 0; i < num; ++i) {
        result[i] = start + i * step;
    }
    return result;
}

std::vector<double> gelu_expect(const std::vector<double>& xs) {
    std::vector<double> result(xs.size());
    for (size_t i = 0; i < xs.size(); ++i) {
        double x = xs[i];
        result[i] = 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (x + 0.044715 * std::pow(x, 3))));
    }
    return result;
}

TEST_F(NonLinearTest, SignApprox) {
    const CC& cc = inf.cc();
    const int S  = inf.slots;

    const std::vector<double> xs     = {  0.8,  0.3, -0.5, -0.9 };
    const std::vector<double> expect = {  1.0,  1.0, -1.0, -1.0 };

    for (size_t i = 0; i < xs.size(); ++i) {
        std::vector<double> msg(S, xs[i]);
        auto pt = cc->MakeCKKSPackedPlaintext(msg);
        auto ct = cc->Encrypt(inf.fhe->pk(), pt);

        auto result = sign(inf, ct);
        std::cout << "Level after sign: " << level_of(result) << std::endl;
        double got = decrypt(cc, result, inf.fhe->sk())[0];
        std::cout << "sign(" << xs[i] << ") = " << got
                  << "  (expected " << expect[i] << ")\n";

        EXPECT_NEAR(got, expect[i], 0.05);
    }
}

TEST_F(NonLinearTest, LtApprox) {
    const CC& cc = inf.cc();
    const int S  = inf.slots;

    const std::vector<double> xs     = {  0.8,  0.3, -0.5, -0.9 };
    const std::vector<double> expect = {  0.0,  0.0, 1.0, 1.0 };

    for (size_t i = 0; i < xs.size(); ++i) {
        std::vector<double> msg(S, xs[i]);
        auto pt = cc->MakeCKKSPackedPlaintext(msg);
        auto ct = cc->Encrypt(inf.fhe->pk(), pt);

        auto result = lt_function(inf, ct, 0.0, 1.0);

        double got = decrypt(cc, result, inf.fhe->sk())[0];
        std::cout << "lt_function(" << xs[i] << ") = " << got
                  << "  (expected " << expect[i] << ")\n";

        EXPECT_NEAR(got, expect[i], 0.05);
    }
}

TEST_F(NonLinearTest, LtApproxWithBts) {
    const CC& cc = inf.cc();
    const int S  = inf.slots;

    double x_val = 0.6;
    auto pt = cc->MakeCKKSPackedPlaintext(std::vector<double>(S, x_val));
    auto ct = cc->Encrypt(inf.fhe->pk(), pt);

    double thresholds[] = { -4.0, -1.95, 3.0 };
    double expects[]    = {  0.0,  0.0,  1.0 };
    double factor = 0.1;
    Ctx results[3];

    for (int i = 0; i < 3; ++i) {
        results[i] = lt_function(inf, ct, thresholds[i], factor);
        std::cout << "lt(x, " << thresholds[i] << "): level=" << level_of(results[i]);

        double got = decrypt(cc, results[i], inf.fhe->sk())[0];
        std::cout << " val=" << got << " (expected " << expects[i] << ")\n";
        EXPECT_NEAR(got, expects[i], 0.05);

        results[i] = cc->EvalBootstrap(results[i]);
        std::cout << "  post-bts level=" << level_of(results[i]) << "\n";
    }

    match_level(cc, results[0], results[1]);
    Ctx ind = cc->EvalSub(results[2], results[1]); 

    double got = decrypt(cc, ind, inf.fhe->sk())[0];
    std::cout << "\ninterval indicator [-1.95, 3) = " << got << " (expected 1)\n";
    EXPECT_NEAR(got, 1.0, 0.1);
}

TEST_F(NonLinearTest, GeLUApproxWithBts) {
    const CC& cc = inf.cc();
    const int S  = inf.slots;

    const std::vector<double> poly_f0 = {
        -0.5054031199708174,
        -0.42226581151983866,
        -0.11807612951181953,
        -0.011034134030615728
    };
    const std::vector<double> poly_f1 = {
        0.008526321541038084,
        0.5,
        0.3603292692789629,
        0.0,
       -0.037688200365904236,
        0.0,
        0.0018067462606141187
    };

    const double factor = 0.1;

    const std::vector<double> xs = { -3.0, -0.5, 0.0, 1.0, 4.0 };
    const std::vector<double> expect = gelu_expect(xs);

    for (size_t t = 0; t < xs.size(); ++t) {
        auto pt = cc->MakeCKKSPackedPlaintext(std::vector<double>(S, xs[t]));
        auto ct = cc->Encrypt(inf.fhe->pk(), pt);
        std::cout << "\n--- gelu(" << xs[t] << ") ---\n";

        Ctx lt_m4   = lt_function(inf, ct, -4.0,  factor);
        std::cout << "  lt_m4   level=" << level_of(lt_m4) << "\n";
        Ctx lt_m195 = lt_function(inf, ct, -1.95, factor);
        std::cout << "  lt_m195 level=" << level_of(lt_m195) << "\n";
        Ctx lt_p3   = lt_function(inf, ct,  3.0,  factor);
        std::cout << "  lt_p3   level=" << level_of(lt_p3) << "\n";

        lt_m4   = cc->EvalBootstrap(lt_m4);
        lt_m195 = cc->EvalBootstrap(lt_m195);
        lt_p3   = cc->EvalBootstrap(lt_p3);
        std::cout << "  post-bts levels: " << level_of(lt_m4)
                  << ", " << level_of(lt_m195)
                  << ", " << level_of(lt_p3) << "\n";

        match_level(cc, lt_m4, lt_m195);
        match_level(cc, lt_m195, lt_p3);
        Ctx ind_f0  = cc->EvalSub(lt_m195, lt_m4);    // [-4, -1.95)
        Ctx ind_f1  = cc->EvalSub(lt_p3, lt_m195);    // [-1.95, 3)
        Ctx ind_lin = cc->EvalNegate(lt_p3);           // [3, +inf)
        cc->RescaleInPlace(ind_lin);
        cc->EvalAddInPlace(ind_lin, 1.0);

        Ctx p0 = eval_polynomial_ps(cc, ct, poly_f0, inf.fhe->pk(), (size_t)S);
        Ctx p1 = eval_polynomial_ps(cc, ct, poly_f1, inf.fhe->pk(), (size_t)S);
        std::cout << "  poly levels: p0=" << level_of(p0)
                  << " p1=" << level_of(p1) << "\n";

        match_level(cc, ind_f0, p0);
        Ctx term0 = cc->EvalMult(ind_f0, p0);
        cc->RescaleInPlace(term0);

        match_level(cc, ind_f1, p1);
        Ctx term1 = cc->EvalMult(ind_f1, p1);
        cc->RescaleInPlace(term1);

        Ctx x_copy = ct;
        match_level(cc, ind_lin, x_copy);
        Ctx term2 = cc->EvalMult(ind_lin, x_copy);
        cc->RescaleInPlace(term2);

        match_level(cc, term0, term1);
        cc->EvalAddInPlace(term0, term1);
        match_level(cc, term0, term2);
        cc->EvalAddInPlace(term0, term2);

        double got = decrypt(cc, term0, inf.fhe->sk())[0];
        std::cout << "  result=" << got << " (expected " << expect[t] << ")\n";
        EXPECT_NEAR(got, expect[t], 0.15);
    }
}

// GPT-2 hidden dim 768 padded to next power-of-2 for CKKS packing.
// With logN=12 → S=2048, S/hD=2 repeats per ciphertext.
// Input layout: [h0..h1023, h0..h1023] — two copies of the hidden vector.
static constexpr int GPT2_HD = 1024;

static std::vector<double> make_norm_input(int S, int hD) {
    std::vector<double> x(S);
    for (int i = 0; i < S; ++i) {
        int pos = i % hD;
        x[i] = (pos < hD / 2) ? 0.0 : 2.0;
    }
    return x;
}

// Helper: run norm with explicit level-ladder printouts at each stage.
// Mirrors the internal steps of norm() so we can observe levels consumed.
static void run_norm_with_level_ladder(
        Inference& inf, const Ctx& ct,
        int target_btp, const NormConfig& cfg, const char* label) {
    const CC& cc = inf.cc();
    const int S  = inf.slots;
    const int hD = inf.size.hidDim;

    std::cout << "\n=== LayerNorm (" << label << ")  hidDim=" << hD
              << "  S=" << S << "  repeats=" << S / hD << " ===\n";
    std::cout << "  [L0] input                  level=" << level_of(ct) << "\n";

    // ── mean ──
    Ctx mean = compute_average(inf, ct);
    std::cout << "  [L1] after compute_average  level=" << level_of(mean)
              << "  (rotations only, free)\n";

    // ── variance ──
    Ctx varc = compute_variance(inf, ct, mean);
    std::cout << "  [L2] after compute_variance level=" << level_of(varc)
              << "  (x*hD + square + sum + scale = ~4 levels)\n";

    // ── initial guess for 1/sqrt(varc) ──
    std::vector<double> coeffs_std(cfg.nr_init_coeffs.rbegin(),
                                    cfg.nr_init_coeffs.rend());
    Ctx inv_sqrt_init;
    switch (cfg.nr_init_method) {
        case NRInitMethod::LINEAR:
            inv_sqrt_init = eval_polynomial(cc, varc, coeffs_std);
            std::cout << "  [L3] after LINEAR init      level=" << level_of(inv_sqrt_init)
                      << "  (degree-1 poly = 1 level)\n";
            break;
        case NRInitMethod::REMEZ:
            inv_sqrt_init = eval_rational_approx(
                cc, varc, coeffs_std, cfg.remez_q_coeffs,
                cfg.remez_q_min, cfg.remez_q_max,
                inf.fhe->pk(), (size_t)S, cfg.remez_div_iters);
            std::cout << "  [L3] after REMEZ init       level=" << level_of(inv_sqrt_init)
                      << "  (poly + goldschmidt_inv)\n";
            break;
        case NRInitMethod::TAYLOR: {
            auto tc = taylor_inv_sqrt_coeffs(cfg.taylor_z0);
            inv_sqrt_init = eval_taylor_inv_sqrt(cc, varc, tc, cfg.taylor_z0);
            std::cout << "  [L3] after TAYLOR init      level=" << level_of(inv_sqrt_init)
                      << "  (shift + degree-3 poly)\n";
            break;
        }
    }
    std::cout << "  [L3] varc before bootstrap  level=" << level_of(varc) << "\n";

    // ── bootstrap ──
    varc          = bootstrap_to(inf, varc, (uint32_t)target_btp);
    inv_sqrt_init = bootstrap_to(inf, inv_sqrt_init, (uint32_t)target_btp);
    std::cout << "  [L4] after bootstrap        varc level=" << level_of(varc)
              << "  init level=" << level_of(inv_sqrt_init)
              << "  (target=" << target_btp << ")\n";

    // ── Goldschmidt refinement ──
    Ctx inv_sqrt_varc = goldschmidt_inv_sqrt(cc, varc, inv_sqrt_init, cfg.gs_iters);
    std::cout << "  [L5] after goldschmidt_inv_sqrt(" << cfg.gs_iters << " iters)  level="
              << level_of(inv_sqrt_varc) << "\n";

    // ── centering: (x - mean/hD) ──
    Ctx x_centered = ct->Clone();
    Ctx true_mean  = cc->EvalMult(mean, 1.0 / (double)hD);
    cc->RescaleInPlace(true_mean);
    std::cout << "  [L6] true_mean (mean/hD)    level=" << level_of(true_mean) << "\n";

    match_level(cc, x_centered, true_mean);
    cc->EvalSubInPlace(x_centered, true_mean);
    std::cout << "  [L7] x_centered (x - mean)  level=" << level_of(x_centered) << "\n";

    // ── final multiply ──
    match_level(cc, x_centered, inv_sqrt_varc);
    Ctx result = cc->EvalMult(x_centered, inv_sqrt_varc);
    cc->RescaleInPlace(result);
    std::cout << "  [L8] output (x-mean)/sqrt(v) level=" << level_of(result) << "\n";

    // ── verify ──
    auto vals = decrypt(cc, result, inf.fhe->sk());
    double got_lo = vals[0];
    double got_hi = vals[GPT2_HD / 2];
    std::cout << "  result[0]="    << got_lo << " (expected -1)\n";
    std::cout << "  result[hD/2]=" << got_hi << " (expected  1)\n";

    EXPECT_NEAR(got_lo, -1.0, 0.2);
    EXPECT_NEAR(got_hi,  1.0, 0.2);
}

TEST_F(NonLinearTest, LayerNormLinear) {
    inf.size.hidDim = GPT2_HD;
    const CC& cc = inf.cc();
    auto x  = make_norm_input(inf.slots, GPT2_HD);
    auto pt = cc->MakeCKKSPackedPlaintext(x);
    auto ct = cc->Encrypt(inf.fhe->pk(), pt);

    NormConfig cfg;
    cfg.nr_init_method = NRInitMethod::LINEAR;
    cfg.nr_init_coeffs = { -0.5, 1.5 };  // 1/sqrt(x) ≈ 1.5 - 0.5*x near x=1
    cfg.nr_iters       = 4;
    cfg.gs_iters       = 2;

    run_norm_with_level_ladder(inf, ct, /*target_btp=*/9, cfg, "LINEAR");
}

TEST_F(NonLinearTest, LayerNormTaylor) {
    inf.size.hidDim = GPT2_HD;
    const CC& cc = inf.cc();
    auto x  = make_norm_input(inf.slots, GPT2_HD);
    auto pt = cc->MakeCKKSPackedPlaintext(x);
    auto ct = cc->Encrypt(inf.fhe->pk(), pt);

    NormConfig cfg;
    cfg.nr_init_method = NRInitMethod::TAYLOR;
    cfg.taylor_z0      = 1.0;
    cfg.nr_init_coeffs = {};
    cfg.nr_iters       = 4;
    cfg.gs_iters       = 2;

    run_norm_with_level_ladder(inf, ct, /*target_btp=*/9, cfg, "TAYLOR");
}

TEST_F(NonLinearTest, LayerNormRemez) {
    inf.size.hidDim = GPT2_HD;
    const CC& cc = inf.cc();
    auto x  = make_norm_input(inf.slots, GPT2_HD);
    auto pt = cc->MakeCKKSPackedPlaintext(x);
    auto ct = cc->Encrypt(inf.fhe->pk(), pt);

    // Precomputed Remez (3,1) rational for 1/sqrt(x) over [0.5, 2.0]
    NormConfig cfg;
    cfg.nr_init_method  = NRInitMethod::REMEZ;
    cfg.nr_init_coeffs  = {
        1.069785284789309e-01,
       -8.418900522536490e-01,
        4.471840402707294e+00,
        4.474085956084137e+00
    };
    cfg.remez_q_coeffs  = { 1.0, 7.211233906909211e+00 };
    cfg.remez_q_min     = 4.605616953454605e+00;
    cfg.remez_q_max     = 1.542246781381842e+01;
    cfg.remez_div_iters = 5;
    cfg.nr_iters        = 4;
    cfg.gs_iters        = 2;

    run_norm_with_level_ladder(inf, ct, /*target_btp=*/9, cfg, "REMEZ");
}
