#include "gpt2.h"
#include "nonlinear.h"
#include "ckks_primitives.h"
#include <cmath>
#include <vector>
#include <functional>
#include <iostream>
#include <iomanip>

/// @brief Encrypted sign approximation via composite minimax polynomials.
Ctx sign(Inference& inf, const Ctx& x_in, const DepthGuard& dg) {
    const CC& cc = inf.cc();
    const int S  = inf.slots;

    const std::vector<double> F4 = {
        0.0,
         315.0 / 128.0,
         0.0,
        -420.0 / 128.0,
         0.0,
         378.0 / 128.0,
         0.0,
        -180.0 / 128.0,
         0.0,
          35.0 / 128.0
    };

    const std::vector<double> G4 = {
        0.0,
         5850.0 / 1024.0,
         0.0,
        -34974.0 / 1024.0,
         0.0,
         97015.0 / 1024.0,
         0.0,
       -113492.0 / 1024.0,
         0.0,
         46623.0 / 1024.0
    };

    // Composition: F4(F4(G4(G4(x))))
    // Each PS eval consumes ~4 levels. With L=13, we need bootstraps between evals.
    Ctx h = eval_polynomial_ps(cc, x_in, G4, inf.fhe->pk(), (size_t)S);
    h = dg(h, 0);  // bootstrap if remaining < min_remaining
    h     = eval_polynomial_ps(cc, h,    G4, inf.fhe->pk(), (size_t)S);
    h = dg(h, 1);
    h     = eval_polynomial_ps(cc, h,    F4, inf.fhe->pk(), (size_t)S);
    h = dg(h, 2);
    h     = eval_polynomial_ps(cc, h,    F4, inf.fhe->pk(), (size_t)S);

    return h;
}

/// @brief Homomorphic less-than: returns ~1 if x < value, ~0 otherwise.
/// @param rescale_factor  Scales (x - value) into [-1, 1] before sign. Use 1.0 when the input is already in that range.
Ctx lt_function(Inference& inf, const Ctx& x_in, double value,
                double rescale_factor, const DepthGuard& dg) {
    const CC& cc = inf.cc();
    auto x = x_in->Clone();
    Ctx y = cc->EvalAdd(x, -value);
    if (rescale_factor != 1.0)
        cc->EvalMultInPlace(y, rescale_factor);

    Ctx s = sign(inf, y, dg);

    // (1 - s) / 2    
    Ctx res = cc->EvalNegate(s);
    cc->EvalAddInPlace(res, 1.0);
    cc->EvalMultInPlace(res, 0.5);

    return res;
}

/// @brief GELU activation via piecewise polynomial approximation.
///   x < -4          ->  0
///   -4  <= x < -1.95 ->  poly_f0(x)   (degree-3 minimax)
///   -1.95 <= x < 3   ->  poly_f1(x)   (degree-6 minimax)
///   x >= 3           ->  x
/// @param rescale_factor  Scalar applied to (x - threshold) before sign, so that the shifted input lies in [-1, 1] as required
Ctx gelu(Inference& inf, const Ctx& x_in, const GeluConfig& cfg) {
    const double rescale_factor = cfg.rescale_factor;
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

    uint32_t sign_btp_target = (cfg.btp_target_level > 0)
                                   ? (uint32_t)cfg.btp_target_level
                                   : (uint32_t)inf.total_depth;
    DepthGuard sign_dg;
    if (cfg.sign_btp_min_remaining > 0 && inf.total_depth > 0) {
        sign_dg.refresh = [&, sign_btp_target](const Ctx& ct) {
            return bootstrap_to(inf, ct, sign_btp_target);
        };
        sign_dg.total_depth   = (uint32_t)inf.total_depth;
        sign_dg.min_remaining = cfg.sign_btp_min_remaining;
    }

    Ctx lt_m4, lt_m195, lt_p3;

    lt_m4   = lt_function(inf, x_in, -4.0,  rescale_factor, sign_dg);
    lt_m195 = lt_function(inf, x_in, -1.95, rescale_factor, sign_dg);
    lt_p3   = lt_function(inf, x_in,  3.0,  rescale_factor, sign_dg);

    if (cfg.bootstrap_indicators) {
        if (cfg.btp_target_level > 0) {
            lt_m4   = bootstrap_to(inf, lt_m4,   (uint32_t)cfg.btp_target_level);
            lt_m195 = bootstrap_to(inf, lt_m195, (uint32_t)cfg.btp_target_level);
            lt_p3   = bootstrap_to(inf, lt_p3,   (uint32_t)cfg.btp_target_level);
        } else {
            lt_m4   = cc->EvalBootstrap(lt_m4);
            lt_m195 = cc->EvalBootstrap(lt_m195);
            lt_p3   = cc->EvalBootstrap(lt_p3);
        }
    }

    Ctx ind_f0  = cc->EvalSub(lt_m195, lt_m4);
    Ctx ind_f1  = cc->EvalSub(lt_p3,   lt_m195);
    Ctx ind_lin = cc->EvalNegate(lt_p3);
    cc->EvalAddInPlace(ind_lin, 1.0);

    Ctx p0 = eval_polynomial_ps(cc, x_in, poly_f0, inf.fhe->pk(), (size_t)S);
    Ctx p1 = eval_polynomial_ps(cc, x_in, poly_f1, inf.fhe->pk(), (size_t)S);

    Ctx term0 = cc->EvalMult(ind_f0, p0);
    Ctx term1 = cc->EvalMult(ind_f1, p1);

    Ctx x_copy = x_in;
    Ctx term2 = cc->EvalMult(ind_lin, x_copy);

    cc->EvalAddInPlace(term0, term1);    // ← was missing
    cc->EvalAddInPlace(term0, term2);    // ← was missing

    int eD = inf.size.expDim;
    int tp = S / eD;
    if (tp > 1) {
        Ptx final_mask = inf.encode_stride_mask(eD, tp, 1.0);
        term0 = cc->EvalMult(term0, final_mask);
    }

    return term0;
}

Ctx norm(Inference& inf, const Ctx& x_in,
         int target_level_after_btp, const NormConfig& cfg) {
    const CC& cc = inf.cc();
    const int S  = inf.slots;
    const int hD = inf.size.hidDim;
    const int rD = inf.size.getRealHidDim();
 
    Ctx mean = compute_average(inf, x_in);
    Ctx varc = compute_variance_interleaved(inf, x_in, mean);
 
    cc->EvalAddInPlace(varc, cfg.epsilon);
 
    Ctx inv_sqrt_init;
    switch (cfg.nr_init_method) {
        case NRInitMethod::LINEAR: {
            std::vector<double> coeffs_std(cfg.nr_init_coeffs.rbegin(),
                                           cfg.nr_init_coeffs.rend());
            inv_sqrt_init = eval_polynomial(cc, varc, coeffs_std);
            break;
        }
        case NRInitMethod::REMEZ: {
            double alpha = 4.0 / (cfg.gs_d_min + cfg.gs_d_max);
            double beta  = alpha * alpha / 4.0;
            auto div_init = cc->EvalMult(varc, -beta);
            cc->EvalAddInPlace(div_init, alpha);
            inv_sqrt_init = eval_rational_approx(cc, div_init,
                                  cfg.remez_p_coeffs, cfg.remez_q_coeffs,
                                  cfg.remez_q_min, cfg.remez_q_max,
                                  inf.fhe->pk(), inf.slots, cfg.remez_div_iters);
            break;
        }
        case NRInitMethod::TAYLOR: {
            double s = cfg.taylor_rescale;
            Ctx var_scaled = (s != 1.0) ? cc->EvalMult(varc, s) : varc->Clone();
            std::vector<double> taylor_c = taylor_inv_sqrt_coeffs(cfg.taylor_z0);
            inv_sqrt_init = eval_taylor_inv_sqrt(cc, var_scaled, taylor_c, cfg.taylor_z0);
            if (s != 1.0)
                cc->EvalMultInPlace(inv_sqrt_init, std::sqrt(s));
            break;
        }
    }
 
    DepthGuard dg;
    if (!cfg.nr_btp_schedule.empty() || cfg.nr_btp_min_remaining > 0) {
        dg.refresh = [&](const Ctx& ct) {
            return bootstrap_to(inf, ct, (uint32_t)target_level_after_btp);
        };
        dg.schedule      = cfg.nr_btp_schedule;
        dg.total_depth   = (uint32_t)inf.total_depth;
        dg.min_remaining = cfg.nr_btp_min_remaining;
    }
 
    Ctx inv_sqrt_varc = inv_sqrt_newton(cc, varc, inv_sqrt_init, cfg.nr_iters, dg);
 
    Ctx x_centered = x_in->Clone();
    Ctx true_mean  = cc->EvalMult(mean, 1.0 / (double)rD);
    cc->EvalSubInPlace(x_centered, true_mean);
 
    Ctx result = cc->EvalMult(x_centered, inv_sqrt_varc);
 
    return result;
}

/// @brief exp(x) ≈ (1 + x / 2^r)^{2^r}.
Ctx exp_approx(const CC& cc, const Ctx& x, int r) {
    double inv_2r = 1.0 / (double)(1 << r);
    Ctx y = cc->EvalMult(x, inv_2r);    // x / 2^r
    cc->EvalAddInPlace(y, 1.0);          // 1 + x/2^r

    for (int i = 0; i < r; ++i)          // square r times
        cc->EvalSquareInPlace(y);
    return y;
}


/// @brief Numerically-stable softmax: exp(x - max) / sum(exp(x - max)).
/// @param precomputed_max  If non-null, skip the sign-based max (oracle path).
Ctx softmax(Inference& inf, const Ctx& x_in,
             const SoftmaxConfig& cfg,
             const Ctx& precomputed_max,
             const Ptx& causal_mask) {
    const CC& cc = inf.cc();
    const int S  = inf.slots;
    const int seq_dim = cfg.seq_dim;
    const int stride  = S / seq_dim;

    Ctx x_masked;
    if (causal_mask != nullptr) {
        Ptx mask_copy = causal_mask; 
        x_masked = cc->EvalAdd(x_in, mask_copy);
        std::cout << "  [softmax] causal mask applied\n";
    } else {
        x_masked = x_in->Clone();
    }

    auto maybe_btp = [&](Ctx& ct, const char* tag) {
        if (cfg.btp_min_remaining == 0) return;
        uint32_t consumed  = level_of(ct);
        uint32_t remaining = ((uint32_t)inf.total_depth > consumed)
                             ? ((uint32_t)inf.total_depth - consumed) : 0u;
        if (remaining < cfg.btp_min_remaining) {
            ct = bootstrap_to(inf, ct, (uint32_t)cfg.btp_target_level);
            std::cout << "  [softmax] " << tag << " bootstrap, consumed="
                      << level_of(ct) << "\n";
        }
    };

    bool use_scalar_max = false;
    Ctx x_max;
    if (precomputed_max != nullptr) {
        x_max = precomputed_max->Clone();
        maybe_btp(x_max, "post-max");
    } else if (cfg.given_max_val > 0.0) {
        use_scalar_max = true;
    } else {
        // TODO: throw error as we don't support it yet
    }

    Ctx x_shifted = x_masked->Clone();
    if (use_scalar_max) {
        cc->EvalSubInPlace(x_shifted, cfg.given_max_val);
    } else {
        cc->EvalSubInPlace(x_shifted, x_max);
    }

    maybe_btp(x_shifted, "pre-exp");
    Ctx exp_ct = exp_approx(cc, x_shifted, cfg.exp_r);

    Ctx exp_sum = exp_ct->Clone();
    for (int gap = stride; gap < S; gap *= 2) {
        Ctx tmp = cc->EvalRotate(exp_sum, gap);
        cc->EvalAddInPlace(exp_sum, tmp);
    }

    double alpha = cfg.gs_inv_init / (double)(seq_dim * inf.size.hidDim);
    Ctx inv_init = encrypt_const(cc, alpha, (size_t)S, inf.fhe->pk());

    maybe_btp(exp_sum, "pre-inv");

    DepthGuard dg;
    if (!cfg.gs_btp_schedule.empty() || cfg.gs_btp_min_remaining > 0) {
        dg.refresh = [&](const Ctx& ct) {
            return bootstrap_to(inf, ct, (uint32_t)cfg.btp_target_level);
        };
        dg.schedule      = cfg.gs_btp_schedule;
        dg.total_depth   = (uint32_t)inf.total_depth;
        dg.min_remaining = cfg.gs_btp_min_remaining;
    }

    Ctx inv_sum = goldschmidt_inv(cc, exp_sum, inv_init, cfg.gs_inv_iters, dg);

    maybe_btp(exp_ct, "pre-final-mult");

    Ctx y = cc->EvalMult(exp_ct, inv_sum);
    return y;
}
