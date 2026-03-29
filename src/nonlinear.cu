#include "gpt2.h"
#include "nonlinear.h"
#include "ckks_primitives.h"
#include <cmath>
#include <vector>
#include <functional>
#include <iostream>
#include <iomanip>

static Ptx const_pt(const CC& cc, double val, int slots, uint32_t level) {
    return encode_const(cc, val, (size_t)slots, (int)level);
}


static void dbg_lvl(const char* label, const Ctx& ct) {
    std::cout << "  [DBG] " << label
              << "  consumed=" << level_of(ct) << "\n";
}

static void dbg_val(const char* label, const CC& cc,
                    Ctx ct,                          // by value (copy for decrypt)
                    const PrivateKey<DCRTPoly>& sk) {
    auto v = decrypt(cc, ct, sk);
    std::cout << "  [DBG] " << label
              << "  consumed=" << level_of(ct)
              << "  vals=[";
    for (int i = 0; i < std::min((int)v.size(), 4); ++i)
        std::cout << std::setprecision(6) << v[i]
                  << (i < 3 ? ", " : "");
    std::cout << "]\n";
}

/// @brief SiLU in the Cachemir's implementation
Ctx silu(Inference& inf, const Ctx& x_in) {
    const CC& cc = inf.cc();
    auto fn = [](double x) { return x / (std::exp(-x) + 1.0); };
    return eval_chebyshev_f(cc, x_in, fn, -20.0, 20.0, 127);
}

Ctx silu_ffDim(Inference& inf, const Ctx& x_in) {
    const CC& cc = inf.cc();
    const int S  = inf.slots;

    Ctx y = silu(inf, x_in);

    std::vector<double> mask_vec(S, 0.0);
    for (int i = 0; i < inf.size.ffDim; ++i) {
        int idx = i * S / inf.size.ffDim;
        if (idx < S) mask_vec[idx] = 1.0;
    }
    Ptx pt_mask = cc->MakeCKKSPackedPlaintext(
        mask_vec, /*noiseScaleDeg=*/1, (uint32_t)level_of(y));

    return cc->EvalMult(y, pt_mask);
}

/// @brief Encrypted sign approximation via composite minimax polynomials.
/// With L=13, each degree-9 PS eval consumes 4 levels. The DepthGuard
/// bootstraps between evaluations when remaining depth is too low.
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

    // Build DepthGuard for sign() internal bootstraps (needed at L=13).
    // Each degree-9 PS eval consumes ~4 levels; with L=13 we need to
    // bootstrap between the 4 composed evals in sign().
    // Target level = gelu_btp_level (= 13 in paper config) so sign
    // gets a full L=13 remaining after each internal bootstrap.
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

    cc->EvalAddInPlace(term0, term1);
    cc->EvalAddInPlace(term0, term2);

    return term0;
}

/// @brief LayerNorm: (x - mean) * 1/sqrt(variance).
/// Uses realHidDim (unpadded) for scaling; padded slots (e.g. 768→1024)
/// must be zero on entry (guaranteed by weight-padding in linear layers
/// and the paper's fused ciphertext extraction, Section 4.2).
Ctx norm(Inference& inf, const Ctx& x_in,
         int target_level_after_btp, const NormConfig& cfg) {
    const CC& cc = inf.cc();
    const int S  = inf.slots;
    const int hD = inf.size.hidDim;           // padded
    const int rD = inf.size.getRealHidDim();  // unpadded

    // Zero out trailing garbage beyond hD-aligned region.
    // With power-of-2 padded hD this is typically a no-op (hD divides S).
    Ctx x = mask_slots(cc, x_in, S, hD);

    Ctx mean = compute_average(inf, x);
    Ctx varc = compute_variance(inf, x, mean);

    // Add LayerNorm epsilon to variance before computing 1/sqrt(var + eps)
    cc->EvalAddInPlace(varc, cfg.epsilon);

    std::vector<double> coeffs_std(cfg.nr_init_coeffs.rbegin(), cfg.nr_init_coeffs.rend());

    Ctx inv_sqrt_init;
    switch (cfg.nr_init_method) {
        case NRInitMethod::LINEAR:
            inv_sqrt_init = eval_polynomial(cc, varc, coeffs_std);
            break;

        case NRInitMethod::REMEZ: {
            double alpha = 4.0 / (cfg.gs_d_min + cfg.gs_d_max);
            double beta  = alpha * alpha / 4.0;
            auto div_init = cc->EvalMult(varc, -beta);
            cc->EvalAddInPlace(div_init, alpha);

            inv_sqrt_init = eval_rational_approx(cc, div_init,
                                  cfg.remez_p_coeffs, cfg.remez_q_coeffs,
                                  cfg.remez_q_min, cfg.remez_q_max, inf.fhe->pk(), inf.slots,
                                  cfg.remez_div_iters);
            break;
        }

        case NRInitMethod::TAYLOR: {
            // Rescale variance so it lands near taylor_z0 where the
            // degree-3 expansion converges.  taylor_rescale is derived
            // from profiled per-layer variance ranges at config time.
            // 1/sqrt(var) = sqrt(s) * 1/sqrt(s * var)
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

    Ctx inv_sqrt_varc;
    inv_sqrt_varc = inv_sqrt_newton(cc, varc, inv_sqrt_init, cfg.nr_iters, dg);

    Ctx x_centered = x_in->Clone();
    // Divide sum by the REAL (unpadded) dimension to get the true mean.
    // The sum from compute_average includes zero-padded positions, which
    // contribute nothing, so dividing by rD is correct.
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

/// @brief [TODO:check this] Original Cachemir softmax (no max-subtraction).
Ctx softmax_cachemir(Inference& inf, const Ctx& x_in,
                      int target_level_after_btp, int temp) {
    const CC& cc = inf.cc();
    const int S  = inf.slots;

    Ctx x = x_in;

    Ptx pt_inv128 = const_pt(cc, 0.0078125, S, level_of(x));
    cc->EvalMultInPlace(x, pt_inv128);
    cc->EvalAddInPlace(x, 1.0);

    for (int i = 0; i < (7 + temp); ++i)
        cc->EvalSquareInPlace(x);

    Ctx exp_ct = x->Clone();

    const double v = 8.0 / S;
    cc->EvalMultInPlace(x, v);

    for (int j = 1; j < 256; j *= 2) {
        Ctx tmp = cc->EvalRotate(x, 1024 / j);
        cc->EvalAddInPlace(x, tmp);
    }

    x = bootstrap_to(inf, x, (uint32_t)target_level_after_btp);

    Ctx res = cc->EvalNegate(x);
    cc->EvalAddInPlace(res, 1.0);
    Ctx dnm = cc->EvalAdd(res, 1.0);

    for (int i = 0; i < 7; ++i) {
        cc->EvalSquareInPlace(res);

        Ctx tmp = cc->EvalAdd(res, 1.0);
        dnm = cc->EvalMult(dnm, tmp);
    }

    reduce_to_level(cc, exp_ct, level_of(dnm), S);

    Ctx y = cc->EvalMult(exp_ct, dnm);
    cc->EvalMultInPlace(y, v);

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

    // Apply causal mask: add large negative values at future positions
    // so that exp(x + mask) ≈ 0 for masked positions.
    // The mask plaintext should contain 0 at valid positions and a large
    // negative value (e.g. -1e4) at positions to mask out.
    //
    // TODO: implement PCMM (Packed Ciphertext Matrix Multiplication) masking
    // as in HETAL (Lee et al., 2023) for efficient causal attention.
    // PCMM avoids the O(seq^2) plaintext mask by encoding the triangular
    // structure directly into the packing layout and rotation pattern,
    // requiring only O(seq) extra rotations instead of a full mask multiply.
    Ctx x_masked;
    if (causal_mask != nullptr) {
        Ptx mask_copy = causal_mask;  // EvalAdd needs non-const Ptx&
        x_masked = cc->EvalAdd(x_in, mask_copy);
        std::cout << "  [softmax] causal mask applied\n";
    } else {
        x_masked = x_in->Clone();
    }

    // ---- Autonomous bootstrap guard ----
    // Fires before any sub-circuit that would otherwise exhaust the modulus chain.
    // btp_min_remaining = exp_r + 2 ensures exp_approx (exp_r+1 levels) + final
    // EvalMult (1 level) always have budget.  Set to 0 to disable.
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

    // ---- Phase 1: compute or apply max ----
    // For given_max_val we use scalar subtraction (EvalSub with double)
    // to avoid cipher-cipher EvalSub across a large level gap, which can
    // trigger FLEXIBLEAUTO alignment failures in FIDESlib/OpenFHE.
    bool use_scalar_max = false;
    Ctx x_max;
    if (precomputed_max != nullptr) {
        x_max = precomputed_max->Clone();
        maybe_btp(x_max, "post-max");
    } else if (cfg.given_max_val > 0.0) {
        use_scalar_max = true;
        std::cout << "  [softmax] using given_max_val=" << cfg.given_max_val
                  << " (scalar sub)\n";
    } else {
        x_max = x_masked->Clone();
        for (int gap = stride; gap < S; gap *= 2) {
            Ctx shifted = cc->EvalRotate(x_max, gap);

            Ctx diff = cc->EvalSub(x_max, shifted);
            Ctx s    = sign(inf, diff);

            Ctx abs_diff = cc->EvalMult(s, diff);      // |diff| ≈ sign(diff)*diff

            Ctx sum = cc->EvalAdd(x_max, shifted);

            cc->EvalAddInPlace(sum, abs_diff);          // sum + |diff|
            cc->EvalMultInPlace(sum, 0.5);              // (sum + |diff|) / 2

            x_max = sum;
        }
        std::cout << "  [softmax] sign-max done, consumed=" << level_of(x_max) << "\n";
        maybe_btp(x_max, "post-max");
    }

    // ---- Phase 2: exp(x - max) ----
    Ctx x_shifted = x_masked->Clone();
    if (use_scalar_max) {
        cc->EvalSubInPlace(x_shifted, cfg.given_max_val);
    } else {
        cc->EvalSubInPlace(x_shifted, x_max);
    }

    std::cout << "  [softmax] input to exp_approx, consumed=" << level_of(x_shifted) << "\n";
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

    // Build DepthGuard for goldschmidt_inv iterations
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

    // Bootstrap exp_ct if it has exhausted its depth budget.
    // exp_ct was computed before Goldschmidt (which bootstraps inv_sum internally)
    // and may be at a very high consumed level.  The final EvalMult needs ≥ 1
    // remaining level.  Refresh exp_ct if it's too stale.
    maybe_btp(exp_ct, "pre-final-mult");

    Ctx y = cc->EvalMult(exp_ct, inv_sum);
    return y;
}

Ctx argmax(Inference& inf, const Ctx& x_in) {
    const CC& cc = inf.cc();
    const int S  = inf.slots;

    Ctx logit = softmax_cachemir(inf, x_in, 14, 3); // temp > 0 is needed to make the distribution more peaky

    Ctx sum = cc->EvalAdd(logit, 0.0); // TODO: again with copying like so
    for (int j = 1; j < 256; j *= 2) {
        Ctx tmp = cc->EvalRotate(sum, 1024 / j);
        cc->EvalAddInPlace(sum, tmp);
    }

    sum = bootstrap_to(inf, sum, 16);


    // --- Goldschmidt for inverse ---
    // TODO: WHYYYYY?! Why not just call the primitive?
    Ctx res = cc->EvalNegate(sum);
    cc->EvalAddInPlace(res, 1.0);
    Ctx dnm = cc->EvalAdd(res, 1.0);

    for (int i = 0; i < 12; ++i) {
        cc->EvalSquareInPlace(res);

        Ctx tmp = cc->EvalAdd(res, 1.0);
        dnm = cc->EvalMult(dnm, tmp);
    }
    // --- End Goldschmidt for inverse ---

    Ctx logit_copy = cc->EvalAdd(logit, 0.0); // TODO: i dunno what to say anymore.
    reduce_to_level(cc, logit_copy, level_of(dnm), S);
    Ctx y = cc->EvalMult(logit_copy, dnm); // TODO: Soo we doing probabilities * (1/sum of probabilities) = probabilities, right?

    // Actual argmax trick, if we have high enough temp
    // then the softmax distribution with almost be one-hot encoding,
    // and thus we can do this stuff to get the index of the max element.
    std::vector<double> idx_vec(S, 0.0);
    for (int i = 0; i < S; ++i)
        idx_vec[i] = (double)i;
    Ptx idx = cc->MakeCKKSPackedPlaintext(idx_vec, 1, level_of(y));
    cc->EvalMultInPlace(y, idx);

    for (int i = 1; i < 256; i *= 2) {
        Ctx tmp = cc->EvalRotate(y, 1024 / i);
        cc->EvalAddInPlace(y, tmp);
    }

    return y;
}
