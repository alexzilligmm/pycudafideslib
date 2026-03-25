#include "llama.h"
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
Ctx sign(Inference& inf, const Ctx& x_in) {
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
    Ctx h = eval_polynomial_ps(cc, x_in, G4, inf.fhe->pk(), (size_t)S);
    h     = eval_polynomial_ps(cc, h,    G4, inf.fhe->pk(), (size_t)S);
    h     = eval_polynomial_ps(cc, h,    F4, inf.fhe->pk(), (size_t)S);
    h     = eval_polynomial_ps(cc, h,    F4, inf.fhe->pk(), (size_t)S);

    return h;
}

/// @brief Homomorphic less-than: returns ~1 if x < value, ~0 otherwise.
/// @param rescale_factor  Scales (x - value) into [-1, 1] before sign. Use 1.0 when the input is already in that range.
Ctx lt_function(Inference& inf, const Ctx& x_in, double value,
                double rescale_factor) {
    const CC& cc = inf.cc();
    auto x = x_in->Clone();        
    Ctx y = cc->EvalAdd(x, -value);
    if (rescale_factor != 1.0) {
        cc->EvalMultInPlace(y, rescale_factor);
        cc->RescaleInPlace(y);
    }

    Ctx s = sign(inf, y);

    // (1 - s) / 2
    Ctx res = cc->EvalNegate(s);
    cc->RescaleInPlace(res);
    cc->EvalAddInPlace(res, 1.0);
    cc->EvalMultInPlace(res, 0.5);
    cc->RescaleInPlace(res);

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
    std::cout << "Computing GeLU with rescale_factor = " << rescale_factor << "\n";

    Ctx lt_m4, lt_m195, lt_p3;

    lt_m4   = lt_function(inf, x_in, -4.0,  rescale_factor);
    std::cout << "Level after lt_m4: " << level_of(lt_m4) << "\n";
    lt_m195 = lt_function(inf, x_in, -1.95, rescale_factor);
    std::cout << "Level after lt_m195: " << level_of(lt_m195) << "\n";
    lt_p3   = lt_function(inf, x_in,  3.0,  rescale_factor);
    std::cout << "Level after lt_p3: " << level_of(lt_p3) << "\n";
    std::cout << "Level of x_in: " << level_of(x_in) << "\n";
    

    std::cout << "Interval computed" << "\n";

    Ctx ind_f0  = cc->EvalSub(lt_m195, lt_m4);
    Ctx ind_f1  = cc->EvalSub(lt_p3,   lt_m195);
    Ctx ind_lin = cc->EvalNegate(lt_p3);
    cc->RescaleInPlace(ind_lin);
    cc->EvalAddInPlace(ind_lin, 1.0);

    Ctx p0 = eval_polynomial_ps(cc, x_in, poly_f0, inf.fhe->pk(), (size_t)S);
    Ctx p1 = eval_polynomial_ps(cc, x_in, poly_f1, inf.fhe->pk(), (size_t)S);

    match_level(cc, ind_f0, p0);
    Ctx term0 = cc->EvalMult(ind_f0, p0);
    cc->RescaleInPlace(term0);

    match_level(cc, ind_f1, p1);
    Ctx term1 = cc->EvalMult(ind_f1, p1);
    cc->RescaleInPlace(term1);

    Ctx x_copy = x_in;
    match_level(cc, ind_lin, x_copy);
    Ctx term2 = cc->EvalMult(ind_lin, x_copy);
    cc->RescaleInPlace(term2);

    match_level(cc, term0, term1);
    cc->EvalAddInPlace(term0, term1);
    match_level(cc, term0, term2);
    cc->EvalAddInPlace(term0, term2);

    return term0;
}

/// @brief LayerNorm: (x - mean) * 1/sqrt(variance).
/// Initial 1/sqrt guess is computed via LINEAR, REMEZ, or TAYLOR (set in cfg),
/// then refined with Goldschmidt iterations.
Ctx norm(Inference& inf, const Ctx& x_in,
         int target_level_after_btp, const NormConfig& cfg) {
    const CC& cc = inf.cc();
    const int S  = inf.slots;
    const int hD = inf.size.hidDim;

    Ctx mean = compute_average(inf, x_in);
    Ctx varc = compute_variance(inf, x_in, mean);

    std::vector<double> coeffs_std(cfg.nr_init_coeffs.rbegin(), cfg.nr_init_coeffs.rend());

    Ctx inv_sqrt_init;
    switch (cfg.nr_init_method) {
        case NRInitMethod::LINEAR:
            inv_sqrt_init = eval_polynomial(cc, varc, coeffs_std);
            break;

        case NRInitMethod::REMEZ:
            inv_sqrt_init = eval_rational_approx(
                cc, varc, coeffs_std, cfg.remez_q_coeffs,
                cfg.remez_q_min, cfg.remez_q_max,
                inf.fhe->pk(), (size_t)S, cfg.remez_div_iters);
            break;

        case NRInitMethod::TAYLOR: {
            std::vector<double> taylor_c = taylor_inv_sqrt_coeffs(cfg.taylor_z0);
            inv_sqrt_init = eval_taylor_inv_sqrt(cc, varc, taylor_c, cfg.taylor_z0);
            break;
        }
    }

    varc          = bootstrap_to(inf, varc, (uint32_t)target_level_after_btp);
    inv_sqrt_init = bootstrap_to(inf, inv_sqrt_init, (uint32_t)target_level_after_btp);

    Ctx inv_sqrt_varc = goldschmidt_inv_sqrt(cc, varc, inv_sqrt_init, cfg.gs_iters);

    Ctx x_centered = x_in->Clone();
    Ctx true_mean  = cc->EvalMult(mean, 1.0 / (double)hD);
    cc->RescaleInPlace(true_mean);

    match_level(cc, x_centered, true_mean);
    cc->EvalSubInPlace(x_centered, true_mean);

    match_level(cc, x_centered, inv_sqrt_varc);
    Ctx result = cc->EvalMult(x_centered, inv_sqrt_varc);
    cc->RescaleInPlace(result);

    return result;
}

/// @brief Softmax in the Cachemir's implementation
/// @param temp Additional squarings such that the computed function is actually
///             exp(x * 2^temp) instead of exp(x). To add, as the name suggests, a temperature effect.
Ctx softmax(Inference& inf, const Ctx& x_in,
             int target_level_after_btp, int temp) {
    const CC& cc = inf.cc();
    const int S  = inf.slots;
    const auto& sk = inf.fhe->sk();

    Ctx x = x_in;

    // These guys are approximating exp(x) via the
    // canonical limit definition: exp(x) = lim_{n->inf} (1 + x/n)^n.
    // They choose n=128=2^7, so they need 7 squarings to get the final result.
    Ptx pt_inv128 = const_pt(cc, 0.0078125, S, level_of(x));
    cc->EvalMultInPlace(x, pt_inv128);
    cc->RescaleInPlace(x);

    Ptx pt_one = const_pt(cc, 1.0, S, level_of(x));
    cc->EvalAddInPlace(x, pt_one);

    for (int i = 0; i < (7 + temp); ++i) {
        cc->EvalSquareInPlace(x);
        cc->RescaleInPlace(x);
    }

    Ctx exp_ct = cc->EvalAdd(x, 0.0);
    // TODO: why do they make a copy like so?
    // can't they just
    //   Ctx exp_ct = x; ???

    const double v = 8.0 / S;   // = 1/256 for S=2048
    cc->EvalMultInPlace(x, v);
    cc->RescaleInPlace(x);      // NL -> 1

    // Trick to compute total sum in log-time
    for (int j = 1; j < 256; j *= 2) {
        Ctx tmp = cc->EvalRotate(x, 1024 / j);
        cc->EvalAddInPlace(x, tmp);
    }

    x = bootstrap_to(inf, x, (uint32_t)target_level_after_btp);

    // TODO, understand: x right now is the total sum, right?
    // if so then dnm should be 2 - x right now no?
    Ctx res = cc->EvalNegate(x);         // NL=2 (or auto-rescale from bootstrap NL=2)
    cc->RescaleInPlace(res);             // NL->1
    cc->EvalAddInPlace(res, 1.0);        // res = 1 - v*sum ~ 0
    Ctx dnm = cc->EvalAdd(res, 1.0);     // dnm ~ 1  (will converge to 1/(v*sum))

    // TODO This just goldschmidt for 7 iterations, why not calling the primitive?
    for (int i = 0; i < 7; ++i) {
        cc->EvalSquareInPlace(res);
        cc->RescaleInPlace(res);

        Ctx tmp = cc->EvalAdd(res, 1.0);

        match_level(cc, dnm, tmp);
        dnm = cc->EvalMult(dnm, tmp);
        cc->RescaleInPlace(dnm);

    }

    { // Rescaling exp_ct to match dnm's level, so we can multiply them together.
      // TODO: Why the heck do we need to do this iteratively like this? Don't we have
      // a 'dropToLevel()' or something?
        uint32_t target = level_of(dnm);
        int iters = 0;
        while (level_of(exp_ct) < target) {
            uint32_t cur = level_of(exp_ct);
            std::vector<double> ones(S, 1.0);
            Ptx pt_one2 = cc->MakeCKKSPackedPlaintext(ones, 1, (uint32_t)cur);
            cc->EvalMultInPlace(exp_ct, pt_one2);
            cc->RescaleInPlace(exp_ct);
            ++iters;
        }
    }


    Ctx y = cc->EvalMult(exp_ct, dnm);
    cc->RescaleInPlace(y);

    cc->EvalMultInPlace(y, v);   // algebraically: exp * (1/(v*sum)) * v = exp/sum
    cc->RescaleInPlace(y);

    return y;
}

Ctx argmax(Inference& inf, const Ctx& x_in) {
    const CC& cc = inf.cc();
    const int S  = inf.slots;

    Ctx logit = softmax(inf, x_in, 14, 3); // temp > 0 is needed to make the distribution more peaky

    Ctx sum = cc->EvalAdd(logit, 0.0); // TODO: again with copying like so
    for (int j = 1; j < 256; j *= 2) {
        Ctx tmp = cc->EvalRotate(sum, 1024 / j);
        cc->EvalAddInPlace(sum, tmp);
    }

    sum = bootstrap_to(inf, sum, 16);


    // --- Goldschmidt for inverse ---
    // TODO: WHYYYYY?! Why not just call the primitive?
    Ctx res = cc->EvalNegate(sum);
    cc->RescaleInPlace(res);
    cc->EvalAddInPlace(res, 1.0);
    Ctx dnm = cc->EvalAdd(res, 1.0);

    for (int i = 0; i < 12; ++i) {
        cc->EvalSquareInPlace(res);
        cc->RescaleInPlace(res);

        Ctx tmp = cc->EvalAdd(res, 1.0);
        match_level(cc, dnm, tmp);
        dnm = cc->EvalMult(dnm, tmp);
        cc->RescaleInPlace(dnm);
    }
    // --- End Goldschmidt for inverse ---

    Ctx logit_copy = cc->EvalAdd(logit, 0.0); // TODO: i dunno what to say anymore.
    reduce_to_level(cc, logit_copy, level_of(dnm), S);
    Ctx y = cc->EvalMult(logit_copy, dnm); // TODO: Soo we doing probabilities * (1/sum of probabilities) = probabilities, right?
    cc->RescaleInPlace(y);

    // Actual argmax trick, if we have high enough temp
    // then the softmax distribution with almost be one-hot encoding,
    // and thus we can do this stuff to get the index of the max element.
    std::vector<double> idx_vec(S, 0.0);
    for (int i = 0; i < S; ++i)
        idx_vec[i] = (double)i;
    Ptx idx = cc->MakeCKKSPackedPlaintext(idx_vec, 1, level_of(y));
    cc->EvalMultInPlace(y, idx);
    cc->RescaleInPlace(y);

    for (int i = 1; i < 256; i *= 2) {
        Ctx tmp = cc->EvalRotate(y, 1024 / i);
        cc->EvalAddInPlace(y, tmp);
    }

    return y;
}
