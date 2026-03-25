#pragma once

#include "inference.h"
#include <vector>

// ---------------------------------------------------------------------------
// GeluConfig  (paper × model)
// ---------------------------------------------------------------------------
// rescale_factor scales (x - threshold) into [-1, 1] before the sign approx.
// A value of 1/R works when activations live roughly in [-R, R].

struct GeluConfig {
    double rescale_factor = 1.0;
};

//                                              rescale
inline constexpr GeluConfig GELU_ENCLLM_GPT2  { 1.0 / 4.0 };
inline constexpr GeluConfig GELU_CACHEMIR_GPT2 { 1.0 / 4.0 }; // calibrate
inline constexpr GeluConfig GELU_HEAWARE_GPT2  { 1.0 / 4.0 }; // calibrate

// ---------------------------------------------------------------------------
// NormConfig  (paper × model)
// ---------------------------------------------------------------------------
// The Newton-Raphson inv-sqrt needs an initial guess for 1/sqrt(varc).
// Two methods for deriving the polynomial coefficients:
//
//   LINEAR   — degree-1 least-squares fit over [v_lo, v_hi]:
//                import numpy as np
//                v = np.array([v_lo, v_hi])
//                b, c = np.polyfit(v, 1/np.sqrt(v), 1)
//                coeffs = [b, c]
//
//   REMEZ    — minimax optimal polynomial of any degree over [v_lo, v_hi],
//              computed offline (e.g. with the `remez` package or chebfun).
//              Coefficients stored highest-degree first: [a_n, ..., a_1, a_0].
//              Costs more levels but converges faster per NR iteration.
//
// Both methods evaluate identically at runtime via eval_polynomial(cc, varc, coeffs).

enum class NRInitMethod { LINEAR, REMEZ, TAYLOR };

struct NormConfig {
    NRInitMethod        nr_init_method = NRInitMethod::LINEAR;
    std::vector<double> nr_init_coeffs = { -42.1, 7.37 }; // highest-degree first
    int                 nr_iters       = 4;
    int                 gs_iters       = 2;
    // Remez rational approximation (used when nr_init_method == REMEZ)
    // nr_init_coeffs stores numerator p_coeffs (highest-degree first)
    std::vector<double> remez_q_coeffs = {};  // denominator coeffs, lowest-degree first, q[0]=1
    double              remez_q_min    = 1.0;
    double              remez_q_max    = 1.0;
    int                 remez_div_iters = 5;  // Goldschmidt iters for 1/q(x) in rational eval
    // Taylor expansion (used when nr_init_method == TAYLOR)
    double              taylor_z0      = 0.5; // expansion point (midpoint of variance range)
};

//                                                              method               coeffs          NR  GS
inline const NormConfig NORM_ENCLLM_GPT2   { NRInitMethod::LINEAR, { -42.1,  7.37 },  4,  2 };
inline const NormConfig NORM_ENCLLM_LLAMA  { NRInitMethod::LINEAR, { -42.1,  7.37 },  4,  2 }; // calibrate
inline const NormConfig NORM_CACHEMIR_GPT2 { NRInitMethod::LINEAR, { -42.1,  7.37 },  4,  2 }; // calibrate
inline const NormConfig NORM_CACHEMIR_LLAMA{ NRInitMethod::LINEAR, { -42.1,  7.37 },  4,  2 }; // calibrate
inline const NormConfig NORM_HEAWARE_GPT2  { NRInitMethod::LINEAR, { -42.1,  7.37 },  3,  2 }; // calibrate
inline const NormConfig NORM_HEAWARE_LLAMA { NRInitMethod::LINEAR, { -42.1,  7.37 },  3,  2 }; // calibrate

// ---------------------------------------------------------------------------
// Nonlinear op declarations (implemented in nonlinear.cu, shared across models)
// ---------------------------------------------------------------------------

Ctx silu      (Inference& inf, const Ctx& x);
Ctx silu_ffDim(Inference& inf, const Ctx& x);

Ctx sign      (Inference& inf, const Ctx& x);
Ctx lt_function(Inference& inf, const Ctx& x, double value, double rescale_factor = 1.0);

Ctx gelu   (Inference& inf, const Ctx& x, const GeluConfig& cfg = GELU_ENCLLM_GPT2);
Ctx norm   (Inference& inf, const Ctx& x, int target_level_after_btp, const NormConfig& cfg = NORM_ENCLLM_GPT2);

/// exp(x) ≈ (1 + x / 2^r)^{2^r}.  Costs 1 mult + r squarings = r+1 levels.
Ctx exp_approx(const CC& cc, const Ctx& x, int r);

/// [FROZEN] Original Cachemir softmax (no max-subtraction).
Ctx softmax_cachemir(Inference& inf, const Ctx& x, int target_level_after_btp, int temp);

/// Numerically-stable softmax: exp(x - max) / sum(exp(x - max)).
/// @param r          Squarings for exp approx (accuracy vs depth trade-off).
/// @param gs_iters   Goldschmidt iterations for 1/sum division.
/// @param seq_dim    Sequence positions to reduce over (power of 2, divides S).
/// @param precomputed_max  If non-null, used as max(x) instead of computing it.
Ctx softmax(Inference& inf, const Ctx& x,
            int target_level_after_btp,
            int r, int gs_iters, int seq_dim,
            const Ctx& precomputed_max = nullptr);

Ctx argmax (Inference& inf, const Ctx& x);
