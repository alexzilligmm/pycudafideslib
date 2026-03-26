#pragma once

#include "inference.h"
#include <vector>

struct GeluConfig {
    double rescale_factor = 1.0;
};

//                                              rescale
inline constexpr GeluConfig GELU_ENCLLM_GPT2  { 1.0 / 10.0 };
inline constexpr GeluConfig GELU_CACHEMIR_GPT2 { 1.0 / 10.0 }; // calibrate
inline constexpr GeluConfig GELU_HEAWARE_GPT2  { 1.0 / 10.0 }; // calibrate

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
    std::vector<double> nr_init_coeffs = { -42.1, 7.37 };
    int                 nr_iters       = 4;
    int                 gs_iters       = 2;
    std::vector<double> remez_p_coeffs = {};
    std::vector<double> remez_q_coeffs = {};
    double              remez_q_min    = 1.0;
    double              remez_q_max    = 1.0;
    int                 remez_div_iters = 5;
    double              taylor_z0      = 0.5;
    double              gs_d_min       = 0.5;
    double              gs_d_max       = 2.0;

    // LayerNorm epsilon: 1/sqrt(variance + epsilon)
    double              epsilon        = 1e-5;

    // Bootstrap placement for inv_sqrt_newton inside norm().
    // Scheduled mode: sorted iteration indices from the planner.
    // Dynamic mode: min levels remaining before auto-bootstrap (0 = off).
    std::vector<int>    nr_btp_schedule    = {};
    uint32_t            nr_btp_min_remaining = 6;
};

//                                              method               coeffs          NR  GS
inline const NormConfig NORM_ENCLLM_GPT2   { NRInitMethod::TAYLOR, {},               19,  10, {}, {}, 1.0, 1.0, 5, /*taylor_z0=*/0.5, /*gs_d_min=*/1e-6, /*gs_d_max=*/1.0 };
inline const NormConfig NORM_ENCLLM_LLAMA  { NRInitMethod::TAYLOR, {},               4,  2,  {}, {}, 1.0, 1.0, 5, /*taylor_z0=*/0.5, /*gs_d_min=*/1e-6, /*gs_d_max=*/1.0 };
inline const NormConfig NORM_CACHEMIR_GPT2 { NRInitMethod::LINEAR, { -42.1,  7.37 },  4,  2 }; // TODO: calibrate
inline const NormConfig NORM_CACHEMIR_LLAMA{ NRInitMethod::LINEAR, { -42.1,  7.37 },  4,  2 }; // TODO: calibrate
inline const NormConfig NORM_HEAWARE_GPT2  { NRInitMethod::REMEZ,  { -42.1,  7.37 },  3,  2 }; // calibrate
inline const NormConfig NORM_HEAWARE_LLAMA { NRInitMethod::REMEZ,  { -42.1,  7.37 },  3,  2 }; // calibrate

Ctx silu      (Inference& inf, const Ctx& x);
Ctx silu_ffDim(Inference& inf, const Ctx& x);

Ctx sign      (Inference& inf, const Ctx& x);
Ctx lt_function(Inference& inf, const Ctx& x, double value, double rescale_factor = 1.0);

Ctx gelu   (Inference& inf, const Ctx& x, const GeluConfig& cfg = GELU_ENCLLM_GPT2);
Ctx norm   (Inference& inf, const Ctx& x, int target_level_after_btp, const NormConfig& cfg = NORM_ENCLLM_GPT2);

Ctx exp_approx(const CC& cc, const Ctx& x, int r);

Ctx softmax_cachemir(Inference& inf, const Ctx& x, int target_level_after_btp, int temp);

// ---------------------------------------------------------------------------
// SoftmaxConfig  (paper × model)
// ---------------------------------------------------------------------------
// Numerically-stable softmax: exp(x - max) / sum(exp(x - max)).
//
//   exp_r          — repeated-squaring depth for exp_approx: exp(x) ≈ (1 + x/2^r)^{2^r}
//   gs_inv_iters   — Goldschmidt iterations for 1/sum(exp)
//   seq_dim        — sequence length (number of logits being softmaxed)
//
// DepthGuard fields (same semantics as NormConfig):
//   gs_btp_schedule      — sorted Goldschmidt iteration indices for scheduled bootstrap
//   gs_btp_min_remaining — dynamic safety-net for goldschmidt_inv (0 = off)
//   btp_min_remaining    — dynamic safety-net at phase boundaries (after max,
//                          before exp, before inv). Bootstraps whenever remaining
//                          levels drop below this threshold. 0 = off.
//   btp_target_level     — target level after any bootstrap
struct SoftmaxConfig {
    int  exp_r          = 7;
    int  gs_inv_iters   = 10;
    int  seq_dim        = 4;

    // DepthGuard for goldschmidt_inv iterations
    std::vector<int> gs_btp_schedule    = {};
    uint32_t         gs_btp_min_remaining = 6;

    // Phase-boundary dynamic bootstrap: fires when remaining < threshold
    uint32_t         btp_min_remaining   = 0;   // 0 = no phase-boundary bootstraps
    int              btp_target_level    = 14;
};

//                                              r   gs  seq  gs_sched  gs_min  btp_min  btp_lvl
inline const SoftmaxConfig SOFTMAX_ENCLLM_GPT2   { 7,  10,  4,  {},       6,      0,       14 };
inline const SoftmaxConfig SOFTMAX_ENCLLM_LLAMA  { 7,  10,  4,  {},       6,      0,       14 };
inline const SoftmaxConfig SOFTMAX_CACHEMIR_GPT2 { 5,   5,  4,  {},       6,      0,       14 }; // TODO: calibrate
inline const SoftmaxConfig SOFTMAX_NOMAX_GPT2    { 5,   5,  2,  {},       6,      20,      14 }; // non-oracle: needs phase bootstraps

Ctx softmax(Inference& inf, const Ctx& x,
            const SoftmaxConfig& cfg = SOFTMAX_ENCLLM_GPT2,
            const Ctx& precomputed_max = nullptr);

Ctx argmax (Inference& inf, const Ctx& x);
