#pragma once

#include "inference.h"
#include "ckks_primitives.h"
#include <vector>

struct GeluConfig {
    double rescale_factor = 1.0;
    bool   bootstrap_indicators = true;   // bootstrap lt results before combining
    int    btp_target_level     = 0;      // 0 = use raw EvalBootstrap (default)
    // DepthGuard params for sign() internal bootstraps (needed at L=13)
    uint32_t sign_btp_min_remaining = 5;  // bootstrap inside sign when remaining < 5
};

//                                                 rescale  btp_ind  btp_lvl
inline const GeluConfig GELU_ENCLLM_GPT2  { 1.0 / 67.0,  true,    0 }; // 1/(absmax+4), absmax~63 from profiling

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
    double              taylor_rescale = 1.0; // scale variance before Taylor init (derived from profiled ranges)
    double              gs_d_min       = 0.5;
    double              gs_d_max       = 2.0;

    double              epsilon        = 1e-5;

    std::vector<int>    nr_btp_schedule    = {};
    uint32_t            nr_btp_min_remaining = 6;
};

//                                              method               coeffs          NR  GS                                              z0    rescale   d_min   d_max
inline const NormConfig NORM_ENCLLM_GPT2   { NRInitMethod::TAYLOR, {},               16,  14, {}, {}, 1.0, 1.0, 5, /*taylor_z0=*/0.5, /*taylor_rescale=*/1.0, /*gs_d_min=*/1e-6, /*gs_d_max=*/1.0 };


Ctx sign      (Inference& inf, const Ctx& x, const DepthGuard& dg = {});
Ctx lt_function(Inference& inf, const Ctx& x, double value, double rescale_factor = 1.0,
                const DepthGuard& dg = {});

Ctx gelu   (Inference& inf, const Ctx& x, const GeluConfig& cfg = GELU_ENCLLM_GPT2);
Ctx norm   (Inference& inf, const Ctx& x, int target_level_after_btp, const NormConfig& cfg = NORM_ENCLLM_GPT2);

Ctx exp_approx(const CC& cc, const Ctx& x, int r);

struct SoftmaxConfig {
    int  exp_r          = 7;
    int  gs_inv_iters   = 10;
    int  seq_dim        = 4;

    std::vector<int> gs_btp_schedule    = {};
    uint32_t         gs_btp_min_remaining = 6;

    uint32_t         btp_min_remaining   = 0;
    int              btp_target_level    = 14;

    double           gs_inv_init         = 1.0; // Goldschmidt division initial value
    double           given_max_val       = 0.0; // 0.0 = not set (use computed max)
};

//                                                exp  gs  seq  gs_sched  gs_min  btp_min  btp_lvl  gs_init  given_max
inline const SoftmaxConfig SOFTMAX_ENCLLM_GPT2   { 7,  14,  4,  {},       6,      9,       9,       0.628,   205.0 }; 
inline const SoftmaxConfig SOFTMAX_ATTN_GPT2     { 7,  14,  4,  {},       7,      16,       16,      0.628,   205.0 };

Ctx softmax(Inference& inf, const Ctx& x,
            const SoftmaxConfig& cfg = SOFTMAX_ENCLLM_GPT2,
            const Ctx& precomputed_max = nullptr,
            const Ptx& causal_mask = nullptr);

