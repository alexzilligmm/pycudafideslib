#pragma once

#include "inference.h"
#include "nonlinear.h"

struct CacheMirParams {
    bool is_up;
    int d, alpha, t, tp, tp_in, tp_out, r_i, r_o, n_pt;
};

CacheMirParams compute_cm_params(int N, int d_in, int d_out);
std::vector<std::vector<double>> load_matrix_txt(const std::string& path, int d_in, int d_out);
Ctx encode_linear_input(Inference& inf, const std::vector<double>& x, int d_in, int d_out);
std::vector<Ptx> encode_weight_matrix(Inference& inf, const std::vector<std::vector<double>>& W, int d_in, int d_out);
std::vector<Ptx> load_weight_txt(Inference& inf, const std::string& path, int d_in, int d_out);

Ctx linear(Inference& inf, const Ctx& x,
           const std::string& wname, int d_in, int d_out);

Ctx qkv_q(Inference& inf, const Ctx& x);
Ctx qkv_k(Inference& inf, const Ctx& x);
Ctx qkv_v(Inference& inf, const Ctx& x);

std::tuple<Ctx, Ctx> rope(Inference& inf, const Ctx& q, const Ctx& k);

Ctx out_proj     (Inference& inf, const Ctx& x);

std::pair<Ctx, Ctx> up_gate  (Inference& inf, const Ctx& x);
Ctx                 down_proj(Inference& inf, const Ctx& x);

std::vector<std::vector<double>> rearrange_up_weights(
    const std::vector<std::vector<double>>& W, int d_in);

std::vector<std::vector<double>> rearrange_down_weights(
    const std::vector<std::vector<double>>& W, int d_out);

void rotate_add_inplace(Inference& inf, Ctx& x, int step);

// ── MLP block: norm → up → gelu → down → residual ──────────────────────
struct MLPConfig {
    NormConfig  norm_cfg          = NORM_ENCLLM_GPT2;
    int         norm_target_level = 14;

    GeluConfig  gelu_cfg          = GELU_ENCLLM_GPT2;

    int  pre_gelu_btp_level  = 0;   // 0 = no bootstrap
    int  pre_down_btp_level  = 0;
};

inline const MLPConfig MLP_ENCLLM_GPT2 {};

Ctx mlp(Inference& inf, const Ctx& x, const MLPConfig& cfg = MLP_ENCLLM_GPT2);

struct GPT2LayerConfig {
    int          norm1_btp_level    = 7;    
    int          norm1_target_level = 7;   
    NormConfig   norm1_cfg          = NORM_ENCLLM_GPT2;

    int          cache_btp_level    = 0; 

    int          attn_btp_level     = 9; 
    SoftmaxConfig softmax_cfg       = SOFTMAX_ENCLLM_GPT2;
    Ptx          attn_causal_mask   = nullptr;

    int          attn_v_btp_level   = 0;  

    int          norm2_btp_level    = 7;    
    int          norm2_target_level = 7;    
    NormConfig   norm2_cfg          = NORM_ENCLLM_GPT2;

    int          gelu_btp_level     = 13;  
    GeluConfig   gelu_cfg           = GELU_ENCLLM_GPT2;

    int          down_btp_level     = 0;   
};

struct GPT2ModelConfig {
    int num_layers = 12;
    std::vector<GPT2LayerConfig> layers = {}; 

    int          final_norm_btp_level    = 15;
    int          final_norm_target_level = 11;
    NormConfig   final_norm_cfg          = NORM_ENCLLM_GPT2;
};

inline const GPT2ModelConfig GPT2_DEFAULT_CONFIG {};

inline NormConfig make_norm_encllm_gpt2(double taylor_rescale) {
    NormConfig c = NORM_ENCLLM_GPT2;
    c.taylor_rescale = taylor_rescale;
    return c;
}

int interleave_idx(int m, int d, int dim);

std::vector<int32_t> compute_gpt2_rot_indices(
    int S, int hidDim, int ffDim, int numHeads, int seqLen);

Inference make_gpt2(int logN, int hidDim, int ffDim,
                    int seqLen, int numHeads, bool parallel,
                    bool bench = true);

void gpt2_prepare_weights(Inference& inf, const std::vector<std::string>& names);




