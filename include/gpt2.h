#pragma once

#include "inference.h"
#include "nonlinear.h"

Ctx linear_interleaved(Inference& inf, const Ctx& x,
                       const std::string& wname, int d_in, int d_out);

Ctx qkv_q(Inference& inf, const Ctx& x);
Ctx qkv_k(Inference& inf, const Ctx& x);
Ctx qkv_v(Inference& inf, const Ctx& x);

std::tuple<Ctx, Ctx> rope(Inference& inf, const Ctx& q, const Ctx& k);

void cache_kv(Inference& inf, const Ctx& k, const Ctx& v);

Ctx qk_transpose(Inference& inf, const Ctx& q);
Ctx attn_v       (Inference& inf, const Ctx& s);
Ctx out_proj     (Inference& inf, const Ctx& x);

std::pair<Ctx, Ctx> up_gate  (Inference& inf, const Ctx& x);
Ctx                 down_proj(Inference& inf, const Ctx& x);

void rotate_add_inplace(Inference& inf, Ctx& x, int step);

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

inline GPT2ModelConfig make_gpt2_profiled_config() {
    static const double rescales[12] = {
        0.19271061, 0.01123577, 0.00085924, 0.00005702,
        0.00004874, 0.00004363, 0.00004080, 0.00003925,
        0.00003833, 0.00003779, 0.00003752, 0.00003747,
    };

    GPT2ModelConfig cfg;
    cfg.num_layers = 12;
    cfg.layers.resize(12);
    for (int i = 0; i < 12; ++i) {
        cfg.layers[i].norm1_cfg = make_norm_encllm_gpt2(rescales[i]);
        cfg.layers[i].norm2_cfg = make_norm_encllm_gpt2(rescales[i]);
    }
    cfg.final_norm_cfg = make_norm_encllm_gpt2(rescales[11]);
    return cfg;
}

inline const GPT2ModelConfig GPT2_PROFILED_CONFIG = make_gpt2_profiled_config();

std::vector<int32_t> compute_gpt2_rot_indices(
    int S, int hidDim, int ffDim, int numHeads, int seqLen);

Inference make_gpt2(int logN, int hidDim, int ffDim,
                    int seqLen, int numHeads, bool parallel,
                    bool bench = true);

void gpt2_prepare_weights(Inference& inf, const std::vector<std::string>& names);
void gpt2_prepare_cache  (Inference& inf, const std::vector<std::string>& names);

void gpt2_load_weights(Inference& inf, const std::string& weight_dir, int num_layers = 12);

Ctx gpt2_decoder(Inference& inf, const Ctx& x, const GPT2LayerConfig& cfg = {},
                 int layer_idx = -1);
Ctx gpt2_model  (Inference& inf, const Ctx& x, const GPT2ModelConfig& cfg = GPT2_DEFAULT_CONFIG);

