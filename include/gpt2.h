#pragma once

#include "inference.h"
#include "nonlinear.h"

// ── Shared linear / attention ops (CacheMIR interleaved packing) ──────
//
// These live in linear.cu and are used by both GPT-2 and any future model.
// CacheMIR interleaved linear (paper §3, Algorithm 1):
//   d_in*d_out/S weight plaintexts.  Junk at non-data output slots is
//   fused/cleaned by the next element-wise multiplication (paper §3.2).
//   For square: pass d_in = d_out = d (or omit d_out).
Ctx linear_interleaved(Inference& inf, const Ctx& x,
                       const std::string& wname, int d_in, int d_out = 0);

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

// ── Per-layer configuration ────────────────────────────────────────────
// Every tunable knob lives here so that an external optimizer can set
// different params for each of the 12 GPT-2 layers.

struct GPT2LayerConfig {
    // Paper Table 3 level assignments (L=13):
    //   Norm: 0→7, Softmax: 0→9, GELU: 0→13

    // Pre-attention LayerNorm
    int          norm1_btp_level    = 7;    // paper: Add&Norm 0→7
    int          norm1_target_level = 7;    // DepthGuard refresh inside NR
    NormConfig   norm1_cfg          = NORM_ENCLLM_GPT2;

    // KV cache bootstrap (after QKV, before cache_kv)
    int          cache_btp_level    = 0;    // no separate cache BTP

    // Attention softmax
    int          attn_btp_level     = 9;    // paper: Softmax 0→9
    SoftmaxConfig softmax_cfg       = SOFTMAX_ENCLLM_GPT2;
    Ptx          attn_causal_mask   = nullptr;

    // Attention value + output projection
    int          attn_v_btp_level   = 0;    // no BTP before AttnV

    // Pre-MLP LayerNorm
    int          norm2_btp_level    = 7;    // paper: Add&Norm 0→7
    int          norm2_target_level = 7;    // DepthGuard refresh inside NR
    NormConfig   norm2_cfg          = NORM_ENCLLM_GPT2;

    // GELU activation
    int          gelu_btp_level     = 13;   // paper: GELU 0→13 (max level)
    GeluConfig   gelu_cfg           = GELU_ENCLLM_GPT2;

    // Down projection
    int          down_btp_level     = 0;    // no BTP before Down
};

// ── Model-level configuration ──────────────────────────────────────────

struct GPT2ModelConfig {
    int num_layers = 12;
    std::vector<GPT2LayerConfig> layers = {};  // empty → default for every layer

    // Final LayerNorm after all decoder layers
    int          final_norm_btp_level    = 15;
    int          final_norm_target_level = 11;
    NormConfig   final_norm_cfg          = NORM_ENCLLM_GPT2;
};

inline const GPT2ModelConfig GPT2_DEFAULT_CONFIG {};

// ── Profiled per-layer config (from estimate_ranges.py on GPT-2 small) ──
// taylor_rescale = 0.5 / var_max_observed so scaled variance ≈ taylor_z0
inline NormConfig make_norm_encllm_gpt2(double taylor_rescale) {
    NormConfig c = NORM_ENCLLM_GPT2;
    c.taylor_rescale = taylor_rescale;
    return c;
}

inline GPT2ModelConfig make_gpt2_profiled_config() {
    // Per-layer taylor_rescale derived from profiled variance ranges
    // L00 var_max=2.59, L01=44.5, ..., L11=13344
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
    // Final norm: use layer-11 rescale
    cfg.final_norm_cfg = make_norm_encllm_gpt2(rescales[11]);
    return cfg;
}

inline const GPT2ModelConfig GPT2_PROFILED_CONFIG = make_gpt2_profiled_config();

// ── Factory / helpers ──────────────────────────────────────────────────

std::vector<int32_t> compute_gpt2_rot_indices(
    int S, int hidDim, int ffDim, int numHeads, int seqLen);

Inference make_gpt2(int logN, int hidDim, int ffDim,
                    int seqLen, int numHeads, bool parallel,
                    bool bench = true);

void gpt2_prepare_weights(Inference& inf, const std::vector<std::string>& names);
void gpt2_prepare_cache  (Inference& inf, const std::vector<std::string>& names);

// Load real GPT-2 weights from .txt files produced by prepare_gpt2_weights.py.
// Weight files have LN gamma absorbed into W columns and LN beta folded into
// bias vectors.  The per-layer absorbed bias plaintexts (b_q, b_k, ...) are
// added at runtime via EvalAdd (zero levels consumed).
// Sets bench_mode=false on inf.
void gpt2_load_weights(Inference& inf, const std::string& weight_dir, int num_layers = 12);

// ── Decoder / model ────────────────────────────────────────────────────

// layer_idx selects which layer's weights/biases to use (-1 = bench mode).
Ctx gpt2_decoder(Inference& inf, const Ctx& x, const GPT2LayerConfig& cfg = {},
                 int layer_idx = -1);
Ctx gpt2_model  (Inference& inf, const Ctx& x, const GPT2ModelConfig& cfg = GPT2_DEFAULT_CONFIG);

// Diagnostic: run layer-by-layer, decrypt after each to track error growth.
void gpt2_diag_model(Inference& inf, const Ctx& x_in, int num_layers);

// ── Zero-shot classification (non-generative inference) ─────────────────
//
// Client-server flow:
//   [CLIENT] tokenize prompt → encrypt hidden states → generate keys
//   [SERVER] gpt2_classify: full model forward → final norm →
//            lm_head inner-product for K candidate label tokens → K logit cts
//   [CLIENT] decrypt K logit ciphertexts → argmax = predicted class
//
// Parameters:
//   candidate_label_indices — token IDs of the K candidate class labels
//     (e.g. {3967, 4633} = tokens "positive" / "negative" in GPT-2 BPE)
//   classify_pos — sequence position whose hidden state is used for
//     classification (-1 = last = seqLen-1, 0 = first / CLS-style)
//
// Only computes lm_head for the K candidate tokens, avoiding the full
// 50257-vocab projection.  Uses gpt2 weight name "lm_head_<token_id>".
std::vector<Ctx> gpt2_classify(
    Inference&              inf,
    const Ctx&              x_in,
    const std::vector<int>& candidate_label_indices,
    int                     classify_pos = -1,
    const GPT2ModelConfig&  cfg          = GPT2_DEFAULT_CONFIG
);

// ── Text generation (autoregressive, client-server round-trip) ─────────
//
// Client-server flow per token:
//   [CLIENT] embed token+position → encrypt → send to server
//   [SERVER] run full model forward → lm_head projection → return logit cts
//   [CLIENT] decrypt logits → argmax → new token → repeat
//
// Returns: vector of generated token IDs.
std::vector<int> gpt2_generate(
    Inference&              inf,
    const std::vector<int>& prompt_token_ids,
    int                     max_new_tokens    = 20,
    const GPT2ModelConfig&  cfg               = GPT2_DEFAULT_CONFIG
);
