#pragma once

#include "inference.h"

Inference make_llama(int logN, int hidDim, int ffDim,
                     int seqLen, int numHeads, bool parallel);

void prepare_weights(Inference& inf, const std::vector<std::string>& names);
void prepare_cache  (Inference& inf, const std::vector<std::string>& names);

Ctx linear(Inference& inf, const Ctx& x, const std::string& wname, int expand);

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

// Nonlinear ops (shared across models via nonlinear.cu)
Ctx silu     (Inference& inf, const Ctx& x);
Ctx silu_ffDim(Inference& inf, const Ctx& x);   // SiLU masked to ffDim slots
Ctx softmax  (Inference& inf, const Ctx& x, int target_level_after_btp, int temp);
Ctx norm     (Inference& inf, const Ctx& x, int target_level_after_btp);
Ctx sign     (Inference& inf, const Ctx& x);                    // sign(x) via F4∘F4∘G4∘G4
Ctx lt_function(Inference& inf, const Ctx& x, double value,
                double rescale_factor = 1.0);                    // ~1 if x < value
Ctx gelu     (Inference& inf, const Ctx& x, double rescale_factor); // piecewise on 4 intervals
Ctx argmax   (Inference& inf, const Ctx& x);

// Full Llama decoder layer and stacked model
Ctx decoder(Inference& inf, const Ctx& x);
Ctx model  (Inference& inf, const Ctx& x);

void rotate_add_inplace(Inference& inf, Ctx& x, int step);
