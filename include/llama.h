#pragma once

#include "inference.h"
#include "nonlinear.h"

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

Ctx decoder(Inference& inf, const Ctx& x);
Ctx model  (Inference& inf, const Ctx& x);

void rotate_add_inplace(Inference& inf, Ctx& x, int step);
