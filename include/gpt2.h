#pragma once

#include "inference.h"

Inference make_gpt2(int logN, int hidDim, int ffDim,
                    int seqLen, int numHeads, bool parallel);

void gpt2_prepare_weights(Inference& inf, const std::vector<std::string>& names);
void gpt2_prepare_cache  (Inference& inf, const std::vector<std::string>& names);

// GPT-2 uses GELU activation instead of SiLU.
// rescale_factor scales (x - threshold) into [-1, 1] before the sign comparison.
Ctx gelu(Inference& inf, const Ctx& x, double rescale_factor);

Ctx gpt2_decoder(Inference& inf, const Ctx& x);
Ctx gpt2_model  (Inference& inf, const Ctx& x);
