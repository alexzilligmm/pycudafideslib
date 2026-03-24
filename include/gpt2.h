#pragma once

#include "inference.h"
#include "nonlinear.h"

Inference make_gpt2(int logN, int hidDim, int ffDim,
                    int seqLen, int numHeads, bool parallel);

void gpt2_prepare_weights(Inference& inf, const std::vector<std::string>& names);
void gpt2_prepare_cache  (Inference& inf, const std::vector<std::string>& names);

// Nonlinear ops (gelu, norm, softmax, ...) are declared in nonlinear.h (included above)

Ctx gpt2_decoder(Inference& inf, const Ctx& x);
Ctx gpt2_model  (Inference& inf, const Ctx& x);
