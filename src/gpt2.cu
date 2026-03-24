// gpt2.cu – GPT-2 context setup and model components.

#include "gpt2.h"
#include <iostream>
#include <random>

Inference make_gpt2(int logN, int hidDim, int ffDim,
                    int seqLen, int numHeads, bool parallel) {
    Inference inf;
    inf.size     = {hidDim, ffDim, numHeads, seqLen};
    inf.logN     = logN;
    inf.parallel = parallel;

    std::cout << "Creating FIDESlib/OpenFHE CKKS context for GPT-2 (logN=" << logN << ")...\n";
    uint32_t btp_slots = (uint32_t)(1 << (logN - 1));
    constexpr int kDepth    = 24;
    constexpr int kBtpExtra =  9;
    inf.fhe         = make_ckks_context(logN, kDepth, /*scale_bits=*/40,
                                        btp_slots, /*bootstrap=*/true);
    inf.slots       = (int)btp_slots;
    inf.total_depth = kDepth + kBtpExtra;
    std::cout << "  slots=" << inf.slots << "  GPU context loaded.\n";
    return inf;
}

// TODO: implement GPT-2 weight/cache preparation (similar to llama.cu but
//       without the gate projection — GPT-2 uses a single up projection + GELU)
void gpt2_prepare_weights(Inference& inf, const std::vector<std::string>& names) {
    throw std::runtime_error("gpt2_prepare_weights: not yet implemented");
}

void gpt2_prepare_cache(Inference& inf, const std::vector<std::string>& names) {
    throw std::runtime_error("gpt2_prepare_cache: not yet implemented");
}

// TODO: GPT-2 decoder layer
//   differences vs LLaMA:
//   - no RoPE (uses learned position embeddings)
//   - no gated MLP (single linear + GELU instead of SiLU gate)
//   - pre-norm (LayerNorm before attention + MLP, not after)
Ctx gpt2_decoder(Inference& inf, const Ctx& x_in) {
    throw std::runtime_error("gpt2_decoder: not yet implemented");
    return x_in;
}

Ctx gpt2_model(Inference& inf, const Ctx& x_in) {
    Timer t;
    Ctx x = x_in;
    for (int i = 0; i < 12; ++i) {   // GPT-2 small: 12 layers
        std::cout << "--- Layer " << i << " ---\n";
        x = gpt2_decoder(inf, x);
    }
    std::cout << "GPT-2 model complete in " << t.elapsed_s() << " s\n";
    return x;
}
