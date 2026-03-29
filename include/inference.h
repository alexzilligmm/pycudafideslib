#pragma once

#include "fideslib_wrapper.h"

#include <unordered_map>
#include <vector>
#include <string>
#include <chrono>
#include <iostream>
#include <functional>

// Generic model dimensions — works for GPT-2, LLaMA, etc.
// hidDim/ffDim may be padded to powers of 2 for BSGS (e.g. 768→1024).
// realHidDim/realFfDim store the unpadded values used for normalization.
struct ModelSize {
    int hidDim     = 768;
    int ffDim      = 3072;   // feed-forward / expansion dimension
    int numHeads   = 12;
    int seqLen     = 1024;
    int realHidDim = 0;      // 0 → same as hidDim (no padding)
    int realFfDim  = 0;      // 0 → same as ffDim  (no padding)

    // Return the actual (unpadded) dimensions for normalization.
    int getRealHidDim() const { return realHidDim > 0 ? realHidDim : hidDim; }
    int getRealFfDim()  const { return realFfDim  > 0 ? realFfDim  : ffDim;  }
};

// Generic FHE inference context shared by all model types.
struct Inference {
    std::shared_ptr<CKKSContext> fhe;

    ModelSize size;
    int   logN        = 12;
    int   slots       = 0;
    int   total_depth = 25;

    bool parallel        = true;
    bool bench_mode      = false;  // true → all rotations use index 5 (minimal keys)

    std::string weight_dir;         // path to weight files (for per-layer streaming)

    std::unordered_map<std::string, std::vector<Ptx>> w;
    // Raw (non-encoded) weight storage for client-side embeddings.
    // wte/wpe/lm_head are 50k+ rows — encoding all to Ptx causes OOM at logN=16.
    // Only the rows needed at runtime are encoded to Ptx on-demand.
    std::unordered_map<std::string, std::vector<std::vector<double>>> raw_w;
    std::unordered_map<std::string, std::vector<Ctx>> cache;
    std::unordered_map<std::string, Ptx> mask;
    std::vector<Ptx>                     cache_mask;

    CC& cc() { return fhe->cc; }
    const CC& cc() const { return fhe->cc; }
};

// Bootstraps ct to at least target_remaining levels of budget remaining.
Ctx bootstrap_to(Inference& inf, const Ctx& ct, uint32_t target_remaining);

struct Timer {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    void reset() { t0 = std::chrono::steady_clock::now(); }
    double elapsed_s() const {
        using namespace std::chrono;
        return duration<double>(steady_clock::now() - t0).count();
    }
};
