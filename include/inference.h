#pragma once

#include "fideslib_wrapper.h"

#include <unordered_map>
#include <vector>
#include <string>
#include <chrono>
#include <iostream>
#include <functional>

struct ModelSize {
    int dim        = 768;
    int expanded   = 3072;   // feed-forward / expansion dimension
    int hidDim     = 1024;  // dim padded to closed power of 2
    int expDim     = 4096;  // expanded dim padded to closed power of 2
    int numHeads   = 12;
    int seqLen     = 1024;

    int getRealHidDim() const { return dim; }
    int getRealFfDim()  const { return expanded;}
};

// Generic FHE inference context shared by all model types.
struct Inference {
    std::shared_ptr<CKKSContext> fhe;

    ModelSize size;
    int   logN        = 16;
    int   slots       = 0;
    int   total_depth = 25; // TODO: check this

    bool parallel        = true;
    bool bench_mode      = false;  // true → all rotations use index 5 (minimal keys)

    std::string weight_dir;       

    std::unordered_map<std::string, std::vector<Ptx>> w;
    std::unordered_map<std::string, std::vector<std::vector<double>>> raw_w;
    std::unordered_map<std::string, std::vector<Ctx>> cache;
    std::unordered_map<std::string, Ptx> mask;
    std::vector<Ptx>                     cache_mask;

    int k_count = 0;   // number of keys pushed into K cache
    int v_count = 0;   // number of values pushed into V cache

    CC& cc() { return fhe->cc; }
    const CC& cc() const { return fhe->cc; }
};

Ctx bootstrap_to(Inference& inf, const Ctx& ct, uint32_t target_remaining);

