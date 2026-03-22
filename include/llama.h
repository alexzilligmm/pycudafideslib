#pragma once

#include "fideslib_wrapper.h"

#include <unordered_map>
#include <vector>
#include <string>
#include <chrono>
#include <iostream>
#include <functional>

struct LlamaSize {
    int hidDim   = 4096;
    int expDim   = 16384;
    int numHeads = 32;
    int seqLen   = 512;
};

struct SiluFunc {
    std::vector<double> cheby_coeffs;
    bool ready = false;
};

struct LlamaInference {
    std::shared_ptr<CKKSContext> fhe;

    LlamaSize size;
    int   logN        = 12;
    int   slots       = 0;
    int   total_depth = 25;

    bool parallel = true;

    std::unordered_map<std::string, std::vector<Ptx>> w;

    std::unordered_map<std::string, std::vector<Ctx>> cache;

    std::unordered_map<std::string, Ptx> mask;
    std::vector<Ptx>                     cache_mask;

    SiluFunc silu_func;

    CC& cc() { return fhe->cc; }
    const CC& cc() const { return fhe->cc; }
};

LlamaInference make_llama(int logN, int hidDim, int expDim,
                           int seqLen, int numHeads, bool parallel);

void prepare_weights(LlamaInference& llama,
                     const std::vector<std::string>& names);

void prepare_cache(LlamaInference& llama,
                   const std::vector<std::string>& names);

Ctx linear(LlamaInference& llama, const Ctx& x,
           const std::string& wname, int expand);

Ctx qkv_q(LlamaInference& llama, const Ctx& x);
Ctx qkv_k(LlamaInference& llama, const Ctx& x);
Ctx qkv_v(LlamaInference& llama, const Ctx& x);

std::tuple<Ctx, Ctx> rope(LlamaInference& llama,
                           const Ctx& q, const Ctx& k);

void cache_kv(LlamaInference& llama, const Ctx& k, const Ctx& v);

Ctx qk_transpose(LlamaInference& llama, const Ctx& q);

Ctx attn_v(LlamaInference& llama, const Ctx& s);

Ctx out_proj(LlamaInference& llama, const Ctx& x);

std::pair<Ctx, Ctx> up_gate(LlamaInference& llama, const Ctx& x);

Ctx down_proj(LlamaInference& llama, const Ctx& x);

Ctx silu      (LlamaInference& llama, const Ctx& x);
Ctx silu_expDim(LlamaInference& llama, const Ctx& x);
Ctx softmax(LlamaInference& llama, const Ctx& x,
             int target_level_after_btp, int temp);
Ctx norm   (LlamaInference& llama, const Ctx& x,
             int target_level_after_btp);
Ctx argmax (LlamaInference& llama, const Ctx& x);

Ctx decoder(LlamaInference& llama, const Ctx& x);
Ctx model  (LlamaInference& llama, const Ctx& x);

void rotate_add_inplace(LlamaInference& llama, Ctx& x, int step);

struct Timer {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    void reset() { t0 = std::chrono::steady_clock::now(); }
    double elapsed_s() const {
        using namespace std::chrono;
        return duration<double>(steady_clock::now() - t0).count();
    }
};
