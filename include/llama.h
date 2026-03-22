#pragma once
// llama.h – LLaMA FHE inference types and declarations.
// Backend: FIDESlib (OpenFHE 1.4.2 + GPU acceleration + full bootstrapping).

#include "fideslib_wrapper.h"

#include <unordered_map>
#include <vector>
#include <string>
#include <chrono>
#include <iostream>
#include <functional>

// ── model dimensions ──────────────────────────────────────────────────────
struct LlamaSize {
    int hidDim   = 4096;
    int expDim   = 16384;
    int numHeads = 32;
    int seqLen   = 512;
};

// ── SiLU polynomial cache ─────────────────────────────────────────────────
struct SiluFunc {
    std::vector<double> cheby_coeffs;   // Chebyshev coefficients on [-20,20]
    bool ready = false;
};

// ── main inference struct ─────────────────────────────────────────────────
struct LlamaInference {
    std::shared_ptr<CKKSContext> fhe;   // FIDESlib context + keys

    LlamaSize size;
    int   logN        = 12;
    int   slots       = 0;    // N/2
    int   total_depth = 25;   // depth + approx_bootstrap_depth used in make_ckks_context
                               // Used by bootstrap_to to convert Go-style "remaining moduli"
                               // target levels to CUDA-style "consumed" levels.

    bool parallel = true;

    // Weights: map name → vector of plaintexts
    std::unordered_map<std::string, std::vector<Ptx>> w;

    // KV cache: map name → vector of ciphertexts
    std::unordered_map<std::string, std::vector<Ctx>> cache;

    // Masks
    std::unordered_map<std::string, Ptx> mask;
    std::vector<Ptx>                     cache_mask;

    // SiLU evaluator
    SiluFunc silu_func;

    // Convenience accessor
    CC& cc() { return fhe->cc; }
    const CC& cc() const { return fhe->cc; }
};

// ── context + weight/cache setup ─────────────────────────────────────────
LlamaInference make_llama(int logN, int hidDim, int expDim,
                           int seqLen, int numHeads, bool parallel);

void prepare_weights(LlamaInference& llama,
                     const std::vector<std::string>& names);

void prepare_cache(LlamaInference& llama,
                   const std::vector<std::string>& names);

// ── linear layer operations ───────────────────────────────────────────────
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

// ── nonlinear operations ──────────────────────────────────────────────────
// silu: apply SiLU to all slots (CUDA default, correct for FFN gate outputs).
// silu_expDim: zeros slots outside [0, expDim) after SiLU — matches Go's
//   nonZero slot mapping (polynomial.NewPolynomialVector with expDim slots).
Ctx silu      (LlamaInference& llama, const Ctx& x);
Ctx silu_expDim(LlamaInference& llama, const Ctx& x);
Ctx softmax(LlamaInference& llama, const Ctx& x,
             int target_level_after_btp, int temp);
Ctx norm   (LlamaInference& llama, const Ctx& x,
             int target_level_after_btp);
Ctx argmax (LlamaInference& llama, const Ctx& x);

// ── decoder / model ───────────────────────────────────────────────────────
Ctx decoder(LlamaInference& llama, const Ctx& x);
Ctx model  (LlamaInference& llama, const Ctx& x);

// ── utility ───────────────────────────────────────────────────────────────
// rotate x by step and add to itself in-place
void rotate_add_inplace(LlamaInference& llama, Ctx& x, int step);

struct Timer {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    void reset() { t0 = std::chrono::steady_clock::now(); }
    double elapsed_s() const {
        using namespace std::chrono;
        return duration<double>(steady_clock::now() - t0).count();
    }
};
