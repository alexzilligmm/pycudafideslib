// gpt2.cu – GPT-2 context setup, weight/cache preparation, decoder, model.
//
// GPT-2 (small) vs LLaMA:
//   - 12 layers (not 32)
//   - No RoPE (learned position embeddings, handled at input)
//   - No gated MLP: Up -> GELU -> Down (no gate, no elem_mult)
//   - GELU activation instead of SiLU
//   - Pre-norm (LayerNorm before attention and MLP)
//   - Biases on all linear layers (absorbed into weight plaintexts)
//
// Reuses from llama.cu / linear.cu:
//   linear(), qkv_q/k/v, qk_transpose, attn_v, out_proj, down_proj,
//   cache_kv, rotate_add_inplace

#include "gpt2.h"
#include "llama.h"     // reuse linear ops
#include <iostream>
#include <random>
#include <omp.h>

// ── Context setup ───────────────────────────────────────────────────────

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

// ── Weight / cache preparation ──────────────────────────────────────────

static std::mt19937_64 gpt2_rng_(42);

static Ptx gpt2_rand_plaintext(Inference& inf) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> msg(inf.slots);
    for (auto& v : msg) v = dist(gpt2_rng_);
    return inf.cc()->MakeCKKSPackedPlaintext(msg);
}

static Ctx gpt2_rand_ciphertext(Inference& inf) {
    std::uniform_real_distribution<double> dist(-20.0, 0.0);
    std::vector<double> msg(inf.slots);
    for (auto& v : msg) v = dist(gpt2_rng_);
    Ptx pt = inf.cc()->MakeCKKSPackedPlaintext(msg);
    return inf.cc()->Encrypt(inf.fhe->pk(), pt);
}

void gpt2_prepare_weights(Inference& inf, const std::vector<std::string>& names) {
    std::cout << "Preparing GPT-2 weights...\n";
    const int hD = inf.size.hidDim, fD = inf.size.ffDim, S = inf.slots;
    for (const auto& n : names) {
        int cnt;
        // GPT-2 has no gate, no RoPE
        if      (n == "q" || n == "k" || n == "v" || n == "out") cnt = hD * hD / S;
        else if (n == "up")                                       cnt = hD * fD / S;
        else if (n == "down")                                     cnt = hD * fD / S;
        else throw std::runtime_error("Unknown GPT-2 weight: " + n);

        inf.w[n].clear();
        for (int i = 0; i < cnt; ++i)
            inf.w[n].push_back(gpt2_rand_plaintext(inf));
    }
    std::cout << "GPT-2 weights ready.\n";
}

void gpt2_prepare_cache(Inference& inf, const std::vector<std::string>& names) {
    std::cout << "Preparing GPT-2 cache...\n";
    const int hD = inf.size.hidDim, seqL = inf.size.seqLen;
    const int nH = inf.size.numHeads, S = inf.slots;
    for (const auto& n : names) {
        if (n == "k") {
            inf.cache["k"].clear();
            for (int i = 0; i < hD * seqL / S; ++i)
                inf.cache["k"].push_back(gpt2_rand_ciphertext(inf));
            inf.mask["k"] = gpt2_rand_plaintext(inf);
        } else if (n == "v") {
            inf.cache["v"].clear();
            for (int i = 0; i < hD / nH; ++i)
                inf.cache["v"].push_back(gpt2_rand_ciphertext(inf));
            inf.mask["v"] = gpt2_rand_plaintext(inf);
        } else if (n == "mask") {
            inf.cache_mask.clear();
            for (int i = 0; i < hD / nH; ++i)
                inf.cache_mask.push_back(gpt2_rand_plaintext(inf));
        } else {
            throw std::runtime_error("Unknown GPT-2 cache: " + n);
        }
    }
    std::cout << "GPT-2 cache ready.\n";
}

// ── GPT-2 Decoder ───────────────────────────────────────────────────────
//
// Pipeline (per layer):
//
//   ┌─ x_in ──────────────────────────────────────────────────────────┐
//   │                                                                  │
//   │  1. LayerNorm1                                                   │
//   │  2. QKV projections (reuse linear.cu — no RoPE)                  │
//   │  3. KV cache update                                              │
//   │  4. QK^T -> bootstrap -> Softmax -> AttnV -> OutProj             │
//   │  5. Residual: x = x_in + attn_out                                │
//   │                                                                  │
//   │  6. LayerNorm2                                                   │
//   │  7. Up projection (single, no gate)                              │
//   │  8. bootstrap -> GELU                                            │
//   │  9. Down projection                                              │
//   │ 10. Residual: x = x + mlp_out                                   │
//   │                                                                  │
//   │ 11. bootstrap -> LayerNorm (for next layer's input)              │
//   └─ return x ──────────────────────────────────────────────────────┘

Ctx gpt2_decoder(Inference& inf, const Ctx& x_in) {
    const CC& cc = inf.cc();
    std::cout << "=== GPT-2 Decoder ===\n";
    Timer t;

    // ── 1. Pre-attention LayerNorm ──
    Ctx x_normed = bootstrap_to(inf, x_in, 15);
    x_normed = norm(inf, x_normed, 9);

    // ── 2. QKV projections (no RoPE) ──
    Ctx q, k, v;
    {
        Timer tq;
        if (inf.parallel) {
            #pragma omp parallel sections
            {
                #pragma omp section
                { q = qkv_q(inf, x_normed); }
                #pragma omp section
                { k = qkv_k(inf, x_normed); }
                #pragma omp section
                { v = qkv_v(inf, x_normed); }
            }
        } else {
            q = qkv_q(inf, x_normed);
            k = qkv_k(inf, x_normed);
            v = qkv_v(inf, x_normed);
        }
        std::cout << "  QKV: " << tq.elapsed_s() << " s\n";
    }

    // ── 3. KV cache (no RoPE rotation needed) ──
    cache_kv(inf, k, v);

    // ── 4. Attention: QK^T -> softmax -> AttnV -> OutProj ──
    Ctx s = qk_transpose(inf, q);
    s = bootstrap_to(inf, s, 12);
    s = softmax_cachemir(inf, s, 14, 0);

    Ctx attn_out = attn_v(inf, s);
    attn_out = out_proj(inf, attn_out);

    // ── 5. Residual connection (attention) ──
    Ctx x = x_in;
    cc->EvalAddInPlace(x, attn_out);

    // ── 6. Pre-MLP LayerNorm ──
    x_normed = bootstrap_to(inf, x, 15);
    x_normed = norm(inf, x_normed, 9);

    // ── 7. Up projection (single — no gate in GPT-2) ──
    Ctx up_ct = linear(inf, x_normed, "up", 1);

    // ── 8. GELU activation ──
    up_ct = bootstrap_to(inf, up_ct, 9);
    Ctx y = gelu(inf, up_ct, GELU_ENCLLM_GPT2);

    // ── 9. Down projection ──
    y = down_proj(inf, y);

    // ── 10. Residual connection (MLP) ──
    cc->EvalAddInPlace(x, y);

    std::cout << "GPT-2 Decoder done in " << t.elapsed_s() << " s\n";
    return x;
}

// ── GPT-2 Model (12 layers) ────────────────────────────────────────────

Ctx gpt2_model(Inference& inf, const Ctx& x_in) {
    Timer t;
    Ctx x = x_in;
    for (int i = 0; i < 12; ++i) {
        std::cout << "--- GPT-2 Layer " << i << " ---\n";
        x = gpt2_decoder(inf, x);
    }
    // Final LayerNorm
    x = bootstrap_to(inf, x, 15);
    x = norm(inf, x, 11);
    std::cout << "GPT-2 model complete in " << t.elapsed_s() << " s\n";
    return x;
}
