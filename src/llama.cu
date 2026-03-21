// llama.cu – context setup, weight/cache preparation, Decoder, Model.

#include "llama.h"
#include <iostream>
#include <random>

Ctx bootstrap_to(const LlamaInference&, const Ctx&, uint32_t);

// ── make_llama ────────────────────────────────────────────────────────────
LlamaInference make_llama(int logN, int hidDim, int expDim,
                           int seqLen, int numHeads, bool parallel) {
    LlamaInference llama;
    llama.size     = {hidDim, expDim, numHeads, seqLen};
    llama.logN     = logN;
    llama.parallel = parallel;

    std::cout << "Creating FIDESlib/OpenFHE CKKS context (logN=" << logN << ")...\n";
    uint32_t btp_slots = (uint32_t)(1 << (logN - 1));
    llama.fhe   = make_ckks_context(logN, /*depth=*/16, /*scale_bits=*/40,
                                     btp_slots, /*bootstrap=*/true);
    llama.slots = (int)btp_slots;
    std::cout << "  slots=" << llama.slots
              << "  GPU context loaded.\n";
    return llama;
}

// ── weight / cache helpers ────────────────────────────────────────────────

static std::mt19937_64 rng_(42);

static Ptx rand_plaintext(const LlamaInference& llama) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> msg(llama.slots);
    for (auto& v : msg) v = dist(rng_);
    return llama.cc()->MakeCKKSPackedPlaintext(msg);
}

static Ctx rand_ciphertext(const LlamaInference& llama) {
    std::uniform_real_distribution<double> dist(-20.0, 0.0);
    std::vector<double> msg(llama.slots);
    for (auto& v : msg) v = dist(rng_);
    return llama.cc()->Encrypt(llama.fhe->pk(),
               llama.cc()->MakeCKKSPackedPlaintext(msg));
}

void prepare_weights(LlamaInference& llama,
                      const std::vector<std::string>& names) {
    std::cout << "Preparing weights...\n";
    const int hD = llama.size.hidDim, eD = llama.size.expDim, S = llama.slots;
    for (const auto& n : names) {
        int cnt;
        if      (n == "q"||n == "k"||n == "v"||n == "out") cnt = hD * hD / S;
        else if (n == "up"||n == "gate"||n == "down")        cnt = hD * eD / S;
        else if (n == "RoPE")                                 cnt = 3;
        else throw std::runtime_error("Unknown weight: " + n);

        llama.w[n].clear();
        for (int i = 0; i < cnt; ++i)
            llama.w[n].push_back(rand_plaintext(llama));
    }
    std::cout << "Weights ready.\n";
}

void prepare_cache(LlamaInference& llama,
                    const std::vector<std::string>& names) {
    std::cout << "Preparing cache...\n";
    const int hD = llama.size.hidDim, seqL = llama.size.seqLen;
    const int nH = llama.size.numHeads, S = llama.slots;
    for (const auto& n : names) {
        if (n == "k") {
            llama.cache["k"].clear();
            for (int i = 0; i < hD * seqL / S; ++i)
                llama.cache["k"].push_back(rand_ciphertext(llama));
            llama.mask["k"] = rand_plaintext(llama);
        } else if (n == "v") {
            llama.cache["v"].clear();
            for (int i = 0; i < hD / nH; ++i)
                llama.cache["v"].push_back(rand_ciphertext(llama));
            llama.mask["v"] = rand_plaintext(llama);
        } else if (n == "mask") {
            llama.cache_mask.clear();
            for (int i = 0; i < hD / nH; ++i)
                llama.cache_mask.push_back(rand_plaintext(llama));
        } else {
            throw std::runtime_error("Unknown cache name: " + n);
        }
    }
    std::cout << "Cache ready.\n";
}

// ── Decoder ───────────────────────────────────────────────────────────────
Ctx decoder(LlamaInference& llama, const Ctx& x_in) {
    const CC& cc = llama.cc();
    std::cout << "=== Decoder ===\n";
    Timer t;

    Ctx x = x_in;

    // ── Self-attention ────────────────────────────────────────────────────
    Ctx q, k, v;
    {
        Timer tq;
        if (llama.parallel) {
            #pragma omp parallel sections
            {
                #pragma omp section  { q = qkv_q(llama, x); }
                #pragma omp section  { k = qkv_k(llama, x); }
                #pragma omp section  { v = qkv_v(llama, x); }
            }
        } else {
            q = qkv_q(llama, x); k = qkv_k(llama, x); v = qkv_v(llama, x);
        }
        std::cout << "  QKV: " << tq.elapsed_s() << " s\n";
    }

    auto [yq, yk] = rope(llama, q, k);
    q = yq; k = yk;

    cache_kv(llama, k, v);

    Ctx s = qk_transpose(llama, q);
    s = bootstrap_to(llama, s, 12);
    s = softmax(llama, s, 14, 0);

    Ctx o = attn_v(llama, s);
    o = out_proj(llama, o);

    match_level(cc, x, o);
    cc->EvalAddInPlace(x, o);                // residual

    x = bootstrap_to(llama, x, 15);
    x = norm(llama, x, 9);

    // ── FFN ───────────────────────────────────────────────────────────────
    auto [up_ct, gate_ct] = up_gate(llama, x);

    gate_ct = bootstrap_to(llama, gate_ct, 9);
    gate_ct = silu(llama, gate_ct);

    match_level(cc, up_ct, gate_ct);
    Ctx y = cc->EvalMult(up_ct, gate_ct);

    y = down_proj(llama, y);

    match_level(cc, x, y);
    cc->EvalAddInPlace(x, y);                // residual
    y = x;

    y = bootstrap_to(llama, y, 15);
    y = norm(llama, y, 11);

    std::cout << "Decoder done in " << t.elapsed_s() << " s\n";
    return y;
}

// ── Model ─────────────────────────────────────────────────────────────────
Ctx model(LlamaInference& llama, const Ctx& x_in) {
    Timer t;
    Ctx x = x_in;
    for (int i = 0; i < 32; ++i) {
        std::cout << "--- Layer " << i << " ---\n";
        x = decoder(llama, x);
    }
    std::cout << "Model complete in " << t.elapsed_s() << " s\n";
    return x;
}
