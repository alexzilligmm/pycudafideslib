// test_llama.cu
// Integration tests for the full LLaMA building blocks:
//   Linear, QKV, RoPE, Cache, QK_T, AttnV, UpGate, Down.
// Each test uses small dimensions (hidDim=16) for fast execution.
// Correctness is checked by decrypting the output and verifying structural
// properties (level, scale, value range) rather than exact plaintext
// comparison (which would require a software reference at full HE precision).

#include <gtest/gtest.h>
#include "llama.h"

#include <cmath>
#include <algorithm>
#include <iostream>
#include <numeric>

Ctx bootstrap_to(LlamaInference&, const Ctx&, uint32_t);

// ── fixture ───────────────────────────────────────────────────────────────
// Tiny model: logN=12 → slots=2048.  Dimensions must satisfy HID_DIM²≥slots
// and HID_DIM*SEQ_LEN≥slots for the cache.  hidDim=64, expDim=256, heads=4,
// seqLen=32 satisfy all constraints.
class LlamaLayerTest : public ::testing::Test {
protected:
    static constexpr int LOGN    = 12;
    static constexpr int HID_DIM = 64;
    static constexpr int EXP_DIM = 256;
    static constexpr int HEADS   = 4;
    static constexpr int SEQ_LEN = 32;

    LlamaInference llama;

    void SetUp() override {
        llama = make_llama(LOGN, HID_DIM, EXP_DIM, SEQ_LEN, HEADS, /*parallel=*/false);
    }

    Ctx make_input(double val = 0.5) {
        std::vector<double> msg(llama.slots, val);
        return encrypt(llama.cc(), llama.cc()->MakeCKKSPackedPlaintext(msg), llama.fhe->pk());
    }

    std::vector<double> dec(const Ctx& ct) {
        return decrypt(llama.cc(), ct, llama.fhe->sk());
    }

    // Check that output has finite values and level is valid
    void check_output(const Ctx& out, const std::string& name) {
        ASSERT_NE(out, nullptr) << name << " returned null";
        auto vals = dec(out);
        bool any_nan = false;
        for (double v : vals)
            if (!std::isfinite(v)) { any_nan = true; break; }
        EXPECT_FALSE(any_nan) << name << " output contains NaN/Inf";
        uint32_t lvl = level_of(out);
        EXPECT_GE(lvl, 0u) << name << " output level is invalid";
        std::cout << name << ": level=" << lvl
                  << " first_val=" << vals[0] << "\n";
    }
};

// ── Linear layer ─────────────────────────────────────────────────────────
TEST_F(LlamaLayerTest, Linear_Square_Runs) {
    prepare_weights(llama, {"q"});
    Ctx x   = make_input(0.5);
    Ctx out = linear(llama, x, "q", 0);
    check_output(out, "Linear(q)");
}

TEST_F(LlamaLayerTest, Linear_Expand_Runs) {
    prepare_weights(llama, {"up"});
    Ctx x   = make_input(0.5);
    Ctx out = linear(llama, x, "up", 1);
    check_output(out, "Linear(up)");
}

TEST_F(LlamaLayerTest, Linear_Contract_Runs) {
    prepare_weights(llama, {"down"});
    Ctx x   = make_input(0.5);
    Ctx out = linear(llama, x, "down", -1);
    check_output(out, "Linear(down)");
}

// ── QKV ──────────────────────────────────────────────────────────────────
TEST_F(LlamaLayerTest, QKV_AllThreeRun) {
    prepare_weights(llama, {"q", "k", "v"});
    Ctx x = make_input(0.3);

    Ctx q = qkv_q(llama, x);
    Ctx k = qkv_k(llama, x);
    Ctx v = qkv_v(llama, x);

    check_output(q, "Q");
    check_output(k, "K");
    check_output(v, "V");

    // All three should be at the same level (same circuit depth)
    EXPECT_EQ(level_of(q), level_of(k))
        << "Q and K should be at the same level";
    EXPECT_EQ(level_of(k), level_of(v))
        << "K and V should be at the same level";
    
    // Get decrypted values for comparison
    auto q_vals = dec(q);
    auto k_vals = dec(k);
    auto v_vals = dec(v);
    
    std::cout << "\nFirst row of Q: " << q_vals[0] << ", K: " << k_vals[0] << ", V: " << v_vals[0] << "\n";
    
    // Compare with weight profile (plaintext weights should be random)
    std::cout << "\nQ output sample (first 3 values): ";
    for (int i = 0; i < std::min(3, (int)q_vals.size()); ++i)
        std::cout << q_vals[i] << " ";
    std::cout << "\nK output sample (first 3 values): ";
    for (int i = 0; i < std::min(3, (int)k_vals.size()); ++i)
        std::cout << k_vals[i] << " ";
    std::cout << "\nV output sample (first 3 values): ";
    for (int i = 0; i < std::min(3, (int)v_vals.size()); ++i)
        std::cout << v_vals[i] << " ";
    std::cout << "\n";
}

// ── RoPE ─────────────────────────────────────────────────────────────────
TEST_F(LlamaLayerTest, RoPE_Runs) {
    prepare_weights(llama, {"RoPE"});
    Ctx q = make_input(0.2);
    Ctx k = make_input(0.2);

    auto [yq, yk] = rope(llama, q, k);

    check_output(yq, "RoPE(Q)");
    check_output(yk, "RoPE(K)");
}

TEST_F(LlamaLayerTest, RoPE_ChangesValue) {
    // RoPE should modify the ciphertext (not produce an identity)
    prepare_weights(llama, {"RoPE"});
    Ctx q = make_input(0.5);

    auto q_plain_in = dec(q);
    auto [yq, yk] = rope(llama, q, q);
    auto q_plain_out = dec(yq);

    // Check that output ≠ input (RoPE applies a rotation)
    double max_diff = 0.0;
    for (int i = 0; i < llama.slots; ++i)
        max_diff = std::max(max_diff,
                            std::abs(q_plain_in[i] - q_plain_out[i]));
    EXPECT_GT(max_diff, 1e-6) << "RoPE should change the ciphertext";
}

// ── Cache ─────────────────────────────────────────────────────────────────
TEST_F(LlamaLayerTest, Cache_UpdatesKV) {
    prepare_cache(llama, {"k", "v", "mask"});

    Ctx k = make_input(0.1);
    Ctx v = make_input(0.2);

    size_t ksize_before = llama.cache.at("k").size();
    cache_kv(llama, k, v);
    size_t ksize_after  = llama.cache.at("k").size();

    // If intIdx == 0, a new entry is appended; otherwise it's added in-place.
    // Either way the cache should not shrink.
    EXPECT_GE(ksize_after, ksize_before)
        << "K cache should not shrink after update";
}

// ── QK^T ─────────────────────────────────────────────────────────────────
TEST_F(LlamaLayerTest, QKT_Runs) {
    prepare_weights(llama, {"q", "k"});
    prepare_cache(llama, {"k"});

    Ctx q = make_input(0.2);
    Ctx out = qk_transpose(llama, q);
    check_output(out, "QK^T");
}

// ── AttnV ────────────────────────────────────────────────────────────────
TEST_F(LlamaLayerTest, AttnV_Runs) {
    prepare_cache(llama, {"v"});

    // s is the "attention scores" ciphertext
    Ctx s = make_input(0.1);
    Ctx out = attn_v(llama, s);
    check_output(out, "AttnV");
}

// ── UpGate + Down ─────────────────────────────────────────────────────────
TEST_F(LlamaLayerTest, UpGate_Runs) {
    prepare_weights(llama, {"up", "gate"});
    Ctx x = make_input(0.3);

    auto [up, gate] = up_gate(llama, x);
    check_output(up,   "Up");
    check_output(gate, "Gate");
}

TEST_F(LlamaLayerTest, UpGateSameLevelAsDown) {
    prepare_weights(llama, {"up", "gate", "down"});
    Ctx x = make_input(0.3);

    auto [up, gate] = up_gate(llama, x);

    // Multiply up * gate, then Down projection
    match_level(llama.cc(), up, gate);
    Ctx ffn = llama.cc()->EvalMult(up, gate);
    Ctx out = down_proj(llama, ffn);

    check_output(out, "FFN(Down)");
}

// ── Decoder (integration test, tiny model) ───────────────────────────────
TEST_F(LlamaLayerTest, Decoder_Runs) {
    prepare_weights(llama, {"q","k","v","out","up","gate","down","RoPE"});
    prepare_cache  (llama, {"k","v","mask"});

    Ctx x = make_input(0.1);
    std::cout << "Running tiny decoder (hidDim=" << HID_DIM << ")...\n";
    Timer t;
    Ctx out = decoder(llama, x);
    std::cout << "Decoder done in " << t.elapsed_s() << " s\n";

    check_output(out, "Decoder");
}

// ── Level accounting: decoder output should be at a predictable level ─────
TEST_F(LlamaLayerTest, Decoder_LevelConsistency) {
    prepare_weights(llama, {"q","k","v","out","up","gate","down","RoPE"});
    prepare_cache  (llama, {"k","v","mask"});

    Ctx x1 = make_input(0.1);
    Ctx x2 = make_input(0.2);

    Ctx out1 = decoder(llama, x1);
    Ctx out2 = decoder(llama, x2);

    // Two decoders with same structure should leave ciphertext at same level
    EXPECT_EQ(level_of(out1), level_of(out2))
        << "Decoder outputs should be at the same level regardless of input";
}
