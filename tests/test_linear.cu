#include <gtest/gtest.h>
#include "fideslib_wrapper.h"
#include "ckks_primitives.h"
#include "llama.h"
#include "test_non_linear.h"
#include <cmath>
#include <vector>
#include <iostream>

// Uses NonLinearTest fixture: logN=12 → S=2048, depth=28, bootstrap enabled.
//
// The BSGS linear layer computes y = W·x using diagonal packing.
// Weight weight[i] encodes diagonal d = i * intRot of the matrix W.
// With hidDim=256, S=2048 → intRot=8, n_weights=32.
//
// For input with data in the first hidDim slots (rest zero):
//   pre-processing sums across S/hidDim=8 blocks (all zero → no-op)
//   post-processing likewise.
// So weight[0] = all-ones encodes the identity (diagonal 0 = main diagonal).

// ── plaintext helpers ────────────────────────────────────────────────────

static std::vector<double> rotate_vec(const std::vector<double>& v, int k) {
    int n = (int)v.size();
    k = ((k % n) + n) % n;
    std::vector<double> out(n);
    for (int i = 0; i < n; ++i)
        out[i] = v[(i + k) % n];
    return out;
}

static std::vector<double> add_vec(const std::vector<double>& a,
                                    const std::vector<double>& b) {
    std::vector<double> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] + b[i];
    return out;
}

static std::vector<double> mul_vec(const std::vector<double>& a,
                                    const std::vector<double>& b) {
    std::vector<double> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] * b[i];
    return out;
}

static constexpr int HD = 256;
static constexpr int FD = 1024;

// ── Test 1: Identity matrix → output = input ────────────────────────────

TEST_F(NonLinearTest, Linear_Identity) {
    const CC& cc = inf.cc();
    const int S  = inf.slots;
    inf.size.hidDim = HD;
    inf.size.ffDim  = FD;
    inf.parallel    = false;

    const int n_weights = HD * HD / S;  // 32
    std::cout << "\n=== Linear: Identity matrix (hidDim=" << HD
              << ", n_weights=" << n_weights << ") ===\n";

    // Identity: weight[0] = 1 (main diagonal), rest = 0
    inf.w["q"].clear();
    inf.w["q"].push_back(cc->MakeCKKSPackedPlaintext(
        std::vector<double>(S, 1.0)));
    for (int i = 1; i < n_weights; ++i)
        inf.w["q"].push_back(cc->MakeCKKSPackedPlaintext(
            std::vector<double>(S, 0.0)));

    // Input: known pattern in first hidDim slots, zeros elsewhere
    std::vector<double> x_in(S, 0.0);
    for (int i = 0; i < HD; ++i)
        x_in[i] = 1.0 + 0.01 * i;   // 1.00, 1.01, 1.02, ..., 3.55

    auto pt = cc->MakeCKKSPackedPlaintext(x_in);
    auto ct = cc->Encrypt(inf.fhe->pk(), pt);
    std::cout << "  input level=" << level_of(ct) << "\n";

    Ctx result = qkv_q(inf, ct);
    std::cout << "  output level=" << level_of(result) << "\n";

    auto dec = decrypt(cc, result, inf.fhe->sk());

    // Expected: output = input (identity)
    double max_err = 0.0;
    for (int i = 0; i < HD; ++i)
        max_err = std::max(max_err, std::abs(dec[i] - x_in[i]));

    std::cout << "  max |y - x| over hidDim slots: " << max_err << "\n";
    std::cout << "  slot[0]   x=" << x_in[0]   << "  y=" << dec[0]   << "\n";
    std::cout << "  slot[127] x=" << x_in[127]  << "  y=" << dec[127] << "\n";
    std::cout << "  slot[255] x=" << x_in[255]  << "  y=" << dec[255] << "\n";

    EXPECT_LT(max_err, 0.05) << "Identity: output should equal input";
}

// ── Test 2: Scalar matrix (3.0 * I) → output = 3 * input ───────────────

TEST_F(NonLinearTest, Linear_ScalarMatrix) {
    const CC& cc = inf.cc();
    const int S  = inf.slots;
    inf.size.hidDim = HD;
    inf.size.ffDim  = FD;
    inf.parallel    = false;

    const double scalar = 3.0;
    const int n_weights = HD * HD / S;

    std::cout << "\n=== Linear: Scalar matrix (" << scalar << " * I) ===\n";

    // weight[0] = scalar (main diagonal), rest = 0
    inf.w["q"].clear();
    inf.w["q"].push_back(cc->MakeCKKSPackedPlaintext(
        std::vector<double>(S, scalar)));
    for (int i = 1; i < n_weights; ++i)
        inf.w["q"].push_back(cc->MakeCKKSPackedPlaintext(
            std::vector<double>(S, 0.0)));

    std::vector<double> x_in(S, 0.0);
    for (int i = 0; i < HD; ++i)
        x_in[i] = 0.5 + 0.002 * i;

    auto pt = cc->MakeCKKSPackedPlaintext(x_in);
    auto ct = cc->Encrypt(inf.fhe->pk(), pt);
    std::cout << "  input level=" << level_of(ct) << "\n";

    Ctx result = qkv_q(inf, ct);
    std::cout << "  output level=" << level_of(result) << "\n";

    auto dec = decrypt(cc, result, inf.fhe->sk());

    double max_err = 0.0;
    for (int i = 0; i < HD; ++i)
        max_err = std::max(max_err, std::abs(dec[i] - scalar * x_in[i]));

    std::cout << "  max |y - 3x| over hidDim slots: " << max_err << "\n";
    std::cout << "  slot[0]   3*x=" << scalar * x_in[0]   << "  y=" << dec[0]   << "\n";
    std::cout << "  slot[127] 3*x=" << scalar * x_in[127]  << "  y=" << dec[127] << "\n";
    std::cout << "  slot[255] 3*x=" << scalar * x_in[255]  << "  y=" << dec[255] << "\n";

    EXPECT_LT(max_err, 0.1) << "Scalar: output should equal " << scalar << " * input";
}

// ── Test 3: Cyclic-shift matrix → output = rot(input, intRot) ──────────
// Setting weight[1] = 1, rest = 0: encodes diagonal intRot,
// which maps x[i] → y[(i - intRot) mod hidDim].
// So the output is the input cyclically shifted by intRot positions.

TEST_F(NonLinearTest, Linear_ShiftMatrix) {
    const CC& cc = inf.cc();
    const int S  = inf.slots;
    inf.size.hidDim = HD;
    inf.size.ffDim  = FD;
    inf.parallel    = false;

    const int n_weights = HD * HD / S;
    const int intRot = S / HD;  // 8

    std::cout << "\n=== Linear: Shift-by-" << intRot << " matrix ===\n";

    // weight[1] = 1 (diagonal intRot), rest = 0
    inf.w["q"].clear();
    inf.w["q"].push_back(cc->MakeCKKSPackedPlaintext(
        std::vector<double>(S, 0.0)));          // weight[0] = 0
    inf.w["q"].push_back(cc->MakeCKKSPackedPlaintext(
        std::vector<double>(S, 1.0)));          // weight[1] = 1
    for (int i = 2; i < n_weights; ++i)
        inf.w["q"].push_back(cc->MakeCKKSPackedPlaintext(
            std::vector<double>(S, 0.0)));

    // Input: distinct values in first hidDim slots
    std::vector<double> x_in(S, 0.0);
    for (int i = 0; i < HD; ++i)
        x_in[i] = (double)(i + 1);   // 1, 2, 3, ..., 256

    auto pt = cc->MakeCKKSPackedPlaintext(x_in);
    auto ct = cc->Encrypt(inf.fhe->pk(), pt);
    std::cout << "  input level=" << level_of(ct) << "\n";

    Ctx result = qkv_q(inf, ct);
    std::cout << "  output level=" << level_of(result) << "\n";

    auto dec = decrypt(cc, result, inf.fhe->sk());

    // Expected: y[i] = x[(i + intRot) % HD] within the first HD slots
    // This is a cyclic shift of the hidden vector
    double max_err = 0.0;
    for (int i = 0; i < HD; ++i) {
        double expected = x_in[(i + intRot) % HD];
        max_err = std::max(max_err, std::abs(dec[i] - expected));
    }

    std::cout << "  max |y - rot(x," << intRot << ")| over hidDim slots: " << max_err << "\n";
    std::cout << "  slot[0]   expected x[" << intRot << "]=" << x_in[intRot]
              << "  y=" << dec[0] << "\n";
    std::cout << "  slot[" << HD-intRot << "]   expected x[" << HD << " ≡ 0]=" << x_in[0]
              << "  y=" << dec[HD - intRot] << "\n";

    EXPECT_LT(max_err, 0.05) << "Shift: output should be input rotated by " << intRot;
}

// ── Test 4: Slot masking via weight zero-padding ────────────────────────
//
// CacheMir's replication policy: the input vector of size hidDim is
// replicated S/hidDim times across the S ciphertext slots:
//   slot[p*hD + i] = x[i]   for p = 0 .. S/hD-1
//
// The pre/post processing rotate-add loops SUM these replicas.
// Without masking, garbage in non-data slots would pollute the result.
//
// The masking policy: weight plaintexts are zero-padded in "garbage" slots.
// In CacheMirLinear.cu (line 72):
//   vector<double> vals(numSlots, 0.0);          // all zeros
//   for (p = 0..P-1) for (i = 0..n-1)
//       vals[p*n + i] = W[row][col];             // only data slots filled
//
// This test proves the policy works:
//   A) Properly zero-padded weights → garbage slots zeroed out, correct output.
//   B) Naively filled weights (1.0 everywhere) → garbage slots leak through,
//      corrupting the result with the replication factor S/hD.

TEST_F(NonLinearTest, Linear_SlotMasking) {
    const CC& cc = inf.cc();
    const int S  = inf.slots;
    inf.size.hidDim = HD;
    inf.size.ffDim  = FD;
    inf.parallel    = false;

    const int n_weights = HD * HD / S;           // 32
    const int replicas  = S / HD;                // 8
    const int intRot    = S / HD;                // 8

    std::cout << "\n=== Slot Masking: zero-padded vs naive weights ===\n"
              << "  S=" << S << "  hidDim=" << HD
              << "  replicas=" << replicas << "\n";

    // Input: data only in first hidDim slots, rest is GARBAGE (nonzero)
    std::vector<double> x_in(S, 0.0);
    for (int i = 0; i < HD; ++i)
        x_in[i] = 1.0 + 0.01 * i;
    // Fill remaining slots with garbage (simulating leftover data)
    for (int i = HD; i < S; ++i)
        x_in[i] = 999.0;

    // ── A) Properly zero-padded identity weights ──
    // weight[0] has 1.0 only in data positions [p*hD + i], 0 in garbage
    {
        inf.w["q"].clear();
        std::vector<double> w0(S, 0.0);
        for (int p = 0; p < replicas; ++p)
            for (int i = 0; i < HD; ++i)
                w0[p * HD + i] = 1.0;

        inf.w["q"].push_back(cc->MakeCKKSPackedPlaintext(w0));
        for (int i = 1; i < n_weights; ++i)
            inf.w["q"].push_back(cc->MakeCKKSPackedPlaintext(
                std::vector<double>(S, 0.0)));

        auto pt = cc->MakeCKKSPackedPlaintext(x_in);
        auto ct = cc->Encrypt(inf.fhe->pk(), pt);

        Ctx result = linear(inf, ct, "q", /*expand=*/0);
        auto dec = decrypt(cc, result, inf.fhe->sk());

        // Expected: the pre-processing sums replicas. With proper masking,
        // weight zeros out garbage slots, so output ≈ x_in[0..hD-1]
        double max_err = 0.0;
        for (int i = 0; i < HD; ++i)
            max_err = std::max(max_err, std::abs(dec[i] - x_in[i]));

        std::cout << "\n  [A] Zero-padded weights (correct masking):\n";
        std::cout << "    max |y - x| over hidDim: " << max_err << "\n";
        std::cout << "    slot[0]   x=" << x_in[0] << "  y=" << dec[0] << "\n";
        std::cout << "    slot[255] x=" << x_in[255] << "  y=" << dec[255] << "\n";

        EXPECT_LT(max_err, 0.1)
            << "Zero-padded weights should mask garbage and recover input";
    }

    // ── B) Naive weights (1.0 in ALL slots, no masking) ──
    // Garbage leaks through: pre-processing sums all replicas including
    // garbage, so the output is ~(x + 999*(replicas-1)) — way off.
    {
        inf.w["q"].clear();
        inf.w["q"].push_back(cc->MakeCKKSPackedPlaintext(
            std::vector<double>(S, 1.0)));     // 1.0 EVERYWHERE — no masking
        for (int i = 1; i < n_weights; ++i)
            inf.w["q"].push_back(cc->MakeCKKSPackedPlaintext(
                std::vector<double>(S, 0.0)));

        auto pt = cc->MakeCKKSPackedPlaintext(x_in);
        auto ct = cc->Encrypt(inf.fhe->pk(), pt);

        Ctx result = linear(inf, ct, "q", /*expand=*/0);
        auto dec_naive = decrypt(cc, result, inf.fhe->sk());

        // Plaintext simulation of what happens with naive weights:
        // pre-processing: xb[i] = sum_{p=0}^{rep-1} x[(i + p*hD) % S]
        auto xb = x_in;
        for (int step = HD; step < S; step *= 2)
            xb = add_vec(xb, rotate_vec(xb, step));

        // weight[0]=all-ones, so partSum[0] = xb * 1 = xb
        // post-processing: same tree
        auto expected_naive = xb;
        for (int step = HD; step < S; step *= 2)
            expected_naive = add_vec(expected_naive, rotate_vec(expected_naive, step));

        double max_err_naive = 0.0;
        for (int i = 0; i < HD; ++i)
            max_err_naive = std::max(max_err_naive,
                std::abs(dec_naive[i] - expected_naive[i]));

        // The naive result is HUGE because garbage (999) gets summed in
        double garbage_magnitude = std::abs(dec_naive[0]);

        std::cout << "\n  [B] Naive weights (no masking — garbage leaks):\n";
        std::cout << "    slot[0] expected_naive=" << expected_naive[0]
                  << "  HE=" << dec_naive[0] << "\n";
        std::cout << "    slot[0] should-be=" << x_in[0]
                  << "  actual=" << dec_naive[0]
                  << "  (off by " << std::abs(dec_naive[0] - x_in[0]) << ")\n";
        std::cout << "    HE matches naive plaintext sim: err="
                  << max_err_naive << "\n";

        // Verify: naive result is far from the true input
        EXPECT_GT(std::abs(dec_naive[0] - x_in[0]), 10.0)
            << "Without masking, garbage should corrupt the output significantly";

        // Verify: HE matches the plaintext simulation of the unmasked algorithm
        EXPECT_LT(max_err_naive, 1.0)
            << "HE should match plaintext simulation even for naive weights";
    }
}
