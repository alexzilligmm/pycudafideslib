// test_linear.cu — Validate CacheMIR interleaved linear (paper §3, Algorithm 1).
//
// Tests linear_interleaved() against a plaintext simulation of the same
// algorithm (BSGS interleaved packing with bench_mode rotations).

#include <gtest/gtest.h>
#include "fideslib_wrapper.h"
#include "ckks_primitives.h"
#include "gpt2.h"
#include <cmath>
#include <vector>
#include <random>
#include <iostream>

// ── Plaintext simulation helpers ────────────────────────────────────────

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

// Plaintext simulation of CacheMIR interleaved linear (bench_mode).
// All rotations use index 5 in bench_mode, but the plaintext sim uses
// the real indices to compute the correct mathematical result.
static std::vector<double> linear_interleaved_plain(
        const std::vector<double>& x_in,
        const std::vector<std::vector<double>>& weights,
        int d_in, int d_out, int S) {

    int n_pt = d_in * d_out / S;
    int intRot = S / d_in;
    int inRot  = (int)std::sqrt((double)n_pt / 2.0);
    if (inRot < 1) inRot = 1;
    while (n_pt % inRot != 0) ++inRot;
    int outRot = n_pt / inRot;

    // Step 1: Preprocess (input fold) — replicate d_in block across S slots
    auto xb = x_in;
    for (int step = d_in; step < S; step *= 2)
        xb = add_vec(xb, rotate_vec(xb, step));

    // Step 2: Baby-step rotations
    std::vector<std::vector<double>> ctRot(inRot);
    ctRot[0] = xb;
    for (int i = 1; i < inRot; ++i)
        ctRot[i] = rotate_vec(ctRot[i - 1], intRot);

    // Step 3: Multiply + accumulate (baby-step partial sums)
    std::vector<std::vector<double>> partSum(outRot);
    for (int j = 0; j < outRot; ++j) {
        partSum[j] = mul_vec(ctRot[0], weights[j * inRot]);
        for (int i = 1; i < inRot; ++i)
            partSum[j] = add_vec(partSum[j], mul_vec(ctRot[i], weights[j * inRot + i]));
    }

    // Step 4: Giant-step accumulation
    for (int j = 1; j < outRot; ++j)
        partSum[j] = rotate_vec(partSum[j], j * inRot * intRot);

    auto result = partSum[0];
    for (int j = 1; j < outRot; ++j)
        result = add_vec(result, partSum[j]);

    // Step 5: Postprocess (output unfold)
    for (int step = d_out; step < S; step *= 2)
        result = add_vec(result, rotate_vec(result, step));

    return result;
}

// ── Rotation indices for interleaved linear ─────────────────────────────

static std::vector<int32_t> collect_interleaved_rot_indices(int d_in, int d_out, int S) {
    std::vector<int32_t> rots;

    int n_pt = d_in * d_out / S;
    int intRot = S / d_in;
    int inRot  = (int)std::sqrt((double)n_pt / 2.0);
    if (inRot < 1) inRot = 1;
    while (n_pt % inRot != 0) ++inRot;
    int outRot = n_pt / inRot;

    // baby-step
    for (int i = 1; i < inRot; ++i) rots.push_back(i * intRot);
    // giant-step
    for (int j = 1; j < outRot; ++j) rots.push_back(j * inRot * intRot);
    // pre (input fold)
    for (int step = d_in; step < S; step *= 2) rots.push_back(step);
    // post (output unfold)
    for (int step = d_out; step < S; step *= 2) rots.push_back(step);

    std::sort(rots.begin(), rots.end());
    rots.erase(std::unique(rots.begin(), rots.end()), rots.end());
    return rots;
}

// ── Test fixture ────────────────────────────────────────────────────────

static constexpr int HD = 256;
static constexpr int FD = 1024;

class InterleavedLinearTest : public ::testing::Test {
protected:
    static constexpr uint32_t LOG_N = 16;

    static std::shared_ptr<CKKSContext> ctx;
    static Inference inf;

    static void SetUpTestSuite() {
        if (ctx) return;
        const int S = 1 << (LOG_N - 1);

        // Collect rotation indices for all three linear shapes
        std::vector<int32_t> rots;
        auto add_rots = [&](int d_in, int d_out) {
            auto r = collect_interleaved_rot_indices(d_in, d_out, S);
            rots.insert(rots.end(), r.begin(), r.end());
        };
        add_rots(HD, HD);   // QKV, Out
        add_rots(HD, FD);   // Up
        add_rots(FD, HD);   // Down

        std::sort(rots.begin(), rots.end());
        rots.erase(std::unique(rots.begin(), rots.end()), rots.end());

        ctx = make_ckks_context(
            LOG_N,
            /*depth=*/13,
            /*scale_bits=*/41,
            /*bootstrap_slots=*/0,
            /*enable_bootstrap=*/false,
            /*btp_scale_bits=*/41,
            /*first_mod_bits=*/53,
            /*level_budget_in=*/{},
            /*batch_size=*/0,
            /*h_weight=*/192,
            /*num_large_digits=*/3,
            /*btp_depth_overhead=*/0,
            /*extra_rot_steps=*/rots
        );

        inf.fhe    = ctx;
        inf.slots  = S;
        inf.logN   = LOG_N;
        inf.size.hidDim = HD;
        inf.size.ffDim  = FD;
        inf.parallel = false;
        inf.bench_mode = true;

        std::cout << "--- INTERLEAVED LINEAR TEST (bench_mode, once) ---\n"
                  << "N=" << (1 << LOG_N) << " S=" << S
                  << " hidDim=" << HD << " ffDim=" << FD
                  << " rot_keys=" << rots.size() << "\n";
    }
};

std::shared_ptr<CKKSContext> InterleavedLinearTest::ctx = nullptr;
Inference InterleavedLinearTest::inf = {};

// ── Tests ───────────────────────────────────────────────────────────────

TEST_F(InterleavedLinearTest, HidToHid) {
    const CC& cc = ctx->cc;
    const int S  = inf.slots;
    const int n_pt = HD * HD / S;

    std::cout << "\n=== Interleaved hid->hid: n_pt=" << n_pt << " ===\n";

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-0.1, 0.1);

    std::vector<double> x_plain(S);
    for (auto& v : x_plain) v = dist(rng);

    std::vector<std::vector<double>> w_plain(n_pt);
    inf.w["q"].clear();
    for (int i = 0; i < n_pt; ++i) {
        w_plain[i].resize(S);
        for (auto& v : w_plain[i]) v = dist(rng);
        inf.w["q"].push_back(cc->MakeCKKSPackedPlaintext(w_plain[i]));
    }

    auto expected = linear_interleaved_plain(x_plain, w_plain, HD, HD, S);

    auto pt = cc->MakeCKKSPackedPlaintext(x_plain);
    auto ct = cc->Encrypt(ctx->pk(), pt);
    std::cout << "  input level=" << level_of(ct) << "\n";

    Ctx result = linear_interleaved(inf, ct, "q", HD, HD);
    std::cout << "  output level=" << level_of(result) << "\n";

    auto dec = decrypt(cc, result, ctx->sk());

    double max_err = 0.0;
    for (int i = 0; i < HD; ++i)
        max_err = std::max(max_err, std::abs(dec[i] - expected[i]));

    std::cout << "  max |HE - plain| first " << HD << " slots: " << max_err << "\n";
    for (int i : {0, 1, HD/2, HD-1})
        std::cout << "  slot[" << i << "]  plain=" << expected[i]
                  << "  HE=" << dec[i] << "\n";

    EXPECT_LT(max_err, 0.5);
}

TEST_F(InterleavedLinearTest, HidToFf) {
    const CC& cc = ctx->cc;
    const int S  = inf.slots;
    const int n_pt = HD * FD / S;

    std::cout << "\n=== Interleaved hid->ff: n_pt=" << n_pt << " ===\n";

    std::mt19937 rng(99);
    std::uniform_real_distribution<double> dist(-0.1, 0.1);

    std::vector<double> x_plain(S);
    for (auto& v : x_plain) v = dist(rng);

    std::vector<std::vector<double>> w_plain(n_pt);
    inf.w["up"].clear();
    for (int i = 0; i < n_pt; ++i) {
        w_plain[i].resize(S);
        for (auto& v : w_plain[i]) v = dist(rng);
        inf.w["up"].push_back(cc->MakeCKKSPackedPlaintext(w_plain[i]));
    }

    auto expected = linear_interleaved_plain(x_plain, w_plain, HD, FD, S);

    auto pt = cc->MakeCKKSPackedPlaintext(x_plain);
    auto ct = cc->Encrypt(ctx->pk(), pt);

    Ctx result = linear_interleaved(inf, ct, "up", HD, FD);

    auto dec = decrypt(cc, result, ctx->sk());

    double max_err = 0.0;
    for (int i = 0; i < FD; ++i)
        max_err = std::max(max_err, std::abs(dec[i] - expected[i]));

    std::cout << "  max |HE - plain| first " << FD << " slots: " << max_err << "\n";
    EXPECT_LT(max_err, 0.5);
}

TEST_F(InterleavedLinearTest, FfToHid) {
    const CC& cc = ctx->cc;
    const int S  = inf.slots;
    const int n_pt = FD * HD / S;

    std::cout << "\n=== Interleaved ff->hid: n_pt=" << n_pt << " ===\n";

    std::mt19937 rng(77);
    std::uniform_real_distribution<double> dist(-0.1, 0.1);

    std::vector<double> x_plain(S);
    for (auto& v : x_plain) v = dist(rng);

    std::vector<std::vector<double>> w_plain(n_pt);
    inf.w["down"].clear();
    for (int i = 0; i < n_pt; ++i) {
        w_plain[i].resize(S);
        for (auto& v : w_plain[i]) v = dist(rng);
        inf.w["down"].push_back(cc->MakeCKKSPackedPlaintext(w_plain[i]));
    }

    auto expected = linear_interleaved_plain(x_plain, w_plain, FD, HD, S);

    auto pt = cc->MakeCKKSPackedPlaintext(x_plain);
    auto ct = cc->Encrypt(ctx->pk(), pt);

    Ctx result = linear_interleaved(inf, ct, "down", FD, HD);

    auto dec = decrypt(cc, result, ctx->sk());

    double max_err = 0.0;
    for (int i = 0; i < HD; ++i)
        max_err = std::max(max_err, std::abs(dec[i] - expected[i]));

    std::cout << "  max |HE - plain| first " << HD << " slots: " << max_err << "\n";
    EXPECT_LT(max_err, 0.5);
}
