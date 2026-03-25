#include <gtest/gtest.h>
#include "fideslib_wrapper.h"
#include "ckks_primitives.h"
#include "llama.h"
#include <cmath>
#include <vector>
#include <random>
#include <iostream>

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

// Plaintext replica of linear() — same algorithm, operates on double vectors.
static std::vector<double> linear_plain(
        const std::vector<double>& x_in,
        const std::vector<std::vector<double>>& weights,
        int hidDim, int ffDim, int S, int expand) {

    int preStep  = (expand >= 0) ? hidDim : ffDim;
    int postStep = (expand <= 0) ? hidDim : ffDim;

    int inRot, outRot;
    if (expand == 0) {
        inRot  = (int)std::sqrt((double)(hidDim * hidDim) / (2 * S));
        outRot = hidDim * hidDim / (S * inRot);
    } else {
        inRot  = (int)std::sqrt((double)(hidDim * ffDim) / (2 * S));
        outRot = hidDim * ffDim / (S * inRot);
    }
    int intRot = S / hidDim;
    int n_weights = (int)weights.size();

    auto xb = x_in;
    for (int step = preStep; step < S; step *= 2)
        xb = add_vec(xb, rotate_vec(xb, step));

    std::vector<std::vector<double>> ctRot(inRot);
    ctRot[0] = xb;
    for (int i = 1; i < inRot; ++i)
        ctRot[i] = rotate_vec(ctRot[i - 1], intRot);

    std::vector<std::vector<double>> partSum(n_weights);
    for (int i = 0; i < n_weights; ++i)
        partSum[i] = mul_vec(ctRot[i % inRot], weights[i]);

    for (int i = 0; i < n_weights; ++i) {
        if (i % inRot > 0)
            partSum[i - i % inRot] = add_vec(partSum[i - i % inRot], partSum[i]);
    }

    for (int i = 1; i < outRot; ++i)
        partSum[i * inRot] = rotate_vec(partSum[i * inRot], i * inRot * intRot);

    auto result = partSum[0];
    for (int i = 1; i < outRot; ++i)
        result = add_vec(result, partSum[i * inRot]);

    for (int step = postStep; step < S; step *= 2)
        result = add_vec(result, rotate_vec(result, step));

    return result;
}

// ── Test fixture with rotation keys for linear layer ─────────────────────
// The default context only has power-of-2 rotation keys.
// linear() needs giant-step rotations at i*inRot*intRot which may not be
// powers of 2 (e.g. 96, 160, 192, 224 for hidDim=256, S=2048).
// We compute and register all needed keys.

static constexpr int HD = 256;
static constexpr int FD = 1024;

class LinearTest : public ::testing::Test {
protected:
    static constexpr uint32_t LOG_N = 12;

    std::shared_ptr<CKKSContext> ctx;
    Inference inf;

    void SetUp() override {
        const int S = 1 << (LOG_N - 1);

        // Collect all rotation indices needed by linear() for our dimensions
        int intRot = S / HD;
        auto add_linear_rots = [&](std::vector<int32_t>& rots, int hD, int fD, int expand) {
            int inR, outR;
            if (expand == 0) {
                inR  = (int)std::sqrt((double)(hD * hD) / (2 * S));
                outR = hD * hD / (S * inR);
            } else {
                inR  = (int)std::sqrt((double)(hD * fD) / (2 * S));
                outR = hD * fD / (S * inR);
            }
            int iR = S / hD;
            // baby-step: multiples of intRot
            for (int i = 1; i < inR; ++i) rots.push_back(iR);
            // giant-step: i * inRot * intRot
            for (int i = 1; i < outR; ++i) rots.push_back(i * inR * iR);
            // pre/post: powers of 2 from preStep
            int preStep  = (expand >= 0) ? hD : fD;
            int postStep = (expand <= 0) ? hD : fD;
            for (int step = preStep; step < S; step *= 2) rots.push_back(step);
            for (int step = postStep; step < S; step *= 2) rots.push_back(step);
        };

        std::vector<int32_t> extra_rots;
        add_linear_rots(extra_rots, HD, FD, 0);   // hid->hid
        add_linear_rots(extra_rots, HD, FD, 1);   // hid->ff
        add_linear_rots(extra_rots, HD, FD, -1);  // ff->hid

        // Deduplicate (make_ckks_context adds power-of-2 keys, we add extras)
        // We pass a custom rotation list via the wrapper
        // But make_ckks_context hardcodes its own rotation keys.
        // So we need to use the raw OpenFHE API to add more keys.
        // Simplest: create context, then add extra rotation keys.

        ctx = make_ckks_context(
            LOG_N,
            /*depth=*/16,
            /*scale_bits=*/40,
            /*bootstrap_slots=*/0,
            /*enable_bootstrap=*/false,  // skip bootstrap to keep context small
            /*btp_scale_bits=*/59,
            /*first_mod_bits=*/60,
            /*level_budget_in=*/{3, 3},
            /*batch_size=*/0,
            /*h_weight=*/192,
            /*num_large_digits=*/3,
            /*btp_depth_overhead=*/0
        );

        // Add extra rotation keys for linear layer
        // make_ckks_context already generated power-of-2 keys.
        // We need to add the non-power-of-2 giant-step keys.
        std::sort(extra_rots.begin(), extra_rots.end());
        extra_rots.erase(std::unique(extra_rots.begin(), extra_rots.end()), extra_rots.end());
        // Filter to keys not already generated (non-power-of-2)
        std::vector<int32_t> new_rots;
        for (auto r : extra_rots) {
            if (r > 0 && (r & (r - 1)) != 0) // not a power of 2
                new_rots.push_back(r);
        }
        if (!new_rots.empty()) {
            ctx->cc->EvalRotateKeyGen(ctx->keys.secretKey, new_rots);
            ctx->cc->LoadContext(ctx->keys.publicKey);
        }

        inf.fhe    = ctx;
        inf.slots  = S;
        inf.logN   = LOG_N;
        inf.size.hidDim = HD;
        inf.size.ffDim  = FD;
        inf.parallel = false;

        std::cout << "--- LINEAR TEST CONTEXT ---\n"
                  << "N=" << (1 << LOG_N) << " S=" << S
                  << " hidDim=" << HD << " ffDim=" << FD
                  << " extra_rot_keys=" << new_rots.size() << "\n";
    }
};

// ── Test 1: HE linear matches plaintext simulation (hid->hid) ──────────

TEST_F(LinearTest, HidToHid_PlaintextMatch) {
    const CC& cc = ctx->cc;
    const int S  = inf.slots;
    const int n_weights = HD * HD / S;

    int inRot  = (int)std::sqrt((double)(HD * HD) / (2 * S));
    int outRot = HD * HD / (S * inRot);
    int intRot = S / HD;

    std::cout << "\n=== Linear hid->hid: n_weights=" << n_weights
              << " inRot=" << inRot << " outRot=" << outRot
              << " intRot=" << intRot << " ===\n";

    // Deterministic small-magnitude input and weights
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-0.1, 0.1);

    std::vector<double> x_plain(S);
    for (auto& v : x_plain) v = dist(rng);

    std::vector<std::vector<double>> w_plain(n_weights);
    inf.w["q"].clear();
    for (int i = 0; i < n_weights; ++i) {
        w_plain[i].resize(S);
        for (auto& v : w_plain[i]) v = dist(rng);
        inf.w["q"].push_back(cc->MakeCKKSPackedPlaintext(w_plain[i]));
    }

    // Plaintext reference
    auto expected = linear_plain(x_plain, w_plain, HD, FD, S, 0);

    // HE
    auto pt = cc->MakeCKKSPackedPlaintext(x_plain);
    auto ct = cc->Encrypt(ctx->pk(), pt);
    std::cout << "  input level=" << level_of(ct) << "\n";

    Ctx result = linear(inf, ct, "q", 0);
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

// ── Test 2: Slot masking (zero-padded vs naive weights) ─────────────────

TEST_F(LinearTest, SlotMasking) {
    const CC& cc = ctx->cc;
    const int S  = inf.slots;
    const int n_weights = HD * HD / S;
    const int replicas  = S / HD;

    std::cout << "\n=== Slot Masking: zero-padded vs naive ===\n"
              << "  S=" << S << " hidDim=" << HD << " replicas=" << replicas << "\n";

    // Input: data in first hidDim slots, garbage elsewhere
    std::vector<double> x_in(S, 0.0);
    for (int i = 0; i < HD; ++i)
        x_in[i] = 1.0 + 0.01 * i;
    for (int i = HD; i < S; ++i)
        x_in[i] = 999.0;

    // ── A) Properly zero-padded weight[0] ──
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

        // Plaintext simulation with zero-padded weights
        std::vector<std::vector<double>> w_plain(n_weights, std::vector<double>(S, 0.0));
        w_plain[0] = w0;
        auto expected = linear_plain(x_in, w_plain, HD, FD, S, 0);

        auto pt = cc->MakeCKKSPackedPlaintext(x_in);
        auto ct = cc->Encrypt(ctx->pk(), pt);
        Ctx result = linear(inf, ct, "q", 0);
        auto dec = decrypt(cc, result, ctx->sk());

        double max_err = 0.0;
        for (int i = 0; i < HD; ++i)
            max_err = std::max(max_err, std::abs(dec[i] - expected[i]));

        std::cout << "\n  [A] Zero-padded weights:\n";
        std::cout << "    max |HE - plain_sim|: " << max_err << "\n";
        std::cout << "    slot[0] plain=" << expected[0] << " HE=" << dec[0] << "\n";
        EXPECT_LT(max_err, 0.5);
    }

    // ── B) Naive weight[0] = 1 everywhere (no masking) ──
    {
        inf.w["q"].clear();
        inf.w["q"].push_back(cc->MakeCKKSPackedPlaintext(
            std::vector<double>(S, 1.0)));
        for (int i = 1; i < n_weights; ++i)
            inf.w["q"].push_back(cc->MakeCKKSPackedPlaintext(
                std::vector<double>(S, 0.0)));

        // Plaintext simulation with naive weights
        std::vector<std::vector<double>> w_naive(n_weights, std::vector<double>(S, 0.0));
        w_naive[0] = std::vector<double>(S, 1.0);
        auto expected_naive = linear_plain(x_in, w_naive, HD, FD, S, 0);

        auto pt = cc->MakeCKKSPackedPlaintext(x_in);
        auto ct = cc->Encrypt(ctx->pk(), pt);
        Ctx result = linear(inf, ct, "q", 0);
        auto dec_naive = decrypt(cc, result, ctx->sk());

        double max_err_naive = 0.0;
        for (int i = 0; i < HD; ++i)
            max_err_naive = std::max(max_err_naive,
                std::abs(dec_naive[i] - expected_naive[i]));

        std::cout << "\n  [B] Naive weights (garbage leaks):\n";
        std::cout << "    max |HE - plain_sim|: " << max_err_naive << "\n";
        std::cout << "    slot[0] plain_sim=" << expected_naive[0]
                  << " HE=" << dec_naive[0]
                  << " true_x=" << x_in[0] << "\n";
        std::cout << "    corruption: |result - true| = "
                  << std::abs(dec_naive[0] - x_in[0]) << "\n";

        // HE matches naive plaintext sim
        EXPECT_LT(max_err_naive, 1.0);
        // But naive result is far from true input (garbage leaked through)
        EXPECT_GT(std::abs(expected_naive[0] - x_in[0]), 10.0)
            << "Without masking, garbage should corrupt the plaintext result";
    }
}
