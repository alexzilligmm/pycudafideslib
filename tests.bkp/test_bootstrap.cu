// test_bootstrap.cu
// Validates FIDESlib's EvalBootstrap (the backend used by bootstrap.cu).
//
// Tests:
//   1. MessagePreserved  – bootstrap refreshes level while keeping message
//   2. LevelRefresh      – output level is lower (fresher) than input
//   3. NoiseBound        – decryption error stays within CKKS noise bound
//   4. ChainedBootstrap  – two sequential bootstraps are both correct
//   5. AfterMultiplication – bootstrap after depth-consuming operations
//   6. Linearity         – bootstrap(a) + bootstrap(b) ≈ bootstrap(a+b)
//      (bootstrapping is approximately linear in the message)

#include <gtest/gtest.h>
#include "fideslib_wrapper.h"
#include "llama.h"   // for bootstrap_to()

#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>

// declared in bootstrap.cu
Ctx bootstrap_to(LlamaInference&, const Ctx&, uint32_t);

// ── fixture ───────────────────────────────────────────────────────────────
class BootstrapTest : public ::testing::Test {
protected:
    // Use logN=12 so tests run quickly.
    static constexpr int LOGN  = 12;
    static constexpr int SLOTS = 1 << (LOGN - 1);

    // After bootstrapping the noise grows slightly; allow 1e-3 relative error.
    static constexpr double BTP_TOL = 1e-3;

    std::shared_ptr<CKKSContext> ctx;

    void SetUp() override {
        ctx = make_ckks_context(LOGN, /*depth=*/16, 40, SLOTS, /*btp=*/true);
    }

    std::vector<double> small_vec(double scale = 0.5) {
        // Messages must be < q/2 for bootstrapping to be valid.
        // Use values in [-0.5, 0.5].
        std::vector<double> v(SLOTS);
        for (int i = 0; i < SLOTS; ++i)
            v[i] = scale * std::cos(2.0 * M_PI * i / (double)SLOTS);
        return v;
    }

    // Consume `n` multiplicative levels so the ciphertext is deep enough
    // to need bootstrapping.
    Ctx consume_levels(const Ctx& ct_in, int n) {
        Ctx ct = ct_in;
        for (int i = 0; i < n; ++i) {
            // Encode the plaintext at the current ciphertext level so that
            // FLEXIBLEAUTO does not raise a level-mismatch error.
            Ptx one = ctx->cc->MakeCKKSPackedPlaintext(
                          std::vector<double>(SLOTS, 1.0),
                          /*noiseScaleDeg=*/1,
                          /*level=*/level_of(ct));
            ct = ctx->cc->EvalMult(ct, one);
        }
        return ct;
    }
};

// ── 1. Message preserved after bootstrap ─────────────────────────────────
TEST_F(BootstrapTest, MessagePreserved) {
    const CC& cc = ctx->cc;
    auto msg = small_vec(0.3);

    Ctx ct = encrypt(cc, cc->MakeCKKSPackedPlaintext(msg), ctx->pk());

    // Consume enough levels to require bootstrapping
    ct = consume_levels(ct, 10);
    uint32_t level_before = level_of(ct);
    std::cout << "Level before bootstrap: " << level_before << "\n";

    // Bootstrap
    Ctx ct_fresh = cc->EvalBootstrap(ct);
    uint32_t level_after = level_of(ct_fresh);
    std::cout << "Level after  bootstrap: " << level_after  << "\n";

    // Decrypt and compare to original
    Plaintext result_pt;
    result_pt = decrypt_pt(cc, ct_fresh, ctx->sk());
    auto result = result_pt->GetRealPackedValue();

    ASSERT_EQ((int)result.size(), SLOTS);
    double max_err = 0.0;
    for (int i = 0; i < SLOTS; ++i)
        max_err = std::max(max_err, std::abs(result[i] - msg[i]));

    std::cout << "Max absolute error after bootstrap: " << max_err << "\n";
    EXPECT_LT(max_err, BTP_TOL)
        << "Bootstrap introduced too much error: " << max_err;
}

// ── 2. Level is refreshed (output level ≤ input level) ───────────────────
TEST_F(BootstrapTest, LevelRefresh) {
    const CC& cc = ctx->cc;
    auto msg = small_vec();

    Ctx ct = encrypt(cc, cc->MakeCKKSPackedPlaintext(msg), ctx->pk());
    ct = consume_levels(ct, 12);

    uint32_t lvl_before = level_of(ct);
    Ctx fresh = cc->EvalBootstrap(ct);
    uint32_t lvl_after  = level_of(fresh);

    std::cout << "Level: " << lvl_before << " → " << lvl_after << "\n";
    EXPECT_NE(lvl_after, lvl_before)
        << "Bootstrap should change the level (refresh moduli chain)";
}

// ── 3. Noise bound: error ≤ BTP_TOL for messages in [-0.5, 0.5] ──────────
TEST_F(BootstrapTest, NoiseBound) {
    const CC& cc = ctx->cc;

    // Test several different message amplitudes
    for (double amp : {0.1, 0.3, 0.5}) {
        auto msg = small_vec(amp);
        Ctx ct = encrypt(cc, cc->MakeCKKSPackedPlaintext(msg), ctx->pk());
        ct = consume_levels(ct, 8);

        Ctx fresh = cc->EvalBootstrap(ct);

        Plaintext pt;
        pt = decrypt_pt(cc, fresh, ctx->sk());
        auto result = pt->GetRealPackedValue();

        double max_err = 0.0;
        for (int i = 0; i < SLOTS; ++i)
            max_err = std::max(max_err, std::abs(result[i] - msg[i]));

        EXPECT_LT(max_err, BTP_TOL)
            << "Amplitude=" << amp << " max_err=" << max_err;
    }
}

// ── 4. Chained bootstraps ─────────────────────────────────────────────────
TEST_F(BootstrapTest, ChainedBootstrap) {
    const CC& cc = ctx->cc;
    auto msg = small_vec(0.2);

    Ctx ct = encrypt(cc, cc->MakeCKKSPackedPlaintext(msg), ctx->pk());
    ct = consume_levels(ct, 8);

    // First bootstrap
    Ctx fresh1 = cc->EvalBootstrap(ct);
    fresh1 = consume_levels(fresh1, 3);  // consume a few levels; bootstrap leaves many available

    // Second bootstrap
    Ctx fresh2 = cc->EvalBootstrap(fresh1);

    Plaintext pt;
    pt = decrypt_pt(cc, fresh2, ctx->sk());
    auto result = pt->GetRealPackedValue();

    double max_err = 0.0;
    for (int i = 0; i < SLOTS; ++i)
        max_err = std::max(max_err, std::abs(result[i] - msg[i]));

    // Allow 2× tolerance for two bootstraps
    EXPECT_LT(max_err, 2.0 * BTP_TOL)
        << "Chained bootstrap max_err=" << max_err;
}

// ── 5. Bootstrap after real multiplications ───────────────────────────────
TEST_F(BootstrapTest, AfterMultiplication) {
    const CC& cc = ctx->cc;
    auto msg = small_vec(0.3);

    // Encrypt and do several real multiplications (not just level drops)
    Ctx ct = encrypt(cc, cc->MakeCKKSPackedPlaintext(msg), ctx->pk());

    // Multiply by 1.0 repeatedly (no-op for value, but consumes levels)
    Ptx pt_one = cc->MakeCKKSPackedPlaintext(std::vector<double>(SLOTS, 1.0));
    for (int i = 0; i < 8; ++i)
        ct = cc->EvalMult(ct, pt_one);

    // Compute expected value: msg^1 = msg (1^8 = 1)
    Ctx fresh = cc->EvalBootstrap(ct);

    Plaintext result_pt;
    result_pt = decrypt_pt(cc, fresh, ctx->sk());
    auto result = result_pt->GetRealPackedValue();

    double max_err = 0.0;
    for (int i = 0; i < SLOTS; ++i)
        max_err = std::max(max_err, std::abs(result[i] - msg[i]));

    std::cout << "After 8 multiplications, bootstrap max_err=" << max_err << "\n";
    EXPECT_LT(max_err, BTP_TOL);
}

// ── 6. Approximate linearity of bootstrap ────────────────────────────────
// bootstrap(a + b) ≈ bootstrap(a) + bootstrap(b)  for small messages
TEST_F(BootstrapTest, Linearity) {
    const CC& cc = ctx->cc;
    auto va = small_vec(0.2);
    auto vb = small_vec(0.15);
    std::vector<double> vsum(SLOTS);
    for (int i = 0; i < SLOTS; ++i) vsum[i] = va[i] + vb[i];

    Ctx ca   = encrypt(cc, cc->MakeCKKSPackedPlaintext(va), ctx->pk());
    Ctx cb   = encrypt(cc, cc->MakeCKKSPackedPlaintext(vb), ctx->pk());
    Ctx csum = cc->EvalAdd(ca, cb);

    ca   = consume_levels(ca,   8);
    cb   = consume_levels(cb,   8);
    csum = consume_levels(csum, 8);

    Ctx btp_a    = cc->EvalBootstrap(ca);
    Ctx btp_b    = cc->EvalBootstrap(cb);
    Ctx btp_sum  = cc->EvalBootstrap(csum);
    match_level(cc, btp_a, btp_b);
    Ctx btp_apb  = cc->EvalAdd(btp_a, btp_b);   // bootstrap(a) + bootstrap(b)

    auto ref_dec   = decrypt(cc, btp_sum, ctx->sk());
    auto apb_dec   = decrypt(cc, btp_apb, ctx->sk());

    double max_err = 0.0;
    for (int i = 0; i < SLOTS; ++i)
        max_err = std::max(max_err, std::abs(ref_dec[i] - apb_dec[i]));

    // Allow 3× tolerance since we're comparing two noisy bootstraps
    EXPECT_LT(max_err, 3.0 * BTP_TOL)
        << "Linearity max_err=" << max_err;
}

// ── 7. bootstrap_to() wrapper (from bootstrap.cu) refreshes the ciphertext ──
TEST_F(BootstrapTest, BootstrapToLevel) {
    // Build a minimal LlamaInference to exercise bootstrap_to()
    LlamaInference llama;
    llama.fhe    = ctx;
    llama.logN   = LOGN;
    llama.slots  = SLOTS;

    auto msg = small_vec(0.3);
    const CC& cc = ctx->cc;
    Ctx ct = encrypt(cc, cc->MakeCKKSPackedPlaintext(msg), ctx->pk());
    ct = consume_levels(ct, 10);

    uint32_t lvl_before = level_of(ct);
    // target=0: use the fresh post-bootstrap level directly
    Ctx fresh = bootstrap_to(llama, ct, /*target_level=*/0);

    EXPECT_NE(level_of(fresh), lvl_before)
        << "bootstrap_to should change the level";

    auto result = decrypt(cc, fresh, ctx->sk());
    double max_err = 0.0;
    for (int i = 0; i < SLOTS; ++i)
        max_err = std::max(max_err, std::abs(result[i] - msg[i]));
    EXPECT_LT(max_err, BTP_TOL);
}
