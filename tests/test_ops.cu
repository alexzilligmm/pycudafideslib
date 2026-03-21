// test_ops.cu
// Tests for basic FHE operations: Add, Sub, Mul, Rotate, Rescale, LevelReduce.
// Each test encrypts known values, applies the operation, decrypts, and
// compares against the plaintext reference within the expected CKKS noise bound.

#include <gtest/gtest.h>
#include "fideslib_wrapper.h"

#include <cmath>
#include <numeric>
#include <algorithm>

// ── shared fixture ────────────────────────────────────────────────────────
class OpsTest : public ::testing::Test {
protected:
    static constexpr int LOGN   = 12;
    static constexpr int DEPTH  = 8;
    static constexpr int SLOTS  = 1 << (LOGN - 1);   // 2048

    // Noise tolerance: CKKS error ≤ 2^{-precision} * ||m||
    // For scale 2^40 and depth 8 a tolerance of 1e-5 is conservative.
    static constexpr double TOL = 1e-5;

    std::shared_ptr<CKKSContext> ctx;

    void SetUp() override {
        ctx = make_ckks_context(LOGN, DEPTH, 40, SLOTS, /*btp=*/false);
    }

    // Encrypt a vector, run f on the ciphertext, decrypt, compare to ref.
    template<typename F, typename G>
    void check(const std::vector<double>& input,
               F fhe_op, G ref_op,
               double tol = TOL, const std::string& label = "") {
        Ptx pt = encode(ctx->cc, input);
        Ctx ct = encrypt(ctx->cc, pt, ctx->pk());

        Ctx ct_out  = fhe_op(ct);
        auto result = decrypt(ctx->cc, ct_out, ctx->sk());
        auto ref    = ref_op(input);

        ASSERT_EQ(result.size(), ref.size()) << label;
        double max_err = 0.0;
        for (size_t i = 0; i < ref.size(); ++i)
            max_err = std::max(max_err, std::abs(result[i] - ref[i]));
        EXPECT_LE(max_err, tol) << label << ": max_err=" << max_err;
    }

    std::vector<double> iota_vec(double start = 0.1, double step = 0.01) {
        std::vector<double> v(SLOTS);
        for (int i = 0; i < SLOTS; ++i) v[i] = start + i * step;
        return v;
    }

    std::vector<double> const_vec(double val) {
        return std::vector<double>(SLOTS, val);
    }
};

// ── ct + ct ───────────────────────────────────────────────────────────────
TEST_F(OpsTest, CtCtAdd) {
    const CC& cc = ctx->cc;
    auto v1 = iota_vec(0.1, 0.001);
    auto v2 = iota_vec(1.0, 0.002);

    Ctx ct1 = encrypt(cc, encode(cc, v1), ctx->pk());
    Ctx ct2 = encrypt(cc, encode(cc, v2), ctx->pk());
    Ctx ct_sum = cc->EvalAdd(ct1, ct2);

    auto result = decrypt(cc, ct_sum, ctx->sk());
    for (int i = 0; i < SLOTS; ++i)
        EXPECT_NEAR(result[i], v1[i] + v2[i], TOL)
            << "at slot " << i;
}

// ── ct - ct ───────────────────────────────────────────────────────────────
TEST_F(OpsTest, CtCtSub) {
    const CC& cc = ctx->cc;
    auto v1 = const_vec(3.0);
    auto v2 = const_vec(1.5);

    Ctx ct1 = encrypt(cc, encode(cc, v1), ctx->pk());
    Ctx ct2 = encrypt(cc, encode(cc, v2), ctx->pk());
    Ctx ct_diff = cc->EvalSub(ct1, ct2);

    auto result = decrypt(cc, ct_diff, ctx->sk());
    for (int i = 0; i < SLOTS; ++i)
        EXPECT_NEAR(result[i], 1.5, TOL) << "at slot " << i;
}

// ── ct * ct (with relinearization) ───────────────────────────────────────
TEST_F(OpsTest, CtCtMul) {
    const CC& cc = ctx->cc;
    auto v = iota_vec(0.5, 0.01);

    Ctx ct = encrypt(cc, encode(cc, v), ctx->pk());
    Ctx ct2 = cc->EvalMult(ct, ct);  // includes relinearize + rescale

    auto result = decrypt(cc, ct2, ctx->sk());
    for (int i = 0; i < SLOTS; ++i)
        EXPECT_NEAR(result[i], v[i] * v[i], TOL * 100)
            << "at slot " << i;  // higher tolerance after mul
}

// ── ct * plaintext ────────────────────────────────────────────────────────
TEST_F(OpsTest, CtPtMul) {
    const CC& cc = ctx->cc;
    auto v = iota_vec(1.0, 0.01);
    auto w = const_vec(2.5);

    Ctx ct  = encrypt(cc, encode(cc, v), ctx->pk());
    Ptx pt  = encode(cc, w);
    Ctx out = cc->EvalMult(ct, pt);

    auto result = decrypt(cc, out, ctx->sk());
    for (int i = 0; i < SLOTS; ++i)
        EXPECT_NEAR(result[i], v[i] * 2.5, TOL) << "at slot " << i;
}

// ── ct + scalar ───────────────────────────────────────────────────────────
TEST_F(OpsTest, CtAddScalar) {
    const CC& cc = ctx->cc;
    auto v = const_vec(1.0);

    Ctx ct  = encrypt(cc, encode(cc, v), ctx->pk());
    Ctx out = cc->EvalAdd(ct, cc->MakeCKKSPackedPlaintext(const_vec(3.0)));

    auto result = decrypt(cc, out, ctx->sk());
    for (int i = 0; i < SLOTS; ++i)
        EXPECT_NEAR(result[i], 4.0, TOL) << "at slot " << i;
}

// ── rotation ──────────────────────────────────────────────────────────────
TEST_F(OpsTest, Rotate) {
    const CC& cc = ctx->cc;
    std::vector<double> v(SLOTS, 0.0);
    v[0] = 1.0;  // canonical basis vector: only slot 0 is 1

    Ctx ct  = encrypt(cc, encode(cc, v), ctx->pk());
    Ctx rot = cc->EvalRotate(ct, 1);  // left-rotate by 1

    auto result = decrypt(cc, rot, ctx->sk());

    // After rotating left by 1: slot 0 should be ~0, slot SLOTS-1 should be ~1
    EXPECT_NEAR(result[0],         0.0, TOL) << "slot 0 after rotate";
    EXPECT_NEAR(result[SLOTS - 1], 1.0, TOL) << "slot N-1 after rotate";
}

TEST_F(OpsTest, RotateAndAdd) {
    const CC& cc = ctx->cc;
    // v = [1, 0, 0, ...]; rotate by 1 and add → [1, 1, 0, ...] approximately
    std::vector<double> v(SLOTS, 0.0);
    v[0] = 1.0;

    Ctx ct  = encrypt(cc, encode(cc, v), ctx->pk());
    Ctx rot = cc->EvalRotate(ct, 1);
    match_level(cc, ct, rot);
    Ctx out = cc->EvalAdd(ct, rot);

    auto result = decrypt(cc, out, ctx->sk());
    EXPECT_NEAR(result[0],         1.0, TOL);
    EXPECT_NEAR(result[SLOTS - 1], 1.0, TOL);
}

// ── level reduce ─────────────────────────────────────────────────────────
TEST_F(OpsTest, LevelReduce) {
    const CC& cc = ctx->cc;
    auto v = const_vec(42.0);

    Ctx ct = encrypt(cc, encode(cc, v), ctx->pk());
    EXPECT_EQ(level_of(ct), 0u);

    drop_levels(cc, ct, 2);
    EXPECT_EQ(level_of(ct), 2u);

    auto result = decrypt(cc, ct, ctx->sk());
    for (int i = 0; i < SLOTS; ++i)
        EXPECT_NEAR(result[i], 42.0, TOL * 10) << "slot " << i;
}

// ── chained multiplications (depth test) ─────────────────────────────────
TEST_F(OpsTest, ChainedMul) {
    const CC& cc = ctx->cc;
    auto v = const_vec(2.0);

    Ctx ct = encrypt(cc, encode(cc, v), ctx->pk());
    // 2^8 = 256 via 8 squarings
    for (int i = 0; i < 8; ++i)
        ct = cc->EvalSquare(ct);

    auto result = decrypt(cc, ct, ctx->sk());
    // Allow larger tolerance due to accumulated CKKS error
    for (int i = 0; i < SLOTS; ++i)
        EXPECT_NEAR(result[i], 256.0, 1.0) << "slot " << i;
}

// ── EvalSquare matches EvalMult(ct, ct) ───────────────────────────────────
TEST_F(OpsTest, SquareVsMul) {
    const CC& cc = ctx->cc;
    auto v = iota_vec(0.1, 0.001);

    Ctx ct1 = encrypt(cc, encode(cc, v), ctx->pk());
    Ctx ct2 = encrypt(cc, encode(cc, v), ctx->pk());

    Ctx sq  = cc->EvalSquare(ct1);
    Ctx mul = cc->EvalMult(ct1, ct2);

    auto r_sq  = decrypt(cc, sq,  ctx->sk());
    auto r_mul = decrypt(cc, mul, ctx->sk());

    for (int i = 0; i < SLOTS; ++i)
        EXPECT_NEAR(r_sq[i], r_mul[i], TOL * 100)
            << "square vs mul at slot " << i;
}
