#include <gtest/gtest.h>
#include "test_basics.h"
#include "ckks_primitives.h"
#include <cmath>

// ---------------------------------------------------------------------------
// Depth-budget regression tests
//
// Each test asserts the *theoretical optimal* multiplicative depth for the
// primitive under test.  Depths are derived from a pencil-and-paper analysis
// of the minimum number of sequential multiplications on the critical path,
// NOT from observing the implementation.
//
// FLEXIBLEAUTO behaviour:  OpenFHE's FLEXIBLEAUTO rescaling mode defers the
// last rescale until the next operation that needs it.  Because level_of()
// reads the current tower index, the measured consumption can therefore be
// one level *less* than the theoretical depth when the final rescale is still
// pending.  Every EXPECT below therefore accepts:
//
//     consumed ∈ { kDepth − 1,  kDepth }
//
// where kDepth is the theoretical optimum.
// ---------------------------------------------------------------------------

static Ctx fresh_ct(const CC& cc, double val, const PublicKey<DCRTPoly>& pk) {
    size_t slots = cc->GetRingDimension() / 2;
    auto pt = encode(cc, std::vector<double>(slots, val));
    return encrypt(cc, pt, pk);
}

// ── inv_sqrt_newton ─────────────────────────────────────────────────────────
// y_{i+1} = y_i · (1.5 − 0.5·x·y_i²)
// Critical path per iteration:
//   y² (1 square) → c·y² (1 ct×ct) → y·h (1 ct×ct)  =  3 mults
// Total for n iterations: 3n levels.

TEST_F(OpsTest, DepthBudget_InvSqrtNewton) {
    const CC& cc = ctx->cc;

    Ctx x    = fresh_ct(cc, 0.5, ctx->pk());
    Ctx ans  = fresh_ct(cc, 1.0, ctx->pk());

    uint32_t lvl_before = level_of(ans);
    Ctx result = inv_sqrt_newton(cc, x, ans, /*iters=*/1);
    uint32_t lvl_after = level_of(result);

    uint32_t consumed = lvl_after - lvl_before;
    std::cout << "inv_sqrt_newton (1 iter): consumed " << consumed << " levels"
              << " (before=" << lvl_before << " after=" << lvl_after << ")\n";

    constexpr uint32_t kDepth = 3;  // 3 mults × 1 iter
    EXPECT_GE(consumed, kDepth - 1);
    EXPECT_LE(consumed, kDepth);
}

TEST_F(OpsTest, DepthBudget_InvSqrtNewton_3Iters) {
    const CC& cc = ctx->cc;

    Ctx x    = fresh_ct(cc, 0.5, ctx->pk());
    Ctx ans  = fresh_ct(cc, 1.0, ctx->pk());

    uint32_t lvl_before = level_of(ans);
    Ctx result = inv_sqrt_newton(cc, x, ans, /*iters=*/3);
    uint32_t lvl_after = level_of(result);

    uint32_t consumed = lvl_after - lvl_before;
    std::cout << "inv_sqrt_newton (3 iters): consumed " << consumed << " levels"
              << " (before=" << lvl_before << " after=" << lvl_after << ")\n";

    constexpr uint32_t kDepth = 9;  // 3 mults × 3 iters
    EXPECT_GE(consumed, kDepth - 1);
    EXPECT_LE(consumed, kDepth);
}

// ── newton_inverse ──────────────────────────────────────────────────────────
// y_{i+1} = y_i · (2 − d·y_i)
// Critical path per iteration:
//   y·d (1 ct×ct) → y·(2−y·d) (1 ct×ct)  =  2 mults
// Total for n iterations: 2n levels.

TEST_F(OpsTest, DepthBudget_NewtonInverse) {
    const CC& cc = ctx->cc;

    Ctx dnm = fresh_ct(cc, 0.5, ctx->pk());
    Ctx res = fresh_ct(cc, 1.5, ctx->pk());

    uint32_t lvl_before = level_of(res);
    Ctx result = newton_inverse(cc, res, dnm, /*iters=*/1);
    uint32_t lvl_after = level_of(result);

    uint32_t consumed = lvl_after - lvl_before;
    std::cout << "newton_inverse (1 iter): consumed " << consumed << " levels"
              << " (before=" << lvl_before << " after=" << lvl_after << ")\n";

    constexpr uint32_t kDepth = 2;  // 2 mults × 1 iter
    EXPECT_GE(consumed, kDepth - 1);
    EXPECT_LE(consumed, kDepth);
}

TEST_F(OpsTest, DepthBudget_NewtonInverse_3Iters) {
    const CC& cc = ctx->cc;

    Ctx dnm = fresh_ct(cc, 0.5, ctx->pk());
    Ctx res = fresh_ct(cc, 1.5, ctx->pk());

    uint32_t lvl_before = level_of(res);
    Ctx result = newton_inverse(cc, res, dnm, /*iters=*/3);
    uint32_t lvl_after = level_of(result);

    uint32_t consumed = lvl_after - lvl_before;
    std::cout << "newton_inverse (3 iters): consumed " << consumed << " levels"
              << " (before=" << lvl_before << " after=" << lvl_after << ")\n";

    constexpr uint32_t kDepth = 6;  // 2 mults × 3 iters
    EXPECT_GE(consumed, kDepth - 1);
    EXPECT_LE(consumed, kDepth);
}

// ── goldschmidt_inv ─────────────────────────────────────────────────────────
// Init:  E = 1 − a·x₀                           (1 ct×ct mult)
// Iter:  x₀ = x₀·(1+E)  ‖  E = E²              (1 mult each, parallel)
// x₀ path: init(1) + 1 per iter  =  n+1 levels for n iterations.
// E² is independent and doesn't add to x₀'s critical path.

TEST_F(OpsTest, DepthBudget_GoldschmidtInv) {
    const CC& cc = ctx->cc;

    Ctx a  = fresh_ct(cc, 0.5, ctx->pk());
    Ctx x0 = fresh_ct(cc, 1.5, ctx->pk());

    uint32_t lvl_before = level_of(x0);
    Ctx result = goldschmidt_inv(cc, a, x0, /*iters=*/1);
    uint32_t lvl_after = level_of(result);

    uint32_t consumed = lvl_after - lvl_before;
    std::cout << "goldschmidt_inv (1 iter): consumed " << consumed << " levels"
              << " (before=" << lvl_before << " after=" << lvl_after << ")\n";

    constexpr uint32_t kDepth = 2;  // 1 (init) + 1 (1 iter)
    EXPECT_GE(consumed, kDepth - 1);
    EXPECT_LE(consumed, kDepth);
}

// ── exp_squaring ────────────────────────────────────────────────────────────
// Each iteration: 1 square  =  1 level.
// Total for n iterations: n levels.

TEST_F(OpsTest, DepthBudget_ExpSquaring) {
    const CC& cc = ctx->cc;

    Ctx x = fresh_ct(cc, 0.9, ctx->pk());

    uint32_t lvl_before = level_of(x);
    Ctx result = exp_squaring(cc, x, /*iters=*/1);
    uint32_t lvl_after = level_of(result);

    uint32_t consumed = lvl_after - lvl_before;
    std::cout << "exp_squaring (1 iter): consumed " << consumed << " levels"
              << " (before=" << lvl_before << " after=" << lvl_after << ")\n";

    constexpr uint32_t kDepth = 1;  // 1 square × 1 iter
    EXPECT_GE(consumed, kDepth - 1);
    EXPECT_LE(consumed, kDepth);
}

TEST_F(OpsTest, DepthBudget_ExpSquaring_5Iters) {
    const CC& cc = ctx->cc;

    Ctx x = fresh_ct(cc, 0.99, ctx->pk());

    uint32_t lvl_before = level_of(x);
    Ctx result = exp_squaring(cc, x, /*iters=*/5);
    uint32_t lvl_after = level_of(result);

    uint32_t consumed = lvl_after - lvl_before;
    std::cout << "exp_squaring (5 iters): consumed " << consumed << " levels"
              << " (before=" << lvl_before << " after=" << lvl_after << ")\n";

    constexpr uint32_t kDepth = 5;  // 1 square × 5 iters
    EXPECT_GE(consumed, kDepth - 1);
    EXPECT_LE(consumed, kDepth);
}

// ── eval_linear_wsum ────────────────────────────────────────────────────────
// All ct×scalar multiplications are independent (parallel).
// Depth = 1 (one scalar mult per ciphertext, all at the same depth).

TEST_F(OpsTest, DepthBudget_EvalLinearWsum) {
    const CC& cc = ctx->cc;

    Ctx ct1 = fresh_ct(cc, 1.0, ctx->pk());
    Ctx ct2 = fresh_ct(cc, 2.0, ctx->pk());
    std::vector<Ctx> cts = {ct1, ct2};
    std::vector<double> weights = {0.5, 0.3};

    uint32_t lvl_before = level_of(ct1);
    Ctx result = eval_linear_wsum(cc, cts, weights);
    uint32_t lvl_after = level_of(result);

    uint32_t consumed = lvl_after - lvl_before;
    std::cout << "eval_linear_wsum: consumed " << consumed << " levels"
              << " (before=" << lvl_before << " after=" << lvl_after << ")\n";

    constexpr uint32_t kDepth = 1;  // 1 parallel scalar mult
    EXPECT_GE(consumed, kDepth - 1);
    EXPECT_LE(consumed, kDepth);
}

// ── compute_average ─────────────────────────────────────────────────────────
// Only rotations + additions.  Zero multiplicative depth.

TEST_F(OpsTest, DepthBudget_ComputeAverage) {
    const CC& cc = ctx->cc;
    const size_t slots = cc->GetRingDimension() / 2;

    Inference inf;
    inf.fhe = ctx;
    inf.slots = (int)slots;
    inf.size.hidDim = 256;

    Ctx x = fresh_ct(cc, 3.0, ctx->pk());

    uint32_t lvl_before = level_of(x);
    Ctx result = compute_average(inf, x);
    uint32_t lvl_after = level_of(result);

    uint32_t consumed = lvl_after - lvl_before;
    std::cout << "compute_average: consumed " << consumed << " levels"
              << " (before=" << lvl_before << " after=" << lvl_after << ")\n";

    constexpr uint32_t kDepth = 0;
    EXPECT_EQ(consumed, kDepth);
}

// ── compute_variance ────────────────────────────────────────────────────────
// Operations: EvalMult(x, hD)  →  square  →  EvalMult(varc, 1/hD³)
// Critical path: 3 multiplications (scalar + square + scalar).

TEST_F(OpsTest, DepthBudget_ComputeVariance) {
    const CC& cc = ctx->cc;
    const size_t slots = cc->GetRingDimension() / 2;

    Inference inf;
    inf.fhe = ctx;
    inf.slots = (int)slots;
    inf.size.hidDim = 256;

    std::vector<double> x(slots);
    for (size_t i = 0; i < slots; ++i)
        x[i] = (i < slots / 2) ? 0.0 : 2.0;

    auto pt = encode(cc, x);
    Ctx ct = encrypt(cc, pt, ctx->pk());

    uint32_t lvl_before = level_of(ct);
    Ctx result = compute_variance(inf, ct);
    uint32_t lvl_after = level_of(result);

    uint32_t consumed = lvl_after - lvl_before;
    std::cout << "compute_variance: consumed " << consumed << " levels"
              << " (before=" << lvl_before << " after=" << lvl_after << ")\n";

    constexpr uint32_t kDepth = 3;  // scalar mult + square + scalar mult
    EXPECT_GE(consumed, kDepth - 1);
    EXPECT_LE(consumed, kDepth);
}
