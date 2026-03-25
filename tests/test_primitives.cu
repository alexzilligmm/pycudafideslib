#include <gtest/gtest.h>
#include "fideslib_wrapper.h"
#include <cmath>
#include <numeric>
#include "test_basics.h" // For OpsTest fixture and context setup#include "ckks_primitives.h" // For the primitive implementations we're testing
#include "ckks_primitives.h" 

TEST_F(OpsTest, GoldschmidtInvSqrt) {
    const CC& cc = ctx->cc;
    const size_t slots = cc->GetRingDimension() / 2;
    
    double test_val = 0.333;
    auto pt = encode(cc, std::vector<double>(slots, test_val));
    auto ct = encrypt(cc, pt, ctx->pk());

    double init_guess = 1.0;
    auto guess_pt = encode(cc, std::vector<double>(slots, init_guess));
    auto guess_ct = encrypt(cc, guess_pt, ctx->pk());

    Ctx ct_inv_sqrt = goldschmidt_inv_sqrt(cc, ct, guess_ct, /*iterations=*/5);
    
    auto result = decrypt(cc, ct_inv_sqrt, ctx->sk());
    
    double expected = 1.0 / std::sqrt(test_val);
    EXPECT_NEAR(result[0], expected, 1e-4);
    std::cout << "Level after Goldschmidt InvSqrt: " << level_of(ct_inv_sqrt) << "\n";
    std::cout << "Verified InvSqrt Value: " << result[0] << " (expected: " << expected << ")\n";
}

TEST_F(OpsTest, NewtonRaphsonInvSqrt) {
    const CC& cc = ctx->cc;
    const size_t slots = cc->GetRingDimension() / 2;
    
    double test_val = 0.333;
    auto pt = encode(cc, std::vector<double>(slots, test_val));
    auto ct = encrypt(cc, pt, ctx->pk());

    double init_guess = 1.0;
    auto guess_pt = encode(cc, std::vector<double>(slots, init_guess));
    auto guess_ct = encrypt(cc, guess_pt, ctx->pk());

    Ctx ct_inv_sqrt = inv_sqrt_newton(cc, ct, guess_ct, /*iterations=*/5);
    
    auto result = decrypt(cc, ct_inv_sqrt, ctx->sk());
    
    double expected = 1.0 / std::sqrt(test_val);
    EXPECT_NEAR(result[0], expected, 1e-4);
    std::cout << "Level after Newton-Raphson InvSqrt: " << level_of(ct_inv_sqrt) << "\n";
    std::cout << "Verified InvSqrt Value: " << result[0] << " (expected: " << expected << ")\n";
}

TEST_F(OpsTest, NewtonRaphsonInv) {
    const CC& cc = ctx->cc;
    const size_t slots = cc->GetRingDimension() / 2;
    
    double test_val = 0.333;
    auto pt = encode(cc, std::vector<double>(slots, test_val));
    auto ct = encrypt(cc, pt, ctx->pk());

    double init_guess = 1.5;
    auto guess_pt = encode(cc, std::vector<double>(slots, init_guess));
    auto guess_ct = encrypt(cc, guess_pt, ctx->pk());

    Ctx ct_inv = newton_inverse(cc, guess_ct, ct, /*iterations=*/5);
    
    auto result = decrypt(cc, ct_inv, ctx->sk());
    
    double expected = 1.0 / test_val;
    EXPECT_NEAR(result[0], expected, 1e-4);
    std::cout << "Level after Newton-Raphson Inv: " << level_of(ct_inv) << "\n";
    std::cout << "Verified Inv Value: " << result[0] << " (expected: " << expected << ")\n";
}

TEST_F(OpsTest, ComputeAverage_Constant) {
    const CC& cc = ctx->cc;
    const size_t slots = cc->GetRingDimension() / 2;

    Inference inf;
    inf.fhe    = ctx;
    inf.slots  = (int)slots;
    inf.size.hidDim = 256; 

    const double c = 5.0;
    auto pt = encode(cc, std::vector<double>(slots, c));
    auto ct = encrypt(cc, pt, ctx->pk());

    Ctx result = compute_average(inf, ct);
    auto vals  = decrypt(cc, result, ctx->sk());

    double expected = 256.0 * c;
    EXPECT_NEAR(vals[0], expected, 1e-2);
    std::cout << "ComputeAverage (constant) result[0]=" << vals[0]
              << " expected=" << expected << "\n";
}

TEST_F(OpsTest, ComputeVariance_Constant) {
    const CC& cc = ctx->cc;
    const size_t slots = cc->GetRingDimension() / 2;

    Inference inf;
    inf.fhe    = ctx;
    inf.slots  = (int)slots;
    inf.size.hidDim = 256;

    const double c = 5.0;
    auto pt = encode(cc, std::vector<double>(slots, c));
    auto ct = encrypt(cc, pt, ctx->pk());

    Ctx result = compute_variance(inf, ct);
    auto vals  = decrypt(cc, result, ctx->sk());

    EXPECT_NEAR(vals[0], 0.0, 1e-2);
    std::cout << "ComputeVariance (constant) result[0]=" << vals[0] << "\n";
}

TEST_F(OpsTest, ComputeVariance_TwoValue) {
    const CC& cc = ctx->cc;
    const size_t slots = cc->GetRingDimension() / 2; // 2048

    Inference inf;
    inf.fhe    = ctx;
    inf.slots  = (int)slots;
    inf.size.hidDim = 256; 

    std::vector<double> x(slots);
    for (size_t i = 0; i < slots; ++i)
        x[i] = (i < slots / 2) ? 0.0 : 2.0;

    auto pt = encode(cc, x);
    auto ct = encrypt(cc, pt, ctx->pk());

    Ctx result = compute_variance(inf, ct);
    auto vals  = decrypt(cc, result, ctx->sk());

    EXPECT_NEAR(vals[0], 1.0, 1e-2);
    std::cout << "ComputeVariance (two-value) result[0]=" << vals[0]
              << " expected=1.0\n";
}

TEST_F(OpsTest, GoldschmidtInv) {
    const CC& cc = ctx->cc;
    const size_t slots = cc->GetRingDimension() / 2;
    
    double test_val = 0.333;
    auto pt = encode(cc, std::vector<double>(slots, test_val));
    auto ct = encrypt(cc, pt, ctx->pk());

    double init_guess = 1.5;
    auto guess_pt = encode(cc, std::vector<double>(slots, init_guess));
    auto guess_ct = encrypt(cc, guess_pt, ctx->pk());

    Ctx ct_inv = goldschmidt_inv(cc, ct, guess_ct, /*iterations=*/5);
    
    auto result = decrypt(cc, ct_inv, ctx->sk());
    
    double expected = 1.0 / test_val;
    EXPECT_NEAR(result[0], expected, 1e-4);
    std::cout << "Level after Goldschmidt Inv: " << level_of(ct_inv) << "\n";
    std::cout << "Verified Inv Value: " << result[0] << " (expected: " << expected << ")\n";
}