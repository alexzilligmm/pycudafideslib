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