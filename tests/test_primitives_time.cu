#include <gtest/gtest.h>
#include "fideslib_wrapper.h"
#include <cmath>
#include <numeric>
#include "test_basics.h"
#include "ckks_primitives.h" // For the primitive implementations we're testing
#include <cub/cub.cuh>

TEST_F(OpsTest, GoldschmidtInvSqrtTime) {
    const CC& cc = ctx->cc;
    const size_t slots = cc->GetRingDimension() / 2;
    
    double test_val = 0.333;
    auto pt = encode(cc, std::vector<double>(slots, test_val));
    auto ct = encrypt(cc, pt, ctx->pk());

    double init_guess = 1.0;
    auto guess_pt = encode(cc, std::vector<double>(slots, init_guess));
    auto guess_ct = encrypt(cc, guess_pt, ctx->pk());

    // Warmup
    for (int i = 0; i < 100; ++i) {
        goldschmidt_inv_sqrt(cc, ct, guess_ct, /*iterations=*/5);
    }

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 1000; ++i) {
        goldschmidt_inv_sqrt(cc, ct, guess_ct, /*iterations=*/5);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    std::cout << "Goldschmidt InvSqrt Time: " << milliseconds << " ms\n";
    EXPECT_TRUE(true); // Dummy assertion to ensure test runs
}

TEST_F(OpsTest, NewtonRaphsonInvSqrtTime) {
    const CC& cc = ctx->cc;
    const size_t slots = cc->GetRingDimension() / 2;
    
    double test_val = 0.333;
    auto pt = encode(cc, std::vector<double>(slots, test_val));
    auto ct = encrypt(cc, pt, ctx->pk());

    double init_guess = 1.0;
    auto guess_pt = encode(cc, std::vector<double>(slots, init_guess));
    auto guess_ct = encrypt(cc, guess_pt, ctx->pk());

    // Warmup
    for (int i = 0; i < 100; ++i) {
        inv_sqrt_newton(cc, ct, guess_ct, /*iterations=*/5);
    }

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 1000; ++i) {
        inv_sqrt_newton(cc, ct, guess_ct, /*iterations=*/5);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    std::cout << "Newton-Raphson InvSqrt Time: " << milliseconds << " ms\n";
    EXPECT_TRUE(true); // Dummy assertion to ensure test runs
}