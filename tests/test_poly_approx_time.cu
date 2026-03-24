#include <gtest/gtest.h>
#include "fideslib_wrapper.h"
#include <cmath>
#include <numeric>
#include "test_basics.h"
#include "ckks_primitives.h" // For the primitive implementations we're testing
#include <cub/cub.cuh>

TEST_F(OpsTest, SpacePatersonStockmeyer) {
    const CC& cc = ctx->cc;
    const size_t slots = cc->GetRingDimension() / 2;

    std::vector<double> F4_coeffs = {0.0, 315.0/128.0, 0.0, -420.0/128.0, 0.0};//, 378.0/128.0, 0.0, -180.0/128.0, 0.0, 35.0/128.0};

    double test_val = 0.333;
    auto pt = encode(cc, std::vector<double>(slots, test_val));
    auto ct = encrypt(cc, pt, ctx->pk());

    // Warmup
    for (int i = 0; i < 100; ++i) {
        eval_polynomial_ps(cc, ct, F4_coeffs, ctx->pk(), slots);
    }

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 1000; ++i) {
        eval_polynomial_ps(cc, ct, F4_coeffs, ctx->pk(), slots);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    std::cout << "SpacePatersonStockmeyer Time: " << milliseconds << " ms\n";
    EXPECT_TRUE(true); // Dummy assertion to ensure test runs
}


TEST_F(OpsTest, ComputationalPatersonStockmeyer) {
    const CC& cc = ctx->cc;
    const size_t slots = cc->GetRingDimension() / 2;

    std::vector<double> F4_coeffs = {0.0, 315.0/128.0, 0.0, -420.0/128.0, 0.0};//, 378.0/128.0, 0.0, -180.0/128.0, 0.0, 35.0/128.0};

    double test_val = 0.333;
    auto pt = encode(cc, std::vector<double>(slots, test_val));
    auto ct = encrypt(cc, pt, ctx->pk());

    // Warmup
    for (int i = 0; i < 100; ++i) {
        eval_polynomial_computational_ps(cc, ct, F4_coeffs, ctx->pk(), slots);
    }

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 1000; ++i) {
        eval_polynomial_computational_ps(cc, ct, F4_coeffs, ctx->pk(), slots);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    std::cout << "ComputationalPatersonStockmeyer Time: " << milliseconds << " ms\n";
    EXPECT_TRUE(true); // Dummy assertion to ensure test runs
}

TEST_F(OpsTest, Degree4Polynomial) {
    const CC& cc = ctx->cc;
    const size_t slots = cc->GetRingDimension() / 2;

    std::vector<double> F4_coeffs = {0.0, 315.0/128.0, 0.0, -420.0/128.0, 0.0};//, 378.0/128.0, 0.0, -180.0/128.0, 0.0, 35.0/128.0};

    double test_val = 0.333;
    auto pt = encode(cc, std::vector<double>(slots, test_val));
    auto ct = encrypt(cc, pt, ctx->pk());

    // Warmup
    for (int i = 0; i < 100; ++i) {
        eval_polynomial_deg4(cc, ct, F4_coeffs);
    }

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 1000; ++i) {
        eval_polynomial_deg4(cc, ct, F4_coeffs);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    std::cout << "Degree-4 Polynomial Time: " << milliseconds << " ms\n";
    EXPECT_TRUE(true); // Dummy assertion to ensure test runs
}