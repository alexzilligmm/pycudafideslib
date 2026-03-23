#include <gtest/gtest.h>
#include "test_basics.h"
#include <cmath>
#include <numeric>


TEST_F(OpsTest, CCparams) {
    const CC& cc = ctx->cc;
    uint32_t expectedN = 1 << LOG_N;
    EXPECT_EQ(cc->GetRingDimension(),  expectedN);
    EXPECT_EQ(cc->GetCyclotomicOrder(), expectedN * 2);
}

TEST_F(OpsTest, EncryptDecrypt) {
    const CC& cc = ctx->cc;
    const size_t slots = cc->GetRingDimension() / 2;
    std::vector<double> msg(slots, 0.123456789); 

    auto pt  = encode(cc, msg);
    auto ct  = encrypt(cc, pt, ctx->pk());
    auto dec = decrypt(cc, ct, ctx->sk());

    for (size_t i = 0; i < 100; ++i)
        EXPECT_NEAR(dec[i], 0.123456789, 1e-9);
}

TEST_F(OpsTest, LevelManagement) {
    const CC& cc = ctx->cc;
    const size_t slots = cc->GetRingDimension() / 2;
    
    double val = 0.5;
    auto pt = encode(cc, std::vector<double>(slots, val));
    auto ct = encrypt(cc, pt, ctx->pk());

    reduce_to_level(cc, ct, 5, (int)slots);
    
    EXPECT_EQ(level_of(ct), 5u);
    auto result = decrypt(cc, ct, ctx->sk());
    EXPECT_NEAR(result[0], 0.5, 1e-6);
}

TEST_F(OpsTest, DeepLevelBootstrap) {
    const CC& cc = ctx->cc;
    const size_t slots = cc->GetRingDimension() / 2;
    
    double test_val = 0.333;
    auto pt = encode(cc, std::vector<double>(slots, test_val));
    auto ct = encrypt(cc, pt, ctx->pk());

    uint32_t target_level = 9; 
    reduce_to_level(cc, ct, target_level, (int)slots);
    
    std::cout << "Level before bootstrap: " << level_of(ct) << "\n";
    EXPECT_EQ(level_of(ct), target_level);

    std::cout << "Starting Bootstrapping...\n";
    auto ct_btp = ctx->cc->EvalBootstrap(ct);
    
    uint32_t post_btp_lvl = level_of(ct_btp);
    std::cout << "Level after bootstrap: " << post_btp_lvl << "\n";

    EXPECT_GT(post_btp_lvl, target_level);

    auto result = decrypt(cc, ct_btp, ctx->sk());
    
    EXPECT_NEAR(result[0], test_val, 1e-4);
    std::cout << "Verified Bootstrapped Value: " << result[0] << "\n";
}

TEST_F(OpsTest, BootstrapAfterRealWork) {
    const CC& cc = ctx->cc;
    const size_t slots = cc->GetRingDimension() / 2;
    
    double initial_val = 1.05;
    auto pt = encode(cc, std::vector<double>(slots, initial_val));
    auto ct = encrypt(cc, pt, ctx->pk());
    EXPECT_EQ(level_of(ct), 0u);

    std::cout << "Performing 3 squarings..." << std::endl;
    for (int i = 0; i < 3; ++i) {
        ct = cc->EvalMult(ct, ct);
        cc->RescaleInPlace(ct);
    }
    
    std::cout << "Performing 6 identity multiplications..." << std::endl;
    for (int i = 0; i < 6; ++i) {
        auto pt_one = encode(cc, std::vector<double>(slots, 1.0), level_of(ct));
        cc->EvalMultInPlace(ct, pt_one);
        cc->RescaleInPlace(ct);
    }

    uint32_t current_lvl = level_of(ct);
    std::cout << "Level before bootstrap: " << current_lvl << std::endl;
    EXPECT_EQ(current_lvl, 9u); 

    std::cout << "Starting Bootstrapping..." << std::endl;
    auto ct_btp = cc->EvalBootstrap(ct);
    
    uint32_t post_btp_lvl = level_of(ct_btp);
    std::cout << "Level after bootstrap: " << post_btp_lvl << std::endl;
    EXPECT_GT(post_btp_lvl, current_lvl);

    auto result = decrypt(cc, ct_btp, ctx->sk());

    double expected = std::pow(initial_val, 8.0);
    
    EXPECT_NEAR(result[0], expected, 1e-3);
    
    std::cout << "Successfully bootstrapped after 9 levels of real operations!" << std::endl;
    std::cout << "Expected: " << expected << ", Actual: " << result[0] << std::endl;
}