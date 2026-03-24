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

TEST_F(OpsTest, ChainedBootstraps) {
    const CC& cc = ctx->cc;
    const size_t slots = cc->GetRingDimension() / 2;

    const int mults_per_round = 8;
    const int num_rounds = 4;

    double val = 0.5;
    auto ct = encrypt(cc, encode(cc, std::vector<double>(slots, val)), ctx->pk());

    int total_mults = 0;

    for (int round = 0; round < num_rounds; ++round) {
        std::cout << "--- Round " << round << ": level=" << level_of(ct) << " ---\n";

        for (int i = 0; i < mults_per_round; ++i) {
            auto pt_scale = encode(cc, std::vector<double>(slots, 0.95), level_of(ct));
            cc->EvalMultInPlace(ct, pt_scale);
            cc->RescaleInPlace(ct);
        }
        total_mults += mults_per_round;
        val *= std::pow(0.95, mults_per_round);

        auto dec = decrypt(cc, ct, ctx->sk());
        std::cout << "  after " << total_mults << " mults: level=" << level_of(ct)
                  << " val=" << dec[0] << " (expected " << val << ")\n";
        EXPECT_NEAR(dec[0], val, std::abs(val) * 0.05);

        if (round < num_rounds - 1) {
            std::cout << "  bootstrapping...\n";
            ct = cc->EvalBootstrap(ct);
            std::cout << "  post-bootstrap level=" << level_of(ct) << "\n";
        }
    }

    auto final_dec = decrypt(cc, ct, ctx->sk());
    std::cout << "\n=== " << total_mults << " total multiplications with depth=16 ===\n"
              << "Final value: " << final_dec[0] << " (expected " << val << ")\n";

    EXPECT_NEAR(final_dec[0], val, std::abs(val) * 0.05);
    EXPECT_EQ(total_mults, 32);
}