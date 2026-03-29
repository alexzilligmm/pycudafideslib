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
        EXPECT_NEAR(dec[i], 0.123456789, 1e-6);
}

TEST_F(OpsTest, LevelManagement) {
    const CC& cc = ctx->cc;
    const size_t slots = cc->GetRingDimension() / 2;

    double val = 0.5;
    auto pt = encode(cc, std::vector<double>(slots, val));
    auto ct = encrypt(cc, pt, ctx->pk());

    // Consume levels via EvalMult+Rescale to reach level 5
    reduce_to_level(cc, ct, 5, (int)slots);

    EXPECT_EQ(level_of(ct), 5u);
    auto result = decrypt(cc, ct, ctx->sk());
    EXPECT_NEAR(result[0], 0.5, 1e-4);
}

// Consume L levels (one full working depth), bootstrap, repeat.
// With L=13 and scale=41, each mult+rescale burns 1 level.
// We do 10 mults per round (leaving ~3 levels margin), then bootstrap.
TEST_F(OpsTest, ChainedBootstraps) {
    const CC& cc = ctx->cc;
    const size_t slots = cc->GetRingDimension() / 2;

    const int mults_per_round = 10;  // < L=13 to leave headroom
    const int num_rounds = 3;        // 2 bootstraps total

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
        EXPECT_NEAR(dec[0], val, std::abs(val) * 0.1);

        if (round < num_rounds - 1) {
            std::cout << "  bootstrapping from level=" << level_of(ct) << "...\n";
            ct = cc->EvalBootstrap(ct);
            std::cout << "  post-bootstrap level=" << level_of(ct) << "\n";

            // Verify bootstrap didn't destroy the value
            auto post_btp = decrypt(cc, ct, ctx->sk());
            double btp_err = std::abs(post_btp[0] - val) / std::abs(val);
            std::cout << "  post-bootstrap val=" << post_btp[0]
                      << " rel_err=" << btp_err << "\n";
            EXPECT_LT(btp_err, 0.1) << "Bootstrap precision too low";
        }
    }

    auto final_dec = decrypt(cc, ct, ctx->sk());
    std::cout << "\n=== " << total_mults << " total multiplications (L=" << L << ") ===\n"
              << "Final value: " << final_dec[0] << " (expected " << val << ")\n";

    EXPECT_NEAR(final_dec[0], val, std::abs(val) * 0.15);
    EXPECT_EQ(total_mults, mults_per_round * num_rounds);
}

// Bootstrap from different input levels — find the precision frontier.
// At logN=16, bootstrap from shallow levels (1-3) may have high noise
// while deep levels (10+) should work.
TEST_F(OpsTest, BootstrapPrecisionSweep) {
    const CC& cc = ctx->cc;
    const size_t slots = cc->GetRingDimension() / 2;
    const int total_depth = L + K;

    std::cout << "--- Bootstrap precision sweep (total_depth=" << total_depth << ") ---\n";

    for (int input_level : {5, 8, 10, 12, 13, 15, 20}) {
        if ((uint32_t)input_level >= (uint32_t)total_depth) continue;

        double val = 1.5;
        auto ct = encrypt(cc, encode(cc, std::vector<double>(slots, val)), ctx->pk());
        reduce_to_level(cc, ct, (uint32_t)input_level, (int)slots);

        std::cout << "  BTP from level " << level_of(ct) << "..." << std::flush;
        Ctx fresh = cc->EvalBootstrap(ct);
        std::cout << " output_level=" << level_of(fresh) << std::flush;

        try {
            auto dec = decrypt(cc, fresh, ctx->sk());
            double err = std::abs(dec[0] - val);
            double bits = (err > 0) ? -std::log2(err / val) : 99;
            std::cout << " val=" << dec[0] << " err=" << err
                      << " bits=" << bits << "\n";
            // At deep levels, expect reasonable precision
            if (input_level >= 10) {
                EXPECT_LT(err, 0.5) << "Bootstrap from level " << input_level
                                    << " should preserve value within 0.5";
            }
        } catch (const std::exception& e) {
            std::cout << " DECODE FAIL: " << e.what() << "\n";
            // Shallow levels may fail — only assert for deep levels
            if (input_level >= 10) {
                ADD_FAILURE() << "Bootstrap from level " << input_level
                              << " should not fail decode";
            }
        }
    }
}