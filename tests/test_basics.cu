#include <gtest/gtest.h>
#include "fideslib_wrapper.h"

#include <cmath>
#include <numeric>
#include <algorithm>

class OpsTest : public ::testing::Test {
protected:
    static constexpr double TOL = 1e-6;

    std::shared_ptr<CKKSContext> ctx;

    void SetUp() override {
        ctx = make_ckks_context(       // assign to member, not a local
            /*logN=*/              12,
            /*depth=*/             16,
            /*scale_bits=*/        40,
            /*bootstrap_slots=*/   0,
            /*enable_bootstrap=*/  true,
            /*btp_scale_bits=*/    40,   // Go LogDefaultScale=40
            /*first_mod_bits=*/    52,   // Go LogQ[0]=52
            /*level_budget_in=*/   {3, 3},
            /*batch_size=*/        0,
            /*h_weight=*/          192,  // Go ring.Ternary{H:192} → SPARSE_TERNARY
            /*num_large_digits=*/  3,    // Go LogP has 3 entries
            /*btp_depth_overhead=*/ 9
        );
        const auto& cc = ctx->cc;
        std::cout << "Context created:"
                  << " N="     << cc->GetRingDimension()      // 4096
                  << " 2N="    << cc->GetCyclotomicOrder()    // 8192
                  << " depth=" << cc->multiplicative_depth    // depth + btp_overhead
                  << " keyDist=" << cc->keyDist               // SPARSE_TERNARY when h_weight>0
                  << "\n";
    }
};

TEST_F(OpsTest, CCparams) {
    const CC& cc = ctx->cc;
    EXPECT_EQ(cc->GetRingDimension(),  4096u);   // N = 2^logN
    EXPECT_EQ(cc->GetCyclotomicOrder(), 8192u);  // 2N
    EXPECT_EQ(cc->keyDist, SPARSE_TERNARY);      // h_weight=192 → SPARSE_TERNARY
}