#pragma once

#include <gtest/gtest.h>
#include "fideslib_wrapper.h"
#include <cmath>
#include <numeric>

class OpsTest : public ::testing::Test {
protected:
    // logN=12 (N=4096) is stable and fast for these depth tests
    static constexpr uint32_t LOG_N = 12;

    std::shared_ptr<CKKSContext> ctx;

    void SetUp() override {
        ctx = make_ckks_context(
            /*logN=*/              LOG_N,
            /*depth=*/             16,
            /*scale_bits=*/        40,
            /*bootstrap_slots=*/   0,
            /*enable_bootstrap=*/  true,
            /*btp_scale_bits=*/    59,
            /*first_mod_bits=*/    60,
            /*level_budget_in=*/   {3, 3},
            /*batch_size=*/        0,
            /*h_weight=*/          192,
            /*num_large_digits=*/  3,
            /*btp_depth_overhead=*/ 9
        );
        std::cout << "--- CONTEXT INITIALIZED ---" << "\n"
                  << "N=" << (1 << LOG_N) << "\n";
    }
};
