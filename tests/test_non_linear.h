#pragma once

#include <gtest/gtest.h>
#include "fideslib_wrapper.h"
#include "ckks_primitives.h" 
#include "inference.h"    
#include "llama.h"        
#include <cmath>

inline Inference make_inf(std::shared_ptr<CKKSContext> ctx, int depth, int overhead) {
    Inference inf;
    inf.fhe         = ctx;
    inf.slots       = (int)(ctx->cc->GetRingDimension() / 2);
    inf.logN        = (int)std::round(std::log2((double)ctx->cc->GetRingDimension()));
    inf.total_depth = depth + overhead;
    return inf;
}

class NonLinearTest : public ::testing::Test {
protected:
    static constexpr uint32_t LOG_N    = 12;
    static constexpr int      DEPTH    = 28;
    static constexpr int      OVERHEAD = 9;

    std::shared_ptr<CKKSContext> ctx;
    Inference inf;

    void SetUp() override {
        ctx = make_ckks_context(
            /*logN=*/              LOG_N,
            /*depth=*/             DEPTH,
            /*scale_bits=*/        40,
            /*bootstrap_slots=*/   0,
            /*enable_bootstrap=*/  true,
            /*btp_scale_bits=*/    59,
            /*first_mod_bits=*/    60,
            /*level_budget_in=*/   {3, 3},
            /*batch_size=*/        0,
            /*h_weight=*/          192,
            /*num_large_digits=*/  3,
            /*btp_depth_overhead=*/ OVERHEAD
        );
        inf = make_inf(ctx, DEPTH, OVERHEAD);
        std::cout << "--- CONTEXT INITIALIZED ---\n"
                  << "N=" << (1 << LOG_N)
                  << " depth=" << DEPTH
                  << " overhead=" << OVERHEAD
                  << " total=" << (DEPTH + OVERHEAD) << "\n";
    }
};
