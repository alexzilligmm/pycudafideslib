#pragma once

#include <gtest/gtest.h>
#include "fideslib_wrapper.h"
#include "ckks_primitives.h" 
#include "inference.h"    
#include "gpt2.h"        
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
    // CacheMIR paper §7.1 — hardcoded, no dependency on generated config
    static constexpr uint32_t LOG_N    = 16;
    static constexpr int      DEPTH    = 13;   // L
    static constexpr int      OVERHEAD = 15;   // K

    static std::shared_ptr<CKKSContext> ctx;
    static Inference inf;

    static void SetUpTestSuite() {
        if (ctx) return;  // already built
        // OpenFHE bootstrap needs ≥59-bit moduli (shares Q chain, no separate P primes)
        ctx = make_ckks_context(
            /*logN=*/              LOG_N,
            /*depth=*/             DEPTH,
            /*scale_bits=*/        41,
            /*bootstrap_slots=*/   0,
            /*enable_bootstrap=*/  true,
            /*btp_scale_bits=*/    50,
            /*first_mod_bits=*/    53,
            /*level_budget_in=*/   {4, 3},
            /*batch_size=*/        0,
            /*h_weight=*/          192,
            /*num_large_digits=*/  3,
            /*btp_depth_overhead=*/ OVERHEAD
        );
        inf = make_inf(ctx, DEPTH, OVERHEAD);
        // GPT-2 small: padded hD=1024 (must divide S=32768), real=768
        inf.size.hidDim   = 1024;
        inf.size.dim      = 768;
        inf.size.expDim   = 4096;
        std::cout << "--- CONTEXT INITIALIZED (once) ---\n"
                  << "N=" << (1 << LOG_N)
                  << " slots=" << (1 << (LOG_N-1))
                  << " L=" << DEPTH << " K=" << OVERHEAD
                  << " total=" << (DEPTH + OVERHEAD)
                  << " hidDim=" << inf.size.hidDim
                  << " expDim=" << inf.size.expDim << "\n";
    }
};

inline std::shared_ptr<CKKSContext> NonLinearTest::ctx = nullptr;
inline Inference NonLinearTest::inf = {};
