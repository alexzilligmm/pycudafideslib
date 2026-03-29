#pragma once

#include <gtest/gtest.h>
#include "fideslib_wrapper.h"
#include <cmath>
#include <numeric>

class OpsTest : public ::testing::Test {
protected:
    // CacheMIR paper §7.1 — hardcoded, no dependency on generated config
    static constexpr uint32_t LOG_N = 16;
    static constexpr int L = 13;   // max working level
    static constexpr int K = 15;   // bootstrap circuit depth

    static std::shared_ptr<CKKSContext> ctx;

    static void SetUpTestSuite() {
        if (ctx) return;  // already built
        // OpenFHE bootstrap uses the SAME Q chain (not separate P primes like Lattigo).
        // Paper's 41-bit moduli work in Lattigo because bootstrap has its own 61-bit P primes.
        // In FIDESlib/OpenFHE, all K=15 bootstrap levels share the Q chain → need ≥59-bit moduli.
        ctx = make_ckks_context(
            /*logN=*/              LOG_N,
            /*depth=*/             L,
            /*scale_bits=*/        41,
            /*bootstrap_slots=*/   0,
            /*enable_bootstrap=*/  true,
            /*btp_scale_bits=*/    50,
            /*first_mod_bits=*/    53,
            /*level_budget_in=*/   {4, 3},
            /*batch_size=*/        0,
            /*h_weight=*/          192,
            /*num_large_digits=*/  3,
            /*btp_depth_overhead=*/ K
        );
        std::cout << "--- CONTEXT INITIALIZED (once) ---\n"
                  << "N=" << (1 << LOG_N)
                  << " slots=" << (1 << (LOG_N-1))
                  << " L=" << L << " K=" << K
                  << " total_depth=" << (L + K) << "\n";
    }
};

inline std::shared_ptr<CKKSContext> OpsTest::ctx = nullptr;
