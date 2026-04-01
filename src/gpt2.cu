#include "gpt2.h"
#include "gpt2_config.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <set>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <omp.h>



Inference make_gpt2(int logN, int hidDim, int ffDim,
                    int seqLen, int numHeads, bool parallel,
                    bool bench) {
    Inference inf;
    inf.size.hidDim   = hidDim;
    inf.size.expDim   = ffDim;
    inf.size.numHeads = numHeads;
    inf.size.seqLen   = seqLen;
    inf.logN     = logN;
    inf.parallel = parallel;

    std::cout << "Creating CKKS context for GPT-2 (logN=" << logN << ")...\n";
    uint32_t btp_slots = (uint32_t)(1 << (logN - 1));
    int S = (int)btp_slots;

    constexpr int kDepth    = 13;
    constexpr int kBtpExtra = 15;

    int btp_sbits = kGPT2BtpScaleBits;
    int first_mod = kGPT2FirstModBits;
    
    /// TODO: restore this function
    /// auto extra_rots = compute_gpt2_rot_indices(S, hidDim, ffDim, numHeads, seqLen);
    std::cout << "  Extra rotation indices: " << extra_rots.size() << "\n";

    inf.fhe         = make_ckks_context(logN, kDepth, /*scale_bits=*/41,
                                        btp_slots, /*bootstrap=*/true,
                                        /*btp_scale_bits=*/btp_sbits,
                                        /*first_mod_bits=*/first_mod,
                                        /*level_budget_in=*/{4, 3},
                                        /*batch_size=*/0,
                                        /*h_weight=*/192,
                                        /*num_large_digits=*/3,
                                        /*btp_depth_overhead=*/kBtpExtra,
                                        /*extra_rot_steps=*/extra_rots);
    inf.slots       = (int)btp_slots;
    inf.total_depth = kDepth + kBtpExtra;
    inf.bench_mode  = bench;
    std::cout << "  slots=" << inf.slots
              << "  bench_mode=" << inf.bench_mode
              << "  GPU context loaded.\n";
    return inf;
}

static std::mt19937 gpt2_rng(42);

Ptx gpt2_rand_plaintext(Inference& inf) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> msg(inf.slots);
    for (auto& v : msg) v = dist(gpt2_rng);
    return inf.cc()->MakeCKKSPackedPlaintext(msg);
}


void gpt2_generate_random_weights(Inference& inf, const std::vector<std::string>& names) {
    std::cout << "Preparing GPT-2 weights (interleaved)..." << std::endl;
    const int hidDim = inf.size.hidDim, expDim = inf.size.expDim, numSlots = inf.slots;
    for (const auto& n : names) {
        int cnt;
        if      (n == "q" || n == "k" || n == "v" || n == "out") cnt = hidDim * hidDim / numSlots;
        else if (n == "up" || n == "gate" || n == "down")         cnt = hidDim * expDim / numSlots;
        else throw std::runtime_error("Unknown GPT-2 weight: " + n);

        inf.w[n].clear();
        for (int i = 0; i < cnt; ++i)
            inf.w[n].push_back(gpt2_rand_plaintext(inf));
    }
    std::cout << "GPT-2 weights ready." << std::endl;
}












