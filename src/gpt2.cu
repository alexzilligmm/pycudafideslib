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



static void collect_linear_rots(std::set<int32_t>& rots, int N, int d_in, int d_out) {
    auto p = compute_cm_params(N, d_in, d_out);

    // Input accumulation: step * (t - 1), step = 1,2,4,... while step < tp_in
    for (int step = 1; step < p.tp_in; step *= 2)
        rots.insert(step * (p.t - 1));

    // Input rotation: j * t^2, j = 1..r_i-1
    int rot2 = p.t * p.t;
    for (int j = 1; j < p.r_i; ++j)
        rots.insert(j * rot2);

    // Cascade rotation: t * tp
    rots.insert(p.t * p.tp);

    // Output accumulation: step = 1,2,4,... while step < tp_out
    for (int step = 1; step < p.tp_out; step *= 2)
        rots.insert(step);
}

static void collect_mha_rots(std::set<int32_t>& rots, int N, int hidDim, int numHeads) {
    int t  = N / hidDim;
    int tH = t * numHeads;

    // cache_k_push: rotate key into token slot
    for (int i = 1; i < t; ++i)
        rots.insert(-i);

    // qkt query fill: replicate across token slots
    for (int step = 1; step < t; step *= 2)
        rots.insert(-step);

    // qkt sum_by_rot: reduce across dimension blocks
    for (int s = tH; s < N; s *= 2)
        rots.insert(s);

    // head_reduce_sum: intra-head masked rotations
    for (int step = 1; step < t; step *= 2) {
        rots.insert(step);
        rots.insert(step - t);  // wrap-around rotation
    }
    // head_reduce_sum: inter-block aggregation (same as sum_by_rot, already inserted)

    // softmax_v: rotate scores by tH per metaciphertext lane (already inserted via sum_by_rot)
    // softmax_v: intra-token reduction
    for (int step = 1; step < t; step *= 2)
        rots.insert(step);  // already inserted by head_reduce_sum, but explicit for clarity

    // cache_v_push: rotate value into token slot (same as cache_k_push, already inserted)
}

std::vector<int32_t> compute_gpt2_rot_indices(
    int S, int hidDim, int ffDim, int numHeads, int seqLen) {
    std::set<int32_t> rots;

    // q/k/v/out: hidDim x hidDim
    collect_linear_rots(rots, S, hidDim, hidDim);
    // up/gate: hidDim x ffDim
    collect_linear_rots(rots, S, hidDim, ffDim);
    // down: ffDim x hidDim
    collect_linear_rots(rots, S, ffDim, hidDim);

    // MHA: KCache, qkt
    collect_mha_rots(rots, S, hidDim, numHeads);

    return std::vector<int32_t>(rots.begin(), rots.end());
}

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

    auto extra_rots = compute_gpt2_rot_indices(S, hidDim, ffDim, numHeads, seqLen);
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



void gpt2_prepare_weights(Inference& inf, const std::vector<std::string>& names) {
    const int hidDim = inf.size.hidDim;
    const int expDim = inf.size.expDim;

    for (const auto& n : names) {
        int d_in, d_out;
        if      (n == "q" || n == "k" || n == "v" || n == "out") { d_in = hidDim; d_out = hidDim; }
        else if (n == "up" || n == "gate")                        { d_in = hidDim; d_out = expDim; }
        else if (n == "down")                                     { d_in = expDim; d_out = hidDim; }
        else throw std::runtime_error("Unknown GPT-2 weight: " + n);

        std::string path = inf.weight_dir + "/" + n + ".txt";
        std::cout << "Loading " << n << " (" << d_in << "x" << d_out << ") from " << path << "\n";
        inf.w[n] = load_weight_txt(inf, path, d_in, d_out);
    }
    std::cout << "GPT-2 weights loaded." << std::endl;
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












