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

static int find_bsgs_inRot(int d) {
    int r = (int)std::sqrt((double)d);
    while (r > 1 && d % r != 0) --r;
    return r;
}

std::vector<int32_t> compute_gpt2_rot_indices(
    int S, int hidDim, int ffDim, int numHeads, int seqLen)
{
    std::set<int32_t> indices;

    auto add_interleaved_rots = [&](int d_in, int d_out) {
        const int t_in  = S / d_in;
        const int t_out = S / d_out;
        const int intRot = S / std::max(d_in, d_out);
        const int n_pt  = d_in * d_out / S;

        int inRot = std::max(1, (int)std::sqrt((double)(d_in * d_out / (2 * S))));
        int outRot = n_pt / inRot;

        for (int i = 1; i < inRot; ++i) {
            indices.insert(i * intRot);
            indices.insert(-(i * intRot));
        }

        for (int i = 1; i < outRot; ++i) {
            int rv = i * intRot * inRot;
            indices.insert(rv);
            indices.insert(-rv);
        }

        for (int step = 1; step < t_in; step *= 2) {
            int rv = step * (t_in - 1);
            indices.insert(rv % S);
            indices.insert(-(rv % S));
        }

        for (int step = 1; step < t_out; step *= 2) {
            indices.insert(step);
            indices.insert(-step);
        }
    };

    add_interleaved_rots(hidDim, hidDim);    // Q/K/V/Out
    add_interleaved_rots(hidDim, ffDim);     // Up
    add_interleaved_rots(ffDim, hidDim);     // Down

    int headDim = hidDim / numHeads;
    int intRot  = S / hidDim;

    int inner_step = numHeads * S / hidDim;
    for (int j = 1; j < headDim; j *= 2) {
        int rot_val = inner_step * j;
        indices.insert(rot_val);
        indices.insert(-rot_val);
    }

    int space_qk = S * S / (seqLen * hidDim);
    for (int i = 1; i < hidDim * seqLen / S; ++i)
        indices.insert(S - space_qk * i);

    int space_av = S * numHeads / hidDim;
    int av_inRot = find_bsgs_inRot(headDim);
    int av_outRot = av_inRot * 2;
    indices.insert(space_av);
    indices.insert(-space_av);
    for (int i = 1; i < av_outRot; ++i) {
        indices.insert(i * space_av * av_inRot);
        indices.insert(-(i * space_av * av_inRot));
    }

    for (int idx = 0; idx < intRot; ++idx) {
        indices.insert(idx);
        indices.insert(-idx);
    }
    for (int mid = 0; mid <= seqLen / intRot; ++mid) {
        int rot_idx = mid * S * numHeads / hidDim;
        for (int t = 0; t < intRot; ++t) {
            indices.insert(t + rot_idx);
            indices.insert(-(t + rot_idx));
        }
    }

    indices.erase(0);
    return std::vector<int32_t>(indices.begin(), indices.end());
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


// TODO: make this expect weights are they are in hf model.
static std::vector<std::vector<double>> read_weight_file_txt(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open weight file: " + path);
    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::vector<double> vals;
        double v;
        while (iss >> v) vals.push_back(v);
        if (!vals.empty()) rows.push_back(std::move(vals));
    }
    return rows;
}












