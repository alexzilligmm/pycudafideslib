#include "attention.h"
#include "gpt2.h"
#include "nonlinear.h"
#include "ckks_primitives.h"
#include <cmath>
#include <algorithm>

static constexpr int kBenchRot = 5;

static inline int mha_rot(const Inference& inf, int real_idx) {
    return inf.bench_mode ? kBenchRot : real_idx;
}

std::vector<std::vector<double>> rearrange_qkv_weights(
        const std::vector<std::vector<double>>& W, int H) {
    int d_in  = (int)W.size();
    int d_out = (int)W[0].size();
    int d_head = d_out / H;
    std::vector<std::vector<double>> out(d_in, std::vector<double>(d_out));
    for (int r = 0; r < d_out; ++r) {
        int h  = r % H;
        int ld = r / H;
        int src_col = h * d_head + ld;
        for (int i = 0; i < d_in; ++i)
            out[i][r] = W[i][src_col];
    }
    return out;
}

void prepare_mha_masks(Inference& inf) {
    int N = inf.slots;
    int d = inf.size.hidDim;
    int t = N / d;

    // tok0_mask: 1 at every position where i % t == 0
    std::vector<double> tok0(N, 0.0);
    for (int i = 0; i < N; ++i)
        tok0[i] = (i % t == 0) ? 1.0 : 0.0;
    inf.mask["tok0"] = inf.cc()->MakeCKKSPackedPlaintext(tok0);

    // Initialize empty K cache
    inf.cache["k"] = {};
    inf.k_count = 0;
}

void cache_k_push(Inference& inf, const Ctx& key_ct) {
    const CC& cc = inf.cc();
    int N = inf.slots;
    int d = inf.size.hidDim;
    int t = N / d;

    Ctx masked = cc->EvalMult(key_ct, inf.mask["tok0"]);

    if (inf.k_count % t == 0) {
        // New group
        inf.cache["k"].push_back(std::move(masked));
    } else {
        // Rotate into correct token slot and add to current group
        Ctx rotated = cc->EvalRotate(masked, mha_rot(inf, -(inf.k_count % t)));
        cc->EvalAddInPlace(inf.cache["k"].back(), rotated);
    }
    inf.k_count++;
}

Ctx qkt(Inference& inf, const Ctx& query_ct) {
    const CC& cc = inf.cc();
    int N  = inf.slots;
    int d  = inf.size.hidDim;
    int H  = inf.size.numHeads;
    int t  = N / d;
    int tH = t * H;

    // Mask query to tok=0 positions
    Ctx q = cc->EvalMult(query_ct, inf.mask["tok0"]);

    // Fill: replicate across token slots within each head block
    for (int step = 1; step < t; step *= 2) {
        Ctx tmp = cc->EvalRotate(q, mha_rot(inf, -step));
        cc->EvalAddInPlace(q, tmp);
    }

    // For each cache group: elementwise mult + sum_by_rot + group mask
    int keys_per_ct = t;
    int num_groups  = (int)inf.cache["k"].size();
    Ctx attn_ct;

    for (int g = 0; g < num_groups; ++g) {
        int first_key = g * keys_per_ct;
        int num_tok   = std::min(keys_per_ct, inf.k_count - first_key);

        // Elementwise product
        Ctx result = cc->EvalMult(q, inf.cache["k"][g]);

        // sum_by_rot: reduce across dimension blocks
        for (int s = tH; s < N; s *= 2) {
            Ctx tmp = cc->EvalRotate(result, mha_rot(inf, s));
            cc->EvalAddInPlace(result, tmp);
        }

        // Group mask: keep only slots belonging to group g with valid tokens
        std::vector<double> gmask(N, 0.0);
        for (int i = 0; i < N; ++i) {
            if (i / tH == g && i % t < num_tok)
                gmask[i] = 1.0;
        }
        Ptx gmask_pt = cc->MakeCKKSPackedPlaintext(gmask);
        result = cc->EvalMult(result, gmask_pt);

        if (g == 0)
            attn_ct = std::move(result);
        else
            cc->EvalAddInPlace(attn_ct, result);
    }

    // Scale by 1/sqrt(d_head)
    double scale = 1.0 / std::sqrt((double)(d / H));
    cc->EvalMultInPlace(attn_ct, scale);

    return attn_ct;
}

Ctx head_reduce_sum(Inference& inf, const Ctx& ct) {
    const CC& cc = inf.cc();
    int N  = inf.slots;
    int d  = inf.size.hidDim;
    int H  = inf.size.numHeads;
    int t  = N / d;
    int tH = t * H;

    Ctx out = ct->Clone();

    // Phase 1: intra-head reduction within t-blocks
    // TODO: precompute these masks in prepare_mha_masks and encode at a fixed level
    //       to avoid consuming ciphertext levels on plaintext-ct multiply
    for (int step = 1; step < t; step *= 2) {
        std::vector<double> mask_nw(N), mask_w(N);
        for (int i = 0; i < N; ++i) {
            mask_nw[i] = ((i % t) + step < t) ? 1.0 : 0.0;
            mask_w[i]  = 1.0 - mask_nw[i];
        }
        Ptx pt_nw = cc->MakeCKKSPackedPlaintext(mask_nw);
        Ptx pt_w  = cc->MakeCKKSPackedPlaintext(mask_w);

        Ctx rot_fwd  = cc->EvalRotate(out, mha_rot(inf, step));
        Ctx rot_wrap = cc->EvalRotate(out, mha_rot(inf, step - t));
        Ctx shifted  = cc->EvalAdd(cc->EvalMult(rot_fwd, pt_nw),
                                    cc->EvalMult(rot_wrap, pt_w));
        cc->EvalAddInPlace(out, shifted);
    }

    // Phase 2: inter-block aggregation
    for (int s = tH; s < N; s *= 2) {
        Ctx tmp = cc->EvalRotate(out, mha_rot(inf, s));
        cc->EvalAddInPlace(out, tmp);
    }

    return out;
}

Ctx attention_softmax(Inference& inf, const Ctx& scores,
                      int num_keys, double given_max,
                      const SoftmaxConfig& cfg) {
    const CC& cc = inf.cc();
    int N = inf.slots;
    int d = inf.size.hidDim;
    int H = inf.size.numHeads;
    int t = N / d;

    // neg_inf_mask: -given_max everywhere except valid token positions
    std::vector<double> ninf(N, -given_max);
    for (int h = 0; h < H; ++h)
        for (int tok = 0; tok < num_keys; ++tok)
            ninf[tok / t * t * H + h * t + tok % t] = 0.0;
    Ptx ninf_pt = cc->MakeCKKSPackedPlaintext(ninf);

    // scores + neg_inf_mask - given_max
    Ctx ct = cc->EvalAdd(scores, ninf_pt);
    cc->EvalSubInPlace(ct, given_max);

    // TODO: exp_r and gs_inv_iters should come from a config struct (like SoftmaxConfig)
    Ctx e = exp_approx(cc, ct, cfg.exp_r);

    // head_reduce_sum to broadcast denominator into every slot
    Ctx s = head_reduce_sum(inf, e);

    // TODO: initial estimate is a rough guess — tune based on actual value range
    double alpha = 1.0 / (double)num_keys;
    Ctx inv_init = encrypt_const(cc, alpha, (size_t)N, inf.fhe->pk());
    DepthGuard dg;
    Ctx inv_s = goldschmidt_inv(cc, s, inv_init, cfg.gs_inv_iters, dg);

    return cc->EvalMult(e, inv_s);
}

void prepare_vcache(Inference& inf) {
    int d_head = inf.size.hidDim / inf.size.numHeads;
    int N = inf.slots;

    // Initialize d_head zero ciphertexts as metaciphertext lanes
    std::vector<double> zeros(N, 0.0);
    Ctx zero_ct = encrypt(inf.cc(),
                          inf.cc()->MakeCKKSPackedPlaintext(zeros),
                          inf.fhe->pk());
    inf.cache["v"] = {};
    for (int i = 0; i < d_head; ++i)
        inf.cache["v"].push_back(zero_ct->Clone());

    inf.v_count = 0;
}

void cache_v_push(Inference& inf, const Ctx& value_ct) {
    const CC& cc = inf.cc();
    int N      = inf.slots;
    int d      = inf.size.hidDim;
    int H      = inf.size.numHeads;
    int t      = N / d;
    int tH     = t * H;
    int d_head = d / H;

    int right_rot = inf.v_count % t;

    Ctx v_rot = (right_rot == 0) ? value_ct->Clone()
              : cc->EvalRotate(value_ct, mha_rot(inf, -right_rot));

    for (int i = 0; i < d_head; ++i) {
        // TODO: precompute these masks to avoid level consumption
        std::vector<double> mask(N, 0.0);
        for (int h = 0; h < H; ++h)
            mask[i * tH + h * t + right_rot] = 1.0;
        Ptx mask_pt = cc->MakeCKKSPackedPlaintext(mask);
        Ctx tmp = cc->EvalMult(v_rot, mask_pt);

        int lane = ((inf.v_count / t) - i + d_head) % d_head;
        cc->EvalAddInPlace(inf.cache["v"][lane], tmp);
    }

    inf.v_count++;
}

Ctx softmax_v(Inference& inf, const Ctx& softmax_scores) {
    const CC& cc = inf.cc();
    int N      = inf.slots;
    int d      = inf.size.hidDim;
    int H      = inf.size.numHeads;
    int t      = N / d;
    int tH     = t * H;
    int d_head = d / H;

    Ctx scores = softmax_scores->Clone();
    Ctx res = cc->EvalMult(inf.cache["v"][0], scores);
    for (int i = 1; i < d_head; ++i) {
        scores = cc->EvalRotate(scores, mha_rot(inf, tH));
        Ctx tmp = cc->EvalMult(inf.cache["v"][i], scores);
        cc->EvalAddInPlace(res, tmp);
    }

    // Intra-token reduction
    for (int step = 1; step < t; step *= 2) {
        Ctx tmp = cc->EvalRotate(res, mha_rot(inf, step));
        cc->EvalAddInPlace(res, tmp);
    }

    // Mask to tok=0 positions
    res = cc->EvalMult(res, inf.mask["tok0"]);
    return res;
}
