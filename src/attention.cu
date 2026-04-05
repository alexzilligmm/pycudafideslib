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

std::vector<std::vector<double>> rearrange_wo_weights(
        const std::vector<std::vector<double>>& W, int H) {
    int d_in  = (int)W.size();
    int d_out = (int)W[0].size();
    int d_head = d_in / H;
    std::vector<std::vector<double>> out(d_in, std::vector<double>(d_out));
    for (int r = 0; r < d_in; ++r) {
        int h  = r % H;
        int ld = r / H;
        int src_row = h * d_head + ld;
        out[r] = W[src_row];
    }
    return out;
}

void prepare_mha_masks(Inference& inf) {
    int N = inf.slots;
    int d = inf.size.hidDim;
    int t = N / d;
    inf.mask["tok0"] = inf.encode_stride_mask(d, t, 1.0);  
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
        inf.cache["k"].push_back(std::move(masked));
    } else {
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

    Ctx q = cc->EvalMult(query_ct, inf.mask["tok0"]);

    for (int step = 1; step < t; step *= 2) {
        Ctx tmp = cc->EvalRotate(q, mha_rot(inf, -step));
        cc->EvalAddInPlace(q, tmp);
    }

    int keys_per_ct = t;
    int num_groups  = (int)inf.cache["k"].size();
    Ctx attn_ct;

    for (int g = 0; g < num_groups; ++g) {
        int first_key = g * keys_per_ct;
        int num_tok   = std::min(keys_per_ct, inf.k_count - first_key);

        Ctx result = cc->EvalMult(q, inf.cache["k"][g]);

        for (int s = tH; s < N; s *= 2) {
            Ctx tmp = cc->EvalRotate(result, mha_rot(inf, s));
            cc->EvalAddInPlace(result, tmp);
        }

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

    double scale = 1.0 / std::sqrt((double)(d / H));
    cc->EvalMultInPlace(attn_ct, scale);

    return attn_ct;
}

Ctx head_reduce_sum(Inference& inf, const Ctx& ct,
                    const DepthGuard& dg) {
    const CC& cc = inf.cc();
    int N  = inf.slots;
    int d  = inf.size.hidDim;
    int H  = inf.size.numHeads;
    int t  = N / d;
    int tH = t * H;

    Ctx out = ct->Clone();

    int step_idx = 0;
    for (int step = 1; step < t; step *= 2, ++step_idx) {
        if (dg) { out = dg(out, step_idx); }

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

    auto maybe_btp = [&](Ctx& ct, const char* tag) {
        if (cfg.btp_min_remaining == 0) return;
        uint32_t consumed  = level_of(ct);
        uint32_t remaining = ((uint32_t)inf.total_depth > consumed)
                             ? ((uint32_t)inf.total_depth - consumed) : 0u;
        if (remaining < cfg.btp_min_remaining) {
            ct = bootstrap_to(inf, ct, cfg.btp_target_level); 
            std::cout << "  [attn_softmax] " << tag << " bootstrap, consumed="
                      << level_of(ct) << "\n";
        }
    };

    double mask_max = std::max(given_max, std::pow(2.0, cfg.exp_r - 1));

    std::vector<double> mask(N, -mask_max - given_max);
    for (int h = 0; h < H; ++h)
        for (int tok = 0; tok < num_keys; ++tok)
            mask[tok / t * t * H + h * t + tok % t] = -given_max;
    Ptx mask_pt = cc->MakeCKKSPackedPlaintext(mask);
    Ctx ct = cc->EvalAdd(scores, mask_pt);

    maybe_btp(ct, "pre-exp");
    Ctx e = exp_approx(cc, ct, cfg.exp_r);

    DepthGuard hrs_dg;
    if (!cfg.gs_btp_schedule.empty() || cfg.gs_btp_min_remaining > 0) {
        hrs_dg.refresh = [&](const Ctx& c) {
            return bootstrap_to(inf, c, (uint32_t)cfg.btp_target_level);
        };
        hrs_dg.total_depth   = (uint32_t)inf.total_depth;
        hrs_dg.min_remaining = cfg.gs_btp_min_remaining;
    }
    maybe_btp(e, "pre-head-reduce");
    Ctx s = head_reduce_sum(inf, e, hrs_dg);

    double avg_exp = 0.5 + 0.5 * std::exp(-2.0 * given_max);
    double alpha = 1.0 / ((double)num_keys * avg_exp);
    Ctx inv_init = encrypt_const(cc, alpha, (size_t)N, inf.fhe->pk());

    maybe_btp(s, "pre-inv");

    DepthGuard gs_dg;
    if (!cfg.gs_btp_schedule.empty() || cfg.gs_btp_min_remaining > 0) {
        gs_dg.refresh = [&](const Ctx& c) {
            return bootstrap_to(inf, c, (uint32_t)cfg.btp_target_level);
        };
        gs_dg.schedule      = cfg.gs_btp_schedule;
        gs_dg.total_depth   = (uint32_t)inf.total_depth;
        gs_dg.min_remaining = cfg.gs_btp_min_remaining;
    }
    Ctx inv_s = goldschmidt_inv(cc, s, inv_init, cfg.gs_inv_iters, gs_dg);
    maybe_btp(inv_s, "pre-final-mult-inv");
    maybe_btp(e, "pre-final-mult-exp");
    Ctx result = cc->EvalMult(e, inv_s);
    return result;
}

void prepare_vcache(Inference& inf) {
    int d_head = inf.size.hidDim / inf.size.numHeads;
    int N = inf.slots;

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
        // TODO: are these masks at the right level 
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

    for (int step = 1; step < t; step *= 2) {
        Ctx tmp = cc->EvalRotate(res, mha_rot(inf, step));
        cc->EvalAddInPlace(res, tmp);
    }

    res = cc->EvalMult(res, inf.mask["tok0"]);
    return res;
}
