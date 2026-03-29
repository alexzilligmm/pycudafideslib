#include "gpt2.h"
#include <omp.h>
#include <cmath>
#include <stdexcept>

static constexpr int kBenchRot = 5;

static inline int rot(const Inference& inf, int real_idx) {
    return inf.bench_mode ? kBenchRot : real_idx;
}

void rotate_add_inplace(Inference& inf, Ctx& x, int step) {
    const CC& cc = inf.cc();
    Ctx tmp = cc->EvalRotate(x, rot(inf, step));
    cc->EvalAddInPlace(x, tmp);
}

//   expand == 0  ↔  d_in == d_out  (QKV, Out)
//   expand == 1  ↔  d_in < d_out   (Up, Gate)
//   expand == -1 ↔  d_in > d_out   (Down)
Ctx linear_interleaved(Inference& inf, const Ctx& x_in,
                       const std::string& wname, int d_in, int d_out) {
    const CC& cc = inf.cc();
    int numSlots = inf.slots;
    int hidDim   = inf.size.hidDim;
    int expDim   = inf.size.expDim;

    int expand = (d_in > d_out) ? -1 : (d_in < d_out) ? 1 : 0;
    int preProc, postProc, inRot, outRot;
    int intRot = numSlots / std::max(d_in, d_out);


    if (expand >= 0) {
        preProc = numSlots / hidDim;
    } else {
        preProc = numSlots / expDim;
    }
    if (expand <= 0) {
        postProc = numSlots / hidDim;
    } else {
        postProc = numSlots / expDim;
    }

    if (expand == 0) {
        inRot  = (int)std::sqrt((double)(hidDim * hidDim / (2 * numSlots)));
        outRot = hidDim * hidDim / (numSlots * inRot);
    } else {
        inRot  = (int)std::sqrt((double)(hidDim * expDim / (2 * numSlots)));
        outRot = hidDim * expDim / (numSlots * inRot);
    }

    int d_pre = (expand >= 0) ? hidDim : expDim;

    auto& weight = inf.w.at(wname);
    std::vector<Ctx> ctRot(inRot);
    std::vector<Ctx> partSum(weight.size());

    Ctx x = x_in->Clone();
    for (int i = 1; i < preProc; i *= 2) {
        Ctx tmp = cc->EvalRotate(x, rot(inf, i * (d_pre - 1)));
        cc->EvalAddInPlace(x, tmp);
    }

    ctRot[0] = x;
    for (int i = 1; i < inRot; ++i) {
        ctRot[i] = cc->EvalRotate(x, rot(inf, i * intRot));
    }

    for (int i = 0; i < (int)weight.size(); ++i) {
        partSum[i] = cc->EvalMult(ctRot[i % inRot], weight[i]);
        if (i % inRot > 0) {
            cc->EvalAddInPlace(partSum[i - i % inRot], partSum[i]);
        }
    }

    for (int i = 1; i < outRot; ++i) {
        partSum[i * inRot] = cc->EvalRotate(
            partSum[i * inRot],
            rot(inf, i * intRot * inRot));
    }

    for (int i = 1; i < outRot; ++i) {
        cc->EvalAddInPlace(partSum[0], partSum[i * inRot]);
    }

    for (int i = 1; i < postProc; i *= 2) {
        Ctx tmp = cc->EvalRotate(partSum[0], rot(inf, i));
        cc->EvalAddInPlace(partSum[0], tmp);
    }

    return partSum[0];
}

Ctx qkv_q(Inference& inf, const Ctx& x) {
    return linear_interleaved(inf, x, "q", inf.size.hidDim, inf.size.hidDim);
}
Ctx qkv_k(Inference& inf, const Ctx& x) {
    return linear_interleaved(inf, x, "k", inf.size.hidDim, inf.size.hidDim);
}
Ctx qkv_v(Inference& inf, const Ctx& x) {
    return linear_interleaved(inf, x, "v", inf.size.hidDim, inf.size.hidDim);
}

static Ctx rope_single(Inference& inf, const Ctx& x) {
    const CC& cc  = inf.cc();
    int numSlots = inf.slots;
    int hidDim   = inf.size.hidDim;
    int intRot   = numSlots / hidDim;
    auto& weight = inf.w.at("RoPE");  // [cos, sin+, sin-]

    Ctx xCos  = cc->EvalMult(x, weight[0]);

    Ctx xSin0 = cc->EvalMult(x, weight[1]);
    xSin0     = cc->EvalRotate(xSin0, rot(inf, intRot));

    Ctx xSin1 = cc->EvalMult(x, weight[2]);
    xSin1     = cc->EvalRotate(xSin1, rot(inf, -intRot));

    Ctx y = cc->EvalAdd(xCos, xSin0);
    cc->EvalAddInPlace(y, xSin1);
    return y;
}

std::tuple<Ctx, Ctx> rope(Inference& inf,
                           const Ctx& q, const Ctx& k) {
    Ctx yQ = rope_single(inf, q);
    Ctx yK = rope_single(inf, k);
    return {yQ, yK};
}

void cache_kv(Inference& inf, const Ctx& k, const Ctx& v) {
    const CC& cc     = inf.cc();
    int numSlots     = inf.slots;
    int hidDim       = inf.size.hidDim;
    int numHeads     = inf.size.numHeads;
    int seqLen       = inf.size.seqLen;
    int intRot       = numSlots / hidDim;
    int intIdx       = seqLen % intRot;
    int midIdx       = seqLen / intRot;

    auto& kCache = inf.cache.at("k");
    if (intIdx == 0) {
        kCache.push_back(k);
    } else {
        Ctx k_rot = cc->EvalRotate(k, rot(inf, intIdx));
        cc->EvalAddInPlace(kCache[midIdx], k_rot);
    }

    int rot_idx = intIdx + numSlots * numHeads * midIdx / hidDim;
    auto& vCache    = inf.cache.at("v");
    auto& cacheMask = inf.cache_mask;
    Ctx v_rot = cc->EvalRotate(v, rot(inf, rot_idx));

    for (int i = 0; i < hidDim / numHeads; ++i) {
        Ctx tmp = cc->EvalMult(v_rot, cacheMask[i]);
        cc->EvalAddInPlace(vCache[i], tmp);
    }
}

Ctx qk_transpose(Inference& inf, const Ctx& q) {
    const CC&   cc     = inf.cc();
    const auto& kCache = inf.cache.at("k");
    Ptx  mask   = inf.mask.at("k");
    int numSlots = inf.slots;
    int hidDim   = inf.size.hidDim;
    int numHeads = inf.size.numHeads;
    int seqLen   = inf.size.seqLen;
    int num_rot  = hidDim / numHeads;
    int space    = numSlots * numSlots / (seqLen * hidDim);
    int n        = (int)kCache.size();

    std::vector<Ctx> results(n);

    auto do_block = [&](int i) {
        Ctx ctTmp = cc->EvalMult(q, kCache[i]);
        for (int j = 1; j < num_rot; j *= 2) {
            Ctx tmp = cc->EvalRotate(ctTmp, rot(inf, numHeads * numSlots / hidDim * j));
            cc->EvalAddInPlace(ctTmp, tmp);
        }
        ctTmp = cc->EvalMult(ctTmp, mask);
        if (i > 0) ctTmp = cc->EvalRotate(ctTmp, rot(inf, numSlots - space * i));
        results[i] = std::move(ctTmp);
    };

    for (int i = 0; i < n; ++i) do_block(i);

    Ctx y = results[0];
    for (int i = 1; i < n; ++i) {
        cc->EvalAddInPlace(y, results[i]);
    }
    return y;
}

Ctx attn_v(Inference& inf, const Ctx& s) {
    const CC&   cc     = inf.cc();
    const auto& vCache = inf.cache.at("v");
    Ptx  mask    = inf.mask.at("v");
    int numSlots = inf.slots;
    int numHeads = inf.size.numHeads;
    int hidDim   = inf.size.hidDim;
    int seqLen   = inf.size.seqLen;
    int inRot    = (int)std::sqrt((double)(hidDim / (2 * numHeads)));
    int outRot   = inRot * 2;
    int nv       = (int)vCache.size();
    int space    = numSlots * numHeads / hidDim;

    Ctx sb = s;
    for (int step = numHeads * seqLen; step < numSlots; step *= 2)
        rotate_add_inplace(inf, sb, step);

    std::vector<Ctx> ctRot(inRot);
    ctRot[0] = sb;
    for (int i = 1; i < inRot; ++i)
        ctRot[i] = cc->EvalRotate(ctRot[i-1], rot(inf, space));

    std::vector<Ctx> partSum(nv);
    for (int i = 0; i < nv; ++i) {
        partSum[i] = cc->EvalMult(ctRot[i % inRot], vCache[i]);
    }

    for (int i = 0; i < nv; ++i) {
        if (i % inRot > 0) {
            cc->EvalAddInPlace(partSum[i - i % inRot], partSum[i]);
        }
    }

    for (int i = 1; i < outRot; ++i) {
        partSum[i * inRot] = cc->EvalRotate(partSum[i * inRot], rot(inf, i * space * inRot));
        cc->EvalAddInPlace(partSum[0], partSum[i * inRot]);
    }

    for (int step = 1; step < numSlots / hidDim; step *= 2)
        rotate_add_inplace(inf, partSum[0], step);

    Ctx y = cc->EvalMult(partSum[0], mask);
    return y;
}

Ctx out_proj(Inference& inf, const Ctx& x) {
    return linear_interleaved(inf, x, "out", inf.size.hidDim, inf.size.hidDim);
}

std::pair<Ctx, Ctx> up_gate(Inference& inf, const Ctx& x) {
    Ctx up   = linear_interleaved(inf, x, "up", inf.size.hidDim, inf.size.expDim);
    Ctx gate = linear_interleaved(inf, x, "gate", inf.size.hidDim, inf.size.expDim);
    return {up, gate};
}

Ctx down_proj(Inference& inf, const Ctx& x) {
    return linear_interleaved(inf, x, "down", inf.size.expDim, inf.size.hidDim);
}
