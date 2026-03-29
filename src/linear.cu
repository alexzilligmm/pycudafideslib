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

Ctx linear(Inference& inf, const Ctx& x, const std::string& wname, int in_dim, int out_dim) {
    int numSlots = inf.slots;
    if(wname == "up"){
       int preProc = numSlots / in_dim
    } else if(wname == "down"){
        postProc = numSlots / out_dim
    } else {
        int preProc = numSlots / out_dim
    }
    


}

Ctx qkv_q(Inference& inf, const Ctx& x) { return linear(inf, x, "q", inf.size.hidDim); }
Ctx qkv_k(Inference& inf, const Ctx& x) { return linear(inf, x, "k", inf.size.hidDim); }
Ctx qkv_v(Inference& inf, const Ctx& x) { return linear(inf, x, "v", inf.size.hidDim); }

static Ctx rope_single(Inference& inf, const Ctx& x) {
    const CC& cc  = inf.cc();
    const int intRot = inf.slots / inf.size.hidDim;
    auto& w = inf.w.at("RoPE");  // [cos, sin+, sin-]

    Ctx xcos  = cc->EvalMult(x, w[0]);

    Ctx xsin0 = cc->EvalMult(x, w[1]);
    xsin0     = cc->EvalRotate(xsin0, rot(inf, intRot));

    Ctx xsin1 = cc->EvalMult(x, w[2]);
    xsin1     = cc->EvalRotate(xsin1, rot(inf, -intRot));

    Ctx y = cc->EvalAdd(xcos, xsin0);
    cc->EvalAddInPlace(y, xsin1);
    return y;
}

std::tuple<Ctx, Ctx> rope(Inference& inf,
                           const Ctx& q, const Ctx& k) {
    Ctx yq, yk;

    yq = rope_single(inf, q);
    yk = rope_single(inf, k);
    
    return {yq, yk};
}

void cache_kv(Inference& inf, const Ctx& k, const Ctx& v) {
    const CC& cc    = inf.cc();
    const int S     = inf.slots;
    const int hD    = inf.size.hidDim;
    const int nH    = inf.size.numHeads;
    const int seqL  = inf.size.seqLen;
    const int intRot = S / hD;
    const int intIdx = seqL % intRot;
    const int midIdx = seqL / intRot;

    auto& kCache = inf.cache.at("k");
    if (intIdx == 0) {
        kCache.push_back(k);
    } else {
        Ctx k_rot = cc->EvalRotate(k, rot(inf, intIdx));
        cc->EvalAddInPlace(kCache[midIdx], k_rot);
    }

    const int rot_idx = intIdx + S * nH * midIdx / hD;
    auto& vCache    = inf.cache.at("v");
    auto& cacheMask = inf.cache_mask;
    Ctx v_rot = cc->EvalRotate(v, rot(inf, rot_idx));
    const int hd = hD / nH;

    for (int i = 0; i < hd; ++i) {
        Ctx tmp = cc->EvalMult(v_rot, cacheMask[i]);
        cc->EvalAddInPlace(vCache[i], tmp);
    }
    
}

Ctx qk_transpose(Inference& inf, const Ctx& q) {
    const CC&   cc     = inf.cc();
    const auto& kCache = inf.cache.at("k");
    Ptx  kmask  = inf.mask.at("k");
    const int   S      = inf.slots;
    const int   hD     = inf.size.hidDim;
    const int   nH     = inf.size.numHeads;
    const int   nrot   = hD / nH;          // head dimension

    const int   inner_step = nH * S / hD;
    const int   seqL   = inf.size.seqLen;
    const int   space  = S * S / (seqL * hD);
    const int   n      = (int)kCache.size();

    std::vector<Ctx> results(n);

    auto do_block = [&](int i) {
        Ctx tmp = cc->EvalMult(q, kCache[i]);
        for (int j = 1; j < nrot; j *= 2) {
            Ctx rotated = cc->EvalRotate(tmp, rot(inf, inner_step * j));
            cc->EvalAddInPlace(tmp, rotated);
        }
        tmp = cc->EvalMult(tmp, kmask);
        if (i > 0) tmp = cc->EvalRotate(tmp, rot(inf, S - space * i));
        results[i] = std::move(tmp);
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
    Ptx  vmask  = inf.mask.at("v");
    const int   S      = inf.slots;
    const int   nH     = inf.size.numHeads;
    const int   hD     = inf.size.hidDim;
    const int   seqL   = inf.size.seqLen;
    const int   hd     = hD / nH;
    const int   inRot  = (int)std::sqrt((double)(hd / 2));
    const int   outRot = inRot * 2;
    const int   nv     = (int)vCache.size();
    const int   space  = S * nH / hD;

    Ctx sb = s;
    for (int step = nH * seqL; step < S; step *= 2)
        rotate_add_inplace(inf, sb, step);

    std::vector<Ctx> sRot(inRot);
    sRot[0] = sb;
    for (int i = 1; i < inRot; ++i)
        sRot[i] = cc->EvalRotate(sRot[i-1], rot(inf, space));

    std::vector<Ctx> partSum(nv);
    auto do_mult = [&](int i) {
        partSum[i] = cc->EvalMult(sRot[i % inRot], vCache[i]);
    };

    for (int i = 0; i < nv; ++i) do_mult(i);

    for (int i = 0; i < nv; ++i) {
        if (i % inRot > 0) {
            cc->EvalAddInPlace(partSum[i - i % inRot], partSum[i]);
        }
    }

    for (int i = 1; i < outRot; ++i) {
        partSum[i * inRot] = cc->EvalRotate(partSum[i * inRot], rot(inf, i * space * inRot));
        cc->EvalAddInPlace(partSum[0], partSum[i * inRot]);
    }

    for (int step = 1; step < S / hD; step *= 2)
        rotate_add_inplace(inf, partSum[0], step);

    Ctx result = cc->EvalMult(partSum[0], vmask);
    return result;
}

Ctx out_proj(Inference& inf, const Ctx& x) {
    return linear(inf, x, "out", inf.size.hidDim);
}

std::pair<Ctx, Ctx> up_proj(Inference& inf, const Ctx& x) {
    return linear(inf, x, "up", inf.size.hidDim, inf.size.ffDim)
}

Ctx down_proj(Inference& inf, const Ctx& x) {
    return linear(inf, x, "down", inf.size.ffDim, inf.size.hidDim);
}
