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

Ctx linear_interleaved(Inference& inf, const Ctx& x,
                       const std::string& wname, int d_in, int d_out) {
    if (d_out <= 0) d_out = d_in;
    const CC& cc     = inf.cc();
    auto& weight     = inf.w.at(wname);
    const int S      = inf.slots;
    const int t_in   = S / d_in;
    const int t_out  = S / d_out;
    const int n_pt   = (int)weight.size(); 

    int r_i = std::max(1, (int)std::floor(std::sqrt((double)n_pt)));
    while (n_pt % r_i != 0 && r_i > 1) --r_i;
    const int r_o = n_pt / r_i;

    const int t_baby  = std::max(t_in, t_out);
    const int t_giant = t_baby * r_i;

    Ctx cxp = x;
    for (int step = 1; step < t_in; step *= 2)
        rotate_add_inplace(inf, cxp, step * (d_in - 1));

    std::vector<Ctx> cx_rot(r_i);
    cx_rot[0] = cxp;
    for (int j = 1; j < r_i; ++j)
        cx_rot[j] = cc->EvalRotate(cx_rot[j-1], rot(inf, t_baby));

    std::vector<Ctx> cy_k(r_o);
    std::vector<bool> cy_k_init(r_o, false);

    auto do_mult = [&](int j, int k) {
        int idx = j * r_o + k;
        Ctx prod = cc->EvalMult(cx_rot[j], weight[idx]);
        if (!cy_k_init[k]) {
            cy_k[k] = prod;
            cy_k_init[k] = true;
        } else {
            cc->EvalAddInPlace(cy_k[k], prod);
        }
    };

    if (inf.parallel) {
        for (int j = 0; j < r_i; ++j) {
            std::vector<Ctx> prods(r_o);
            #pragma omp parallel for schedule(dynamic)
            for (int k = 0; k < r_o; ++k)
                prods[k] = cc->EvalMult(cx_rot[j], weight[j * r_o + k]);
            for (int k = 0; k < r_o; ++k) {
                if (!cy_k_init[k]) { cy_k[k] = prods[k]; cy_k_init[k] = true; }
                else cc->EvalAddInPlace(cy_k[k], prods[k]);
            }
        }
    } else {
        for (int j = 0; j < r_i; ++j)
            for (int k = 0; k < r_o; ++k)
                do_mult(j, k);
    }

    for (int k = r_o - 1; k > 0; --k) {
        Ctx rotated = cc->EvalRotate(cy_k[k], rot(inf, t_giant));
        cc->EvalAddInPlace(cy_k[k-1], rotated);
    }

    Ctx result = cy_k[0];

    for (int step = 1; step < t_out; step *= 2)
        rotate_add_inplace(inf, result, step);

    return result;
}

Ctx qkv_q(Inference& inf, const Ctx& x) { return linear_interleaved(inf, x, "q", inf.size.hidDim); }
Ctx qkv_k(Inference& inf, const Ctx& x) { return linear_interleaved(inf, x, "k", inf.size.hidDim); }
Ctx qkv_v(Inference& inf, const Ctx& x) { return linear_interleaved(inf, x, "v", inf.size.hidDim); }

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
    if (inf.parallel) {
        #pragma omp parallel sections
        {
            #pragma omp section
            yq = rope_single(inf, q);
            #pragma omp section
            yk = rope_single(inf, k);
        }
    } else {
        yq = rope_single(inf, q);
        yk = rope_single(inf, k);
    }
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

    std::cout << "    [cache_kv] v level=" << level_of(v)
              << " vCache[0] level=" << level_of(vCache[0]) << std::endl;

    if (inf.parallel) {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < hd; ++i) {
            Ctx tmp = cc->EvalMult(v_rot, cacheMask[i]);
            cc->EvalAddInPlace(vCache[i], tmp);
        }
    } else {
        for (int i = 0; i < hd; ++i) {
            Ctx tmp = cc->EvalMult(v_rot, cacheMask[i]);
            cc->EvalAddInPlace(vCache[i], tmp);
        }
    }
    std::cout << "    [cache_kv] after update vCache[0] level=" << level_of(vCache[0]) << std::endl;
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

    if (inf.parallel) {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; ++i) do_block(i);
    } else {
        for (int i = 0; i < n; ++i) do_block(i);
    }

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

    std::cout << "    [attn_v] input level=" << level_of(s)
              << " nv=" << nv << " inRot=" << inRot << " outRot=" << outRot
              << " vCache[0] level=" << level_of(vCache[0]) << std::endl;

    Ctx sb = s;
    for (int step = nH * seqL; step < S; step *= 2)
        rotate_add_inplace(inf, sb, step);
    std::cout << "    [attn_v] after broadcast level=" << level_of(sb) << std::endl;

    std::vector<Ctx> sRot(inRot);
    sRot[0] = sb;
    for (int i = 1; i < inRot; ++i)
        sRot[i] = cc->EvalRotate(sRot[i-1], rot(inf, space));
    std::cout << "    [attn_v] after pre-rotate sRot[0] level=" << level_of(sRot[0]) << std::endl;

    std::vector<Ctx> partSum(nv);
    auto do_mult = [&](int i) {
        partSum[i] = cc->EvalMult(sRot[i % inRot], vCache[i]);
    };

    if (inf.parallel) {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < nv; ++i) do_mult(i);
    } else {
        for (int i = 0; i < nv; ++i) do_mult(i);
    }
    std::cout << "    [attn_v] after mults partSum[0] level=" << level_of(partSum[0]) << std::endl;

    for (int i = 0; i < nv; ++i) {
        if (i % inRot > 0) {
            cc->EvalAddInPlace(partSum[i - i % inRot], partSum[i]);
        }
    }
    std::cout << "    [attn_v] after input-sum level=" << level_of(partSum[0]) << std::endl;

    for (int i = 1; i < outRot; ++i) {
        partSum[i * inRot] = cc->EvalRotate(partSum[i * inRot], rot(inf, i * space * inRot));
        cc->EvalAddInPlace(partSum[0], partSum[i * inRot]);
    }
    std::cout << "    [attn_v] after output-sum level=" << level_of(partSum[0]) << std::endl;

    for (int step = 1; step < S / hD; step *= 2)
        rotate_add_inplace(inf, partSum[0], step);
    std::cout << "    [attn_v] after inter-sum level=" << level_of(partSum[0]) << std::endl;

    Ctx result = cc->EvalMult(partSum[0], vmask);
    std::cout << "    [attn_v] after vmask mult level=" << level_of(result) << std::endl;
    return result;
}

Ctx out_proj(Inference& inf, const Ctx& x) {
    return linear_interleaved(inf, x, "out", inf.size.hidDim);
}

std::pair<Ctx, Ctx> up_gate(Inference& inf, const Ctx& x) {
    const int hD = inf.size.hidDim, fD = inf.size.ffDim;
    Ctx up_ct, gate_ct;
    if (inf.parallel) {
        #pragma omp parallel sections
        {
            #pragma omp section
            up_ct   = linear_interleaved(inf, x, "up",   hD, fD);
            #pragma omp section
            gate_ct = linear_interleaved(inf, x, "gate", hD, fD);
        }
    } else {
        up_ct   = linear_interleaved(inf, x, "up",   hD, fD);
        gate_ct = linear_interleaved(inf, x, "gate", hD, fD);
    }
    return {up_ct, gate_ct};
}

Ctx down_proj(Inference& inf, const Ctx& x) {
    return linear_interleaved(inf, x, "down", inf.size.ffDim, inf.size.hidDim);
}
