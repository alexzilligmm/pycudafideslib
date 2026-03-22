#include "llama.h"
#include <omp.h>
#include <cmath>
#include <stdexcept>

// ── helpers ───────────────────────────────────────────────────────────────

void rotate_add_inplace(LlamaInference& llama, Ctx& x, int step) {
    const CC& cc = llama.cc();
    Ctx tmp = cc->EvalRotate(x, step);
    match_level(cc, x, tmp);
    cc->EvalAddInPlace(x, tmp);
}

Ctx linear(LlamaInference& llama, const Ctx& x,
           const std::string& wname, int expand) {
    const CC& cc     = llama.cc();
    auto& weight = llama.w.at(wname);
    const int  hidDim  = llama.size.hidDim;
    const int  expDim  = llama.size.expDim;
    const int  S       = llama.slots;

    int preProc  = (expand >= 0) ? S / hidDim : S / expDim;
    int postProc = (expand <= 0) ? S / hidDim : S / expDim;
    int inRot, outRot;
    if (expand == 0) {
        inRot  = (int)std::sqrt((double)(hidDim * hidDim) / (2 * S));
        outRot = hidDim * hidDim / (S * inRot);
    } else {
        inRot  = (int)std::sqrt((double)(hidDim * expDim) / (2 * S));
        outRot = hidDim * expDim / (S * inRot);
    }

    const int intRot = S / hidDim;

    const int preStep  = (expand >= 0) ? hidDim : expDim;
    const int postStep = (expand <= 0) ? hidDim : expDim;

    const int n_weights = (int)weight.size();

    Ctx xb = x;
    for (int step = preStep; step < S; step *= 2)
        rotate_add_inplace(llama, xb, step);

    std::vector<Ctx> ctRot(inRot);
    ctRot[0] = xb;
    for (int i = 1; i < inRot; ++i)
        ctRot[i] = cc->EvalRotate(ctRot[i-1], intRot);

    std::vector<Ctx> partSum(n_weights);

    auto do_mult = [&](int i) {
        partSum[i] = cc->EvalMult(ctRot[i % inRot], weight[i]);
    };

    if (llama.parallel) {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n_weights; ++i) do_mult(i);
    } else {
        for (int i = 0; i < n_weights; ++i) do_mult(i);
    }

    for (int i = 0; i < n_weights; ++i) {
        if (i % inRot > 0) {
            match_level(cc, partSum[i - i % inRot], partSum[i]);
            cc->EvalAddInPlace(partSum[i - i % inRot], partSum[i]);
        }
    }

    if (llama.parallel) {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 1; i < outRot; ++i)
            partSum[i * inRot] = cc->EvalRotate(partSum[i * inRot],
                                                  i * inRot * intRot);
    } else {
        for (int i = 1; i < outRot; ++i)
            partSum[i * inRot] = cc->EvalRotate(partSum[i * inRot],
                                                  i * inRot * intRot);
    }

    for (int i = 1; i < outRot; ++i) {
        match_level(cc, partSum[0], partSum[i * inRot]);
        cc->EvalAddInPlace(partSum[0], partSum[i * inRot]);
    }

    for (int step = postStep; step < S; step *= 2)
        rotate_add_inplace(llama, partSum[0], step);

    return partSum[0];
}

// ── QKV ──────────────────────────────────────────────────────────────────
Ctx qkv_q(LlamaInference& llama, const Ctx& x) { return linear(llama, x, "q",    0); }
Ctx qkv_k(LlamaInference& llama, const Ctx& x) { return linear(llama, x, "k",    0); }
Ctx qkv_v(LlamaInference& llama, const Ctx& x) { return linear(llama, x, "v",    0); }


static Ctx rope_single(LlamaInference& llama, const Ctx& x) {
    const CC& cc  = llama.cc();
    const int intRot = llama.slots / llama.size.hidDim;
    auto& w = llama.w.at("RoPE");  // [cos, sin+, sin-]

    Ctx xcos  = cc->EvalMult(x, w[0]);

    Ctx xsin0 = cc->EvalMult(x, w[1]);
    xsin0     = cc->EvalRotate(xsin0,  intRot);   // Go: numSlots / hidDim

    Ctx xsin1 = cc->EvalMult(x, w[2]);
    xsin1     = cc->EvalRotate(xsin1, -intRot);   // Go: -(numSlots / hidDim)

    match_level(cc, xcos, xsin0);
    Ctx y = cc->EvalAdd(xcos, xsin0);
    match_level(cc, y, xsin1);
    cc->EvalAddInPlace(y, xsin1);
    return y;
}

std::tuple<Ctx, Ctx> rope(LlamaInference& llama,
                           const Ctx& q, const Ctx& k) {
    Ctx yq, yk;
    if (llama.parallel) {
        #pragma omp parallel sections
        {
            #pragma omp section
            yq = rope_single(llama, q);
            #pragma omp section
            yk = rope_single(llama, k);
        }
    } else {
        yq = rope_single(llama, q);
        yk = rope_single(llama, k);
    }
    return {yq, yk};
}

// ── Cache ─────────────────────────────────────────────────────────────────
// TODO: cache_kv rotation amounts are runtime-dependent on seqLen position.
//   intRot  = S / hidDim
//   intIdx  = seqLen % intRot           (position within current block)
//   midIdx  = seqLen / intRot           (block index)
//   K-cache step = intIdx               (Go: eval.Rotate(k, intIdx, k))
//   V-cache step = intIdx + S * nH * midIdx / hD
//                                       (Go: eval.Rotate(v, rot_idx, v))
// Currently left at the placeholder until sequence-position tracking is added.
void cache_kv(LlamaInference& llama, const Ctx& k, const Ctx& v) {
    const CC& cc    = llama.cc();
    const int S     = llama.slots;
    const int hD    = llama.size.hidDim;
    const int nH    = llama.size.numHeads;
    const int seqL  = llama.size.seqLen;
    const int intRot = S / hD;
    const int intIdx = seqL % intRot;
    const int midIdx = seqL / intRot;

    // K cache
    auto& kCache = llama.cache.at("k");
    if (intIdx == 0) {
        kCache.push_back(k);
    } else {
        // Go: eval.Rotate(k, intIdx, k)
        Ctx k_rot = cc->EvalRotate(k, intIdx);
        match_level(cc, kCache[midIdx], k_rot);
        cc->EvalAddInPlace(kCache[midIdx], k_rot);
    }

    // V cache
    // Go: rot_idx = intIdx + numSlots * numHeads * midIdx / hidDim
    const int rot_idx = intIdx + S * nH * midIdx / hD;
    auto& vCache    = llama.cache.at("v");
    auto& cacheMask = llama.cache_mask;
    // TODO: rot_idx may not be in the registered key set for all seqLen values.
    //       For now use rot_idx; add it to fideslib_wrapper.h key generation if needed.
    Ctx v_rot = cc->EvalRotate(v, rot_idx);
    const int hd = hD / nH;

    if (llama.parallel) {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < hd; ++i) {
            Ctx tmp = cc->EvalMult(v_rot, cacheMask[i]);
            match_level(cc, vCache[i], tmp);
            cc->EvalAddInPlace(vCache[i], tmp);
        }
    } else {
        for (int i = 0; i < hd; ++i) {
            Ctx tmp = cc->EvalMult(v_rot, cacheMask[i]);
            match_level(cc, vCache[i], tmp);
            cc->EvalAddInPlace(vCache[i], tmp);
        }
    }
}

// ── QK^T ─────────────────────────────────────────────────────────────────
// TODO: inner rotation step = nH * S / hD  (= numHeads * numSlots / hidDim)
//       offset rotation for block i = numSlots - space * i
//       where space = numSlots^2 / (seqLen * hidDim)
// These may not be in the default key set; add to fideslib_wrapper.h if needed.
Ctx qk_transpose(LlamaInference& llama, const Ctx& q) {
    const CC&   cc     = llama.cc();
    const auto& kCache = llama.cache.at("k");
    Ptx  kmask  = llama.mask.at("k");
    const int   S      = llama.slots;
    const int   hD     = llama.size.hidDim;
    const int   nH     = llama.size.numHeads;
    const int   nrot   = hD / nH;          // head dimension
    // inner rotation step for summing over head dim (Go: j * numHeads * S / hD)
    const int   inner_step = nH * S / hD;
    // block offset step (Go: numSlots - space * i, space = S * S / (seqLen * hD))
    const int   seqL   = llama.size.seqLen;
    const int   space  = S * S / (seqL * hD);
    const int   n      = (int)kCache.size();

    std::vector<Ctx> results(n);

    auto do_block = [&](int i) {
        Ctx tmp = cc->EvalMult(q, kCache[i]);
        // Sum over head dimension
        for (int j = 1; j < nrot; j *= 2) {
            Ctx rotated = cc->EvalRotate(tmp, inner_step * (int)std::log2(j + 1));
            // TODO: inner_step may need key registration; simpler formula TBD
            match_level(cc, tmp, rotated);
            cc->EvalAddInPlace(tmp, rotated);
        }
        tmp = cc->EvalMult(tmp, kmask);
        if (i > 0) tmp = cc->EvalRotate(tmp, S - space * i);
        results[i] = std::move(tmp);
    };

    if (llama.parallel) {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n; ++i) do_block(i);
    } else {
        for (int i = 0; i < n; ++i) do_block(i);
    }

    Ctx y = results[0];
    for (int i = 1; i < n; ++i) {
        match_level(cc, y, results[i]);
        cc->EvalAddInPlace(y, results[i]);
    }
    return y;
}

// ── AttnV ────────────────────────────────────────────────────────────────
// TODO: space = numSlots * numHeads / hidDim
//   broadcast step = S / (nH * seqLen)  (Go: i / len(vCache))
//   inner step     = space              (Go: i * space)
//   outer step     = i * space * inRot  (Go: i * space * inRot)
Ctx attn_v(LlamaInference& llama, const Ctx& s) {
    const CC&   cc     = llama.cc();
    const auto& vCache = llama.cache.at("v");
    Ptx  vmask  = llama.mask.at("v");
    const int   S      = llama.slots;
    const int   nH     = llama.size.numHeads;
    const int   hD     = llama.size.hidDim;
    const int   seqL   = llama.size.seqLen;
    const int   hd     = hD / nH;
    const int   inRot  = (int)std::sqrt((double)(hd / 2));
    const int   outRot = inRot * 2;
    const int   nv     = (int)vCache.size();
    const int   space  = S * nH / hD;   // Go: numSlots * numHeads / hidDim

    // Broadcast s over sequence positions
    Ctx sb = s;
    for (int step = S / (nH * seqL); step < S; step *= 2)
        rotate_add_inplace(llama, sb, step);

    // Pre-rotate s (baby-step: i * space)
    std::vector<Ctx> sRot(inRot);
    sRot[0] = sb;
    for (int i = 1; i < inRot; ++i)
        sRot[i] = cc->EvalRotate(sRot[i-1], space);

    std::vector<Ctx> partSum(nv);
    auto do_mult = [&](int i) {
        partSum[i] = cc->EvalMult(sRot[i % inRot], vCache[i]);
    };

    if (llama.parallel) {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < nv; ++i) do_mult(i);
    } else {
        for (int i = 0; i < nv; ++i) do_mult(i);
    }

    // Input-sum within blocks
    for (int i = 0; i < nv; ++i) {
        if (i % inRot > 0) {
            match_level(cc, partSum[i - i % inRot], partSum[i]);
            cc->EvalAddInPlace(partSum[i - i % inRot], partSum[i]);
        }
    }

    // Output rotations + sum (giant-step: i * space * inRot)
    for (int i = 1; i < outRot; ++i) {
        partSum[i * inRot] = cc->EvalRotate(partSum[i * inRot], i * space * inRot);
        match_level(cc, partSum[0], partSum[i * inRot]);
        cc->EvalAddInPlace(partSum[0], partSum[i * inRot]);
    }

    // Inter-sum
    for (int step = S / hD; step < S; step *= 2)
        rotate_add_inplace(llama, partSum[0], step);

    return cc->EvalMult(partSum[0], vmask);
}

// ── Out / UpGate / Down ───────────────────────────────────────────────────
Ctx out_proj(LlamaInference& llama, const Ctx& x) { return linear(llama, x, "out",  0); }

std::pair<Ctx, Ctx> up_gate(LlamaInference& llama, const Ctx& x) {
    Ctx up_ct, gate_ct;
    if (llama.parallel) {
        #pragma omp parallel sections
        {
            #pragma omp section
            up_ct   = linear(llama, x, "up",   1);
            #pragma omp section
            gate_ct = linear(llama, x, "gate", 1);
        }
    } else {
        up_ct   = linear(llama, x, "up",   1);
        gate_ct = linear(llama, x, "gate", 1);
    }
    return {up_ct, gate_ct};
}

Ctx down_proj(LlamaInference& llama, const Ctx& x) { return linear(llama, x, "down", -1); }
