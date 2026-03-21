// linear.cu – port of linear.go using FIDESlib (OpenFHE + GPU).
// All operations (EvalMult, EvalAdd, EvalRotate) run on GPU via FIDESlib.
// FLEXIBLEAUTO scaling mode handles rescaling automatically after EvalMult.

#include "llama.h"
#include <omp.h>
#include <cmath>
#include <stdexcept>

// ── helpers ───────────────────────────────────────────────────────────────

void rotate_add_inplace(const LlamaInference& llama, Ctx& x, int step) {
    const CC& cc = llama.cc();
    Ctx tmp = cc->EvalRotate(x, step);
    match_level(cc, x, tmp);
    cc->EvalAddInPlace(x, tmp);
}

// ── Linear ────────────────────────────────────────────────────────────────
// Port of (*LlamaInference).Linear from linear.go.
// Diagonal-style matrix-vector multiply:
//   preProc inter-rotations → inRot inner rotations → weight multiply →
//   outRot outer rotations → postProc inter-sum.

Ctx linear(const LlamaInference& llama, const Ctx& x,
           const std::string& wname, int expand) {
    const CC& cc     = llama.cc();
    const auto& weight = llama.w.at(wname);
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

    const int n_weights = (int)weight.size();

    // ── preProc inter-broadcast ───────────────────────────────────────────
    Ctx xb = x;
    for (int i = 1; i < preProc; i *= 2)
        rotate_add_inplace(llama, xb, 5);

    // ── inner rotations ───────────────────────────────────────────────────
    std::vector<Ctx> ctRot(inRot);
    ctRot[0] = xb;
    for (int i = 1; i < inRot; ++i)
        ctRot[i] = cc->EvalRotate(ctRot[i-1], 5);

    // ── multiply weights + partial sums ───────────────────────────────────
    std::vector<Ctx> partSum(n_weights);

    auto do_mult = [&](int i) {
        partSum[i] = cc->EvalMult(ctRot[i % inRot], weight[i]);
        // FLEXIBLEAUTO handles rescaling inside EvalMult
    };

    if (llama.parallel) {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < n_weights; ++i) do_mult(i);
    } else {
        for (int i = 0; i < n_weights; ++i) do_mult(i);
    }

    // Input-sum: accumulate within each inRot block (sequential to avoid races)
    for (int i = 0; i < n_weights; ++i) {
        if (i % inRot > 0) {
            match_level(cc, partSum[i - i % inRot], partSum[i]);
            cc->EvalAddInPlace(partSum[i - i % inRot], partSum[i]);
        }
    }

    // ── outer rotations ───────────────────────────────────────────────────
    if (llama.parallel) {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 1; i < outRot; ++i)
            partSum[i * inRot] = cc->EvalRotate(partSum[i * inRot], 5);
    } else {
        for (int i = 1; i < outRot; ++i)
            partSum[i * inRot] = cc->EvalRotate(partSum[i * inRot], 5);
    }

    // Output sum
    for (int i = 1; i < outRot; ++i) {
        match_level(cc, partSum[0], partSum[i * inRot]);
        cc->EvalAddInPlace(partSum[0], partSum[i * inRot]);
    }

    // ── postProc inter-sum ────────────────────────────────────────────────
    for (int i = 1; i < postProc; i *= 2)
        rotate_add_inplace(llama, partSum[0], 5);

    return partSum[0];
}

// ── QKV ──────────────────────────────────────────────────────────────────
Ctx qkv_q(const LlamaInference& llama, const Ctx& x) { return linear(llama, x, "q",    0); }
Ctx qkv_k(const LlamaInference& llama, const Ctx& x) { return linear(llama, x, "k",    0); }
Ctx qkv_v(const LlamaInference& llama, const Ctx& x) { return linear(llama, x, "v",    0); }

// ── RoPE ─────────────────────────────────────────────────────────────────
static Ctx rope_single(const LlamaInference& llama, const Ctx& x) {
    const CC& cc  = llama.cc();
    const auto& w = llama.w.at("RoPE");  // [cos, sin+, sin-]

    Ctx xcos  = cc->EvalMult(x, w[0]);

    Ctx xsin0 = cc->EvalMult(x, w[1]);
    xsin0     = cc->EvalRotate(xsin0, 5);

    Ctx xsin1 = cc->EvalMult(x, w[2]);
    xsin1     = cc->EvalRotate(xsin1, 5);

    match_level(cc, xcos, xsin0);
    Ctx y = cc->EvalAdd(xcos, xsin0);
    match_level(cc, y, xsin1);
    cc->EvalAddInPlace(y, xsin1);
    return y;
}

std::tuple<Ctx, Ctx> rope(const LlamaInference& llama,
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
        Ctx k_rot = cc->EvalRotate(k, 5);
        match_level(cc, kCache[midIdx], k_rot);
        cc->EvalAddInPlace(kCache[midIdx], k_rot);
    }

    // V cache
    auto& vCache    = llama.cache.at("v");
    auto& cacheMask = llama.cache_mask;
    Ctx v_rot = cc->EvalRotate(v, 5);
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
Ctx qk_transpose(const LlamaInference& llama, const Ctx& q) {
    const CC&   cc     = llama.cc();
    const auto& kCache = llama.cache.at("k");
    const Ptx&  kmask  = llama.mask.at("k");
    const int   nrot   = llama.size.hidDim / llama.size.numHeads;
    const int   n      = (int)kCache.size();

    std::vector<Ctx> results(n);

    auto do_block = [&](int i) {
        Ctx tmp = cc->EvalMult(q, kCache[i]);
        // Sum over head dimension
        for (int j = 1; j < nrot; j *= 2) {
            Ctx rotated = cc->EvalRotate(tmp, 5);
            match_level(cc, tmp, rotated);
            cc->EvalAddInPlace(tmp, rotated);
        }
        tmp = cc->EvalMult(tmp, kmask);
        if (i > 0) tmp = cc->EvalRotate(tmp, 5);
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
Ctx attn_v(const LlamaInference& llama, const Ctx& s) {
    const CC&   cc     = llama.cc();
    const auto& vCache = llama.cache.at("v");
    const Ptx&  vmask  = llama.mask.at("v");
    const int   S      = llama.slots;
    const int   nH     = llama.size.numHeads;
    const int   hD     = llama.size.hidDim;
    const int   seqL   = llama.size.seqLen;
    const int   hd     = hD / nH;
    const int   inRot  = (int)std::sqrt((double)(hd / 2));
    const int   outRot = inRot * 2;
    const int   nv     = (int)vCache.size();

    // Broadcast s over sequence positions
    Ctx sb = s;
    for (int i = nH * seqL; i < S; i *= 2)
        rotate_add_inplace(llama, sb, 5);

    // Pre-rotate s
    std::vector<Ctx> sRot(inRot);
    sRot[0] = sb;
    for (int i = 1; i < inRot; ++i)
        sRot[i] = cc->EvalRotate(sRot[i-1], 5);

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

    // Output rotations + sum
    for (int i = 1; i < outRot; ++i) {
        partSum[i * inRot] = cc->EvalRotate(partSum[i * inRot], 5);
        match_level(cc, partSum[0], partSum[i * inRot]);
        cc->EvalAddInPlace(partSum[0], partSum[i * inRot]);
    }

    // Inter-sum
    for (int i = 1; i < S / hD; i *= 2)
        rotate_add_inplace(llama, partSum[0], 5);

    return cc->EvalMult(partSum[0], vmask);
}

// ── Out / UpGate / Down ───────────────────────────────────────────────────
Ctx out_proj(const LlamaInference& llama, const Ctx& x) { return linear(llama, x, "out",  0); }

std::pair<Ctx, Ctx> up_gate(const LlamaInference& llama, const Ctx& x) {
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

Ctx down_proj(const LlamaInference& llama, const Ctx& x) { return linear(llama, x, "down", -1); }
