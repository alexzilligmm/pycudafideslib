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

Ctx linear(Inference& inf, const Ctx& x_in,
                       const std::string& wname, int d_in, int d_out) {
    const CC& cc = inf.cc();
    auto p = compute_cm_params(inf.slots, d_in, d_out);

    auto pts_W = inf.w.at(wname);

    Ctx x = x_in;
    for (int step = 1; step < p.tp_in; step *= 2) {
        Ctx tmp = cc->EvalRotate(x, rot(inf, step * (p.t - 1)));
        cc->EvalAddInPlace(x, tmp);
    }

    int rot2 = p.t * p.t;
    std::vector<Ctx> x_rotated(p.r_i);
    x_rotated[0] = x;
    for (int j = 1; j < p.r_i; ++j)
        x_rotated[j] = cc->EvalRotate(x, rot(inf, j * rot2));

    std::vector<Ctx> cy(p.r_o);
    for (int k = 0; k < p.r_o; ++k) {
        cy[k] = cc->EvalMult(x_rotated[0], pts_W[k]);
        for (int j = 1; j < p.r_i; ++j) {
            Ctx tmp = cc->EvalMult(x_rotated[j], pts_W[j * p.r_o + k]);
            cc->EvalAddInPlace(cy[k], tmp);
        }
    }

    int cascade_rot = p.t * p.tp;
    for (int k = p.r_o - 1; k > 0; --k) {
        Ctx tmp = cc->EvalRotate(cy[k], rot(inf, cascade_rot));
        cc->EvalAddInPlace(cy[k - 1], tmp);
    }

    Ctx y = cy[0];
    for (int step = 1; step < p.tp_out; step *= 2)
        rotate_add_inplace(inf, y, step);

    return y;
}

Ctx qkv_q(Inference& inf, const Ctx& x) {
    return linear(inf, x, "q", inf.size.hidDim, inf.size.hidDim);
}
Ctx qkv_k(Inference& inf, const Ctx& x) {
    return linear(inf, x, "k", inf.size.hidDim, inf.size.hidDim);
}
Ctx qkv_v(Inference& inf, const Ctx& x) {
    return linear(inf, x, "v", inf.size.hidDim, inf.size.hidDim);
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

Ctx out_proj(Inference& inf, const Ctx& x) {
    return linear(inf, x, "out", inf.size.hidDim, inf.size.hidDim);
}

std::pair<Ctx, Ctx> up_gate(Inference& inf, const Ctx& x) {
    Ctx up   = linear(inf, x, "up", inf.size.hidDim, inf.size.expDim);
    Ctx gate = linear(inf, x, "gate", inf.size.hidDim, inf.size.expDim);
    return {up, gate};
}

Ctx down_proj(Inference& inf, const Ctx& x) {
    return linear(inf, x, "down", inf.size.expDim, inf.size.hidDim);
}
