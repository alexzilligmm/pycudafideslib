// nonlinear.cu – SiLU, Softmax, Norm, Argmax
// All FHE ops delegate to FIDESlib (OpenFHE + GPU).

#include "llama.h"
#include <cmath>
#include <vector>
#include <iostream>
#include <functional>

// declared in bootstrap.cu
Ctx bootstrap_to(const LlamaInference&, const Ctx&, uint32_t);

// ── Chebyshev coefficient computation ────────────────────────────────────
// Approximate f on [a,b] with a deg-n Chebyshev expansion.
static std::vector<double> cheby_coeffs(std::function<double(double)> f,
                                         double a, double b, int n) {
    std::vector<double> c(n, 0.0);
    for (int k = 0; k < n; ++k) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            double xj = std::cos(M_PI * (2.0 * j + 1) / (2.0 * n));
            double x  = 0.5 * ((b - a) * xj + (a + b));
            sum += f(x) * std::cos(k * std::acos(xj));
        }
        c[k] = (k == 0 ? 1.0 : 2.0) * sum / n;
    }
    return c;
}

// ── SiLU ─────────────────────────────────────────────────────────────────

static void init_silu(LlamaInference& llama) {
    auto fn = [](double x) { return x / (std::exp(-x) + 1.0); };
    llama.silu_func.cheby_coeffs = cheby_coeffs(fn, -20.0, 20.0, 128);
    llama.silu_func.ready = true;
}

Ctx silu(LlamaInference& llama, const Ctx& x) {
    if (!llama.silu_func.ready) init_silu(llama);
    std::cout << "Computing SiLU...\n";
    Timer t;
    // FIDESlib/OpenFHE EvalChebyshevSeries evaluates the polynomial on the GPU
    // using the baby-step giant-step algorithm.
    // The polynomial is expressed in Chebyshev basis on [-20, 20].
    // We first normalise the input to [-1, 1].
    const CC& cc = llama.cc();
    const double a = -20.0, b = 20.0;
    double inv_half = 2.0 / (b - a);
    double shift    = -(a + b) / (b - a);

    // t = x * inv_half + shift  (linear map to [-1,1])
    Ptx pt_scale = encode_const(cc, inv_half, (size_t)llama.slots,
                                  (int)level_of(x));
    Ctx tn = cc->EvalMult(x, pt_scale);
    Ptx pt_shift = encode_const(cc, shift, (size_t)llama.slots,
                                  (int)level_of(tn));
    cc->EvalAddInPlace(tn, cc->MakeCKKSPackedPlaintext(
        std::vector<double>(llama.slots, shift)));

    Ctx y = cc->EvalChebyshevSeries(tn, llama.silu_func.cheby_coeffs, -1.0, 1.0);
    std::cout << "  SiLU done in " << t.elapsed_s()
              << " s, level " << level_of(x) << " → " << level_of(y) << "\n";
    return y;
}

// ── Softmax ───────────────────────────────────────────────────────────────
// exp via (1 + x/128)^{2^{7+temp}} → sum → bootstrap → Newton inverse.
// target_level_after_btp: depth to consume after bootstrap before Newton.

Ctx softmax(const LlamaInference& llama, const Ctx& x_in,
             int target_level_after_btp, int temp) {
    const CC& cc = llama.cc();
    std::cout << "Computing Softmax...\n";
    Timer t;

    Ctx x = x_in;

    // ── exp via squaring trick ────────────────────────────────────────────
    Ptx inv128 = cc->MakeCKKSPackedPlaintext(
        std::vector<double>(llama.slots, 0.0078125));  // 1/128
    x = cc->EvalMult(x, inv128);

    // x = 1 + x/128
    cc->EvalAddInPlace(x, cc->MakeCKKSPackedPlaintext(
        std::vector<double>(llama.slots, 1.0)));

    const int iters = 7 + temp;
    for (int i = 0; i < iters; ++i)
        x = cc->EvalSquare(x);

    Ctx exp_x = x;

    // ── sum over sequence slots ──────────────────────────────────────────
    for (int j = 1; j < 256; j *= 2) {
        Ctx tmp = cc->EvalRotate(exp_x, 1024 / j);
        match_level(cc, exp_x, tmp);
        cc->EvalAddInPlace(exp_x, tmp);
    }

    std::cout << "  exp+sum done in " << t.elapsed_s() << " s\n";
    t.reset();

    // ── bootstrap sum ─────────────────────────────────────────────────────
    Ctx sum_btp = bootstrap_to(llama, exp_x, (uint32_t)target_level_after_btp);

    // Scale by 0.1 before Newton iterations
    cc->EvalMultInPlace(sum_btp, cc->MakeCKKSPackedPlaintext(
        std::vector<double>(llama.slots, 0.1)));

    // ── Newton iterations for 1/sum ─────────────────────────────────────
    // Encode constant ciphertext with value 1
    Ctx one_ct = cc->Encrypt(llama.fhe->pk(),
        cc->MakeCKKSPackedPlaintext(std::vector<double>(llama.slots, 1.0)));

    Ctx res = cc->EvalSub(one_ct, sum_btp);
    Ctx dnm = cc->EvalAdd(one_ct, res);

    for (int i = 0; i < 9; ++i) {
        res = cc->EvalSquare(res);
        match_level(cc, one_ct, res);
        Ctx tmp = cc->EvalAdd(res, one_ct);
        dnm = cc->EvalMult(dnm, tmp);
    }

    // softmax = exp(x) * dnm
    match_level(cc, exp_x, dnm);
    Ctx y = cc->EvalMult(exp_x, dnm);

    // Final scale down by 0.1
    cc->EvalMultInPlace(y, cc->MakeCKKSPackedPlaintext(
        std::vector<double>(llama.slots, 0.1)));

    std::cout << "  softmax done in " << t.elapsed_s()
              << " s, output level " << level_of(y) << "\n";
    return y;
}

// ── Norm (Layer Normalization) ────────────────────────────────────────────
Ctx norm(const LlamaInference& llama, const Ctx& x_in,
          int target_level_after_btp) {
    const CC& cc   = llama.cc();
    const int S    = llama.slots;
    const int hD   = llama.size.hidDim;
    std::cout << "Computing Norm...\n";
    Timer t;

    // ── mean: sum x over hidDim slots ────────────────────────────────────
    Ctx mean = x_in;
    for (int i = S / hD; i < S; i *= 2)
        rotate_add_inplace(llama, mean, i);

    // ── variance: (d*x - mean)^2 / d^3 ──────────────────────────────────
    Ctx xd = cc->EvalMult(x_in, cc->MakeCKKSPackedPlaintext(
        std::vector<double>(S, (double)hD)));

    match_level(cc, mean, xd);
    Ctx varc = cc->EvalSub(xd, mean);
    varc = cc->EvalSquare(varc);

    for (int i = S / hD; i < S; i *= 2)
        rotate_add_inplace(llama, varc, i);

    double inv_d3 = 1.0 / std::pow((double)hD, 3.0);
    cc->EvalMultInPlace(varc, cc->MakeCKKSPackedPlaintext(
        std::vector<double>(S, inv_d3)));

    // ── initial approx of 1/sqrt(varc): linear fit ───────────────────────
    const double slope = -1.29054537e-04;
    const double bias  =  1.29054537e-01;

    Ctx ans = cc->EvalMult(varc, cc->MakeCKKSPackedPlaintext(
        std::vector<double>(S, slope)));
    cc->EvalAddInPlace(ans, cc->MakeCKKSPackedPlaintext(
        std::vector<double>(S, bias)));

    // ── Newton iterations (4 rounds) ─────────────────────────────────────
    // ans_{n+1} = ans_n * (1.5 - 0.5 * varc * ans_n^2)
    for (int iter = 0; iter < 4; ++iter) {
        Ctx ans_sq = cc->EvalSquare(ans);
        match_level(cc, varc, ans_sq);
        Ctx term1  = cc->EvalMult(varc, ans_sq);
        term1 = cc->EvalMult(term1, cc->MakeCKKSPackedPlaintext(
            std::vector<double>(S, -0.5)));
        match_level(cc, ans, term1);
        Ctx term2 = cc->EvalMult(ans, cc->MakeCKKSPackedPlaintext(
            std::vector<double>(S, 1.5)));
        match_level(cc, term1, term2);
        ans = cc->EvalAdd(term1, term2);
    }
    std::cout << "  Newton done in " << t.elapsed_s() << " s\n";

    // ── bootstrap before Goldschmidt ─────────────────────────────────────
    ans = bootstrap_to(llama, ans, (uint32_t)target_level_after_btp);
    t.reset();

    // ── Goldschmidt iterations (2 rounds) ────────────────────────────────
    // Computes 1/sqrt(varc) more accurately.
    match_level(cc, varc, ans);
    Ctx sqrt_ct = cc->EvalMult(varc, ans);  // varc * (1/sqrt(varc)) ≈ sqrt(varc)

    for (int iter = 0; iter < 2; ++iter) {
        Ctx res = cc->EvalMult(sqrt_ct, ans);
        cc->EvalSubInPlace(res, cc->MakeCKKSPackedPlaintext(
            std::vector<double>(S, 2.0)));
        match_level(cc, sqrt_ct, res);
        sqrt_ct = cc->EvalMult(sqrt_ct, res);
        ans     = cc->EvalMult(ans, res);
    }
    // ans ≈ 1/sqrt(varc) after Goldschmidt

    // ── final: y = x / sqrt(var) ─────────────────────────────────────────
    match_level(cc, x_in, ans);
    Ctx y = cc->EvalMult(x_in, ans);

    std::cout << "  Norm done in " << t.elapsed_s()
              << " s, output level " << level_of(y) << "\n";
    return y;
}

// ── Argmax (under construction, mirrors Go stub) ──────────────────────────
Ctx argmax(LlamaInference& llama, const Ctx& x_in) {
    const CC& cc = llama.cc();
    Ctx logit    = softmax(llama, x_in, 14, 3);
    Ctx copy     = logit;

    for (int j = 1; j < 256; j *= 2) {
        Ctx tmp = cc->EvalRotate(logit, 1024 / j);
        match_level(cc, logit, tmp);
        cc->EvalAddInPlace(logit, tmp);
    }

    logit = bootstrap_to(llama, logit, 0);

    Ctx one_ct = cc->Encrypt(llama.fhe->pk(),
        cc->MakeCKKSPackedPlaintext(std::vector<double>(llama.slots, 1.0)));

    Ctx res = cc->EvalSub(one_ct, logit);
    Ctx dnm = cc->EvalAdd(one_ct, res);
    for (int i = 0; i < 12; ++i) {
        res = cc->EvalSquare(res);
        match_level(cc, one_ct, res);
        Ctx tmp = cc->EvalAdd(res, one_ct);
        dnm = cc->EvalMult(dnm, tmp);
    }

    match_level(cc, copy, dnm);
    Ctx y = cc->EvalMult(copy, dnm);

    // Index plaintext
    std::vector<double> idx(llama.slots);
    for (int i = 0; i < llama.slots; ++i)
        idx[i] = (double)((i / 8) % 16) - 8.0;
    cc->EvalMultInPlace(y, cc->MakeCKKSPackedPlaintext(idx));

    for (int i = 1; i < 256; i *= 2) {
        Ctx tmp = cc->EvalRotate(y, 1024 / i);
        match_level(cc, y, tmp);
        cc->EvalAddInPlace(y, tmp);
    }
    return y;
}
