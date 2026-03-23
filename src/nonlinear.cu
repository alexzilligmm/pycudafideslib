#include "llama.h"
#include "ckks_primitives.h"
#include <cmath>
#include <vector>
#include <functional>
#include <iostream>
#include <iomanip>

Ctx bootstrap_to(LlamaInference&, const Ctx&, uint32_t);

static Ptx const_pt(const CC& cc, double val, int slots, uint32_t level) {
    return encode_const(cc, val, (size_t)slots, (int)level);
}


static void dbg_lvl(const char* label, const Ctx& ct) {
    std::cout << "  [DBG] " << label
              << "  consumed=" << level_of(ct) << "\n";
}

static void dbg_val(const char* label, const CC& cc,
                    Ctx ct,                          // by value (copy for decrypt)
                    const PrivateKey<DCRTPoly>& sk) {
    auto v = decrypt(cc, ct, sk);
    std::cout << "  [DBG] " << label
              << "  consumed=" << level_of(ct)
              << "  vals=[";
    for (int i = 0; i < std::min((int)v.size(), 4); ++i)
        std::cout << std::setprecision(6) << v[i]
                  << (i < 3 ? ", " : "");
    std::cout << "]\n";
}

/// @brief SiLU in the Cachemir's implementation
Ctx silu(LlamaInference& llama, const Ctx& x_in) {
    const CC& cc = llama.cc();
    auto fn = [](double x) { return x / (std::exp(-x) + 1.0); };
    return eval_chebyshev_f(cc, x_in, fn, -20.0, 20.0, 127);
}

Ctx silu_expDim(LlamaInference& llama, const Ctx& x_in) {
    const CC& cc = llama.cc();
    const int S  = llama.slots;

    Ctx y = silu(llama, x_in);

    std::vector<double> mask_vec(S, 0.0);
    for (int i = 0; i < llama.size.expDim; ++i) {
        int idx = i * S / llama.size.expDim;
        if (idx < S) mask_vec[idx] = 1.0;
    }
    Ptx pt_mask = cc->MakeCKKSPackedPlaintext(
        mask_vec, /*noiseScaleDeg=*/1, (uint32_t)level_of(y));

    return cc->EvalMult(y, pt_mask);
}

/// @brief Softmax in the Cachemir's implementation
/// @param temp Additional squarings such that the computed function is actually
///             exp(x * 2^temp) instead of exp(x). To add, as the name suggests, a temperature effect.
Ctx softmax(LlamaInference& llama, const Ctx& x_in,
             int target_level_after_btp, int temp) {
    const CC& cc = llama.cc();
    const int S  = llama.slots;
    const auto& sk = llama.fhe->sk();

    Ctx x = x_in;

    // These guys are approximating exp(x) via the 
    // canonical limit definition: exp(x) = lim_{n→∞} (1 + x/n)^n.
    // They choose n=128=2^7, so they need 7 squarings to get the final result.
    Ptx pt_inv128 = const_pt(cc, 0.0078125, S, level_of(x));
    cc->EvalMultInPlace(x, pt_inv128);
    cc->RescaleInPlace(x);

    Ptx pt_one = const_pt(cc, 1.0, S, level_of(x));
    cc->EvalAddInPlace(x, pt_one);

    for (int i = 0; i < (7 + temp); ++i) {
        cc->EvalSquareInPlace(x);
        cc->RescaleInPlace(x);
    }

    Ctx exp_ct = cc->EvalAdd(x, 0.0);
    // TODO: why do they make a copy like so?
    // can't they just
    //   Ctx exp_ct = x; ???

    const double v = 8.0 / S;   // = 1/256 for S=2048
    cc->EvalMultInPlace(x, v);
    cc->RescaleInPlace(x);      // NL → 1

    // Trick to compute total sum in log-time
    for (int j = 1; j < 256; j *= 2) {
        Ctx tmp = cc->EvalRotate(x, 1024 / j);
        cc->EvalAddInPlace(x, tmp);
    }
    
    x = bootstrap_to(llama, x, (uint32_t)target_level_after_btp);

    // TODO, understand: x right now is the total sum, right?
    // if so then dnm should be 2 - x right now no?
    Ctx res = cc->EvalNegate(x);         // NL=2 (or auto-rescale from bootstrap NL=2)
    cc->RescaleInPlace(res);             // NL→1
    cc->EvalAddInPlace(res, 1.0);        // res = 1 - v*sum ≈ 0
    Ctx dnm = cc->EvalAdd(res, 1.0);     // dnm ≈ 1  (will converge to 1/(v*sum))

    // TODO This just goldschmidt for 7 iterations, why not calling the primitive?
    for (int i = 0; i < 7; ++i) {
        cc->EvalSquareInPlace(res);
        cc->RescaleInPlace(res);

        Ctx tmp = cc->EvalAdd(res, 1.0);

        match_level(cc, dnm, tmp);
        dnm = cc->EvalMult(dnm, tmp);
        cc->RescaleInPlace(dnm);

    }

    { // Rescaling exp_ct to match dnm's level, so we can multiply them together.
      // TODO: Why the heck do we need to do this iteratively like this? Don't we have 
      // a 'dropToLevel()' or something?
        uint32_t target = level_of(dnm);
        int iters = 0;
        while (level_of(exp_ct) < target) {
            uint32_t cur = level_of(exp_ct);
            std::vector<double> ones(S, 1.0);
            Ptx pt_one2 = cc->MakeCKKSPackedPlaintext(ones, 1, (uint32_t)cur);
            cc->EvalMultInPlace(exp_ct, pt_one2);
            cc->RescaleInPlace(exp_ct);
            ++iters;
        }
    }


    Ctx y = cc->EvalMult(exp_ct, dnm);
    cc->RescaleInPlace(y);

    cc->EvalMultInPlace(y, v);   // algebraically: exp * (1/(v*sum)) * v = exp/sum
    cc->RescaleInPlace(y);

    return y;
}


Ctx norm(LlamaInference& llama, const Ctx& x_in,
          int target_level_after_btp) {
    const CC& cc = llama.cc();
    const int S  = llama.slots;
    const int hD = llama.size.hidDim;
    const auto& sk = llama.fhe->sk();

    Ctx mean = cc->EvalAdd(x_in, 0.0); // TODO: again making copy like so, wtf
    for (int i = S / hD; i < S; i *= 2) {
        Ctx tmp = cc->EvalRotate(mean, i);
        cc->EvalAddInPlace(mean, tmp);
    }

    Ctx xd = cc->EvalMult(x_in, (double)hD);
    cc->RescaleInPlace(xd);

    drop_levels(cc, mean, 1);
    Ctx varc = cc->EvalSub(xd, mean); // so doing N*(X - mean)

    cc->EvalSquareInPlace(varc); // it is variance so makes sense
    cc->RescaleInPlace(varc);

    for (int i = S / hD; i < S; i *= 2) {
        Ctx tmp = cc->EvalRotate(varc, i);
        cc->EvalAddInPlace(varc, tmp);
    }

    double inv_d3 = 1.0 / std::pow((double)hD, 3.0);
    cc->EvalMultInPlace(varc, inv_d3); // 1/N^3 * sum(N^2*(X - mean)^2) = 1/N * sum((X - mean)^2) = variance
    cc->RescaleInPlace(varc);
    
    // --- Newton-Raphson for inverse square root ---
    // TODO: Again a function without calling the primitive? Why?
    // TODO: why is this not using the goldschmidt_inv_sqrt primitive if first place?
    //      I would expect that both inv_sqrts are pretty much equal, both are second order methods...

    // Do we know the range of varc? Otherwise this
    // initialization of the guess could be fairly sketchy.
    Ctx ans = cc->EvalMult(varc, -42.1);
    cc->RescaleInPlace(ans);
    cc->EvalAddInPlace(ans, 7.37);

    Ctx halfX = cc->EvalMult(varc, -0.5);
    cc->RescaleInPlace(halfX);

    for (int i = 0; i < 4; ++i) {
        Ctx ansSq = cc->EvalSquare(ans);
        cc->RescaleInPlace(ansSq);

        drop_levels(cc, ans, 1);

        Ctx ansCu = cc->EvalMult(ansSq, ans);
        cc->RescaleInPlace(ansCu);

        uint32_t halfX_target = level_of(ansCu);

        Ctx term1 = cc->EvalMult(ansCu, halfX);
        cc->RescaleInPlace(term1);

        Ctx term2 = cc->EvalMult(ans, 1.5);
        cc->RescaleInPlace(term2);

        match_level(cc, term2, term1);

        ans = cc->EvalAdd(term1, term2);
    }
    // --- End Newton-Raphson ---

    ans = bootstrap_to(llama, ans, (uint32_t)target_level_after_btp);

    Ctx varc_r = varc;
    reduce_to_level(cc, varc_r, level_of(ans), S);

    // And now we are calling the primitive... eh?
    ans = goldschmidt_inv_sqrt(cc, S, varc_r, ans, 2);

    Ctx x_copy = x_in;
    reduce_to_level(cc, x_copy, level_of(ans), S);

    Ctx y = cc->EvalMult(x_copy, ans);
    cc->RescaleInPlace(y);

    return y;
}

Ctx argmax(LlamaInference& llama, const Ctx& x_in) {
    const CC& cc = llama.cc();
    const int S  = llama.slots;

    Ctx logit = softmax(llama, x_in, 14, 3); // temp > 0 is needed to make the distribution more peaky

    Ctx sum = cc->EvalAdd(logit, 0.0); // TODO: again with copying like so
    for (int j = 1; j < 256; j *= 2) {
        Ctx tmp = cc->EvalRotate(sum, 1024 / j);
        cc->EvalAddInPlace(sum, tmp);
    }

    sum = bootstrap_to(llama, sum, 16);


    // --- Goldschmidt for inverse ---
    // TODO: WHYYYYY?! Why not just call the primitive?
    Ctx res = cc->EvalNegate(sum);
    cc->RescaleInPlace(res);       
    cc->EvalAddInPlace(res, 1.0);
    Ctx dnm = cc->EvalAdd(res, 1.0);

    for (int i = 0; i < 12; ++i) {
        cc->EvalSquareInPlace(res);
        cc->RescaleInPlace(res);

        Ctx tmp = cc->EvalAdd(res, 1.0);
        match_level(cc, dnm, tmp);
        dnm = cc->EvalMult(dnm, tmp);
        cc->RescaleInPlace(dnm);
    }
    // --- End Goldschmidt for inverse ---

    Ctx logit_copy = cc->EvalAdd(logit, 0.0); // TODO: i dunno what to say anymore.
    reduce_to_level(cc, logit_copy, level_of(dnm), S);
    Ctx y = cc->EvalMult(logit_copy, dnm); // TODO: Soo we doing probabilities * (1/sum of probabilities) = probabilities, right?
    cc->RescaleInPlace(y);

    // Actual argmax trick, if we have high enough temp
    // then the softmax distribution with almost be one-hot encoding,
    // and thus we can do this stuff to get the index of the max element.
    std::vector<double> idx_vec(S, 0.0);
    for (int i = 0; i < S; ++i)
        idx_vec[i] = (double)i;
    Ptx idx = cc->MakeCKKSPackedPlaintext(idx_vec, 1, level_of(y));
    cc->EvalMultInPlace(y, idx);
    cc->RescaleInPlace(y);

    for (int i = 1; i < 256; i *= 2) {
        Ctx tmp = cc->EvalRotate(y, 1024 / i);
        cc->EvalAddInPlace(y, tmp);
    }

    return y;
}
