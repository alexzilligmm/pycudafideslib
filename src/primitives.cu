#include "ckks_primitives.h"
#include <cmath>
#include <map>
#include <functional>

/// @brief Computes the inverse square root of a value using the Newton-Raphson iteration method.
/// @param cc, the crypto context
/// @param x, the input ciphertext for which we want to compute 1/sqrt(x)
/// @param ans_init, the initial guess for 1/sqrt(x)
/// @param iters, the number of iterations to perform
/// @return the i-th newton-raphson approximation of 1/sqrt(x)
/// @todo: this function is instead canonical... if we find out
///        that the goldschmidt method performs the same, we should replace
///        this function with the goldschmidt method, which should be better.
Ctx inv_sqrt_newton(const CC& cc, const Ctx& x, const Ctx& ans_init, int iters,
                    const DepthGuard& dg) {
    int slots = cc->GetRingDimension() / 2;

    Ptx pt_05 = encode_const(cc, 0.5, slots, x->GetLevel());
    Ctx c = cc->EvalMult(x, pt_05);

    Ctx ct = ans_init;

    for (int i = 0; i < iters; ++i) {
        if (dg) { ct = dg(ct, i); c = dg(c, i); }

        Ptx pt_15 = encode_const(cc, 1.5, slots, ct->GetLevel());

        Ctx t = cc->EvalMult(ct, pt_15);

        Ctx y2 = cc->EvalSquare(ct);

        Ctx y3 = cc->EvalMult(y2, ct);

        Ctx s = cc->EvalMult(c, y3);

        ct = cc->EvalSub(t, s);
    }
    return ct;
}

/// TODO: seems to consume four levels while it should use 3?
/// @brief Computes the inverse square root of a value using the Goldschmidt iteration method.
/// @param cc, the crypto context
/// @param x, the input ciphertext for which we want to compute 1/sqrt(x)
/// @param ans_init, the initial guess for 1/sqrt(x)
/// @param iters, the number of iterations to perform
/// @return the i-th goldschmidt approximation of 1/sqrt(x)
/// @todo: this function seems iffy because the naming seems wrong, since it
///        is actually computing almost the same algorithm as newton-raphson.
///        The second iffy part is that, after the scalar mult with -0.5, there
///        is no rescale, which is a bit strange. I assume the reasoning for the 
///        lack of a rescale is that the value is then multiplied with sqrt_ct and
///        ans, which will both be rescaled, so the rescale will be implicitly
///        applied there? We should thoroughly check this, because it promises
///        to cost just 2 levels, and to also parallelize those multiplications.
///        Which I am wary to believe.
Ctx goldschmidt_inv_sqrt(const CC& cc, const Ctx& x, const Ctx& ans_init, int iters,
                         const DepthGuard& dg) {
    Ctx ans = ans_init->Clone();
    Ctx x_copy = x;

    Ctx sqrt_ct = cc->EvalMult(x_copy, ans);

    for (int i = 0; i < iters; ++i) {
        if (dg) { ans = dg(ans, i); sqrt_ct = dg(sqrt_ct, i); }

        Ctx res = cc->EvalMult(sqrt_ct, ans);

        cc->EvalMultInPlace(res, -0.5);
        cc->EvalAddInPlace(res, 1.5);

        sqrt_ct = cc->EvalMult(sqrt_ct, res);

        ans = cc->EvalMult(ans, res);
    }
    return ans;
}

/// @brief Computes the inverse of a value using the Newton iteration method.
/// @param cc, the crypto context
/// @param res, the initial guess for the inverse
/// @param dnm, the ciphertext for which we want to compute 1/dnm
/// @param iters, the number of iterations to perform
/// @return the i-th Newton approximation of 1/dnm
///
/// Depth budget: 2 levels per iteration.
///   y_{i+1} = y_i · (2 − d·y_i) = 2·y_i − d·y_i²
///
/// The "2·y − d·y²" form keeps the constant operation (subtraction) AFTER
/// all multiplications.  This avoids an EvalAdd between two dependent ct×ct
/// mults, which in FLEXIBLEAUTO would force an eager rescale and waste
/// one level per iteration.
///
/// Critical path per iteration:
///   y² (1 square, deferred) → d·y² (1 ct×ct)   = depth 2 from y
///   2·y (1 scalar, parallel)                     = depth 1 from y
///   EvalSub resolves both; result sits at depth 2.
Ctx newton_inverse(const CC& cc, const Ctx& res, Ctx dnm, int iters,
                   const DepthGuard& dg) {
    Ctx y = res;
    for (int i = 0; i < iters; ++i) {
        if (dg) { y = dg(y, i); dnm = dg(dnm, i); }

        Ctx y2    = cc->EvalSquare(y);        // y²       (deferred rescale)
        Ctx dy2   = cc->EvalMult(dnm, y2);    // d·y²     (resolves y², 1 ct×ct)
        Ctx two_y = cc->EvalMult(y, 2.0);     // 2·y      (scalar, parallel path)
        y = cc->EvalSub(two_y, dy2);           // 2·y − d·y²
    }
    return y;
}

/// @brief Computes the inverse of a value using the Goldschmidt iteration method.
/// @param cc, the crypto context
/// @param a, the ciphertext a, for which we want to compute 1/a
/// @param x0_init, the initial guess for 1/a
/// @param iters, the number of iterations to perform
/// @return the i-th goldschmidt approximation of 1/a
/// @todo: this function assumes the inputs could be reused after the call,
///        as it makes copies and avoids touching the inputs. If we want
///        to ensure maximum performance, we could allow in-place updates
///        of the inputs!
///        The reason why it would make sense, is because during runtime
///        we will likely never reuse something that was used as input to this
///        function.
Ctx goldschmidt_inv(const CC& cc, const Ctx& a, const Ctx& x0_init, int iters,
                    const DepthGuard& dg) {
    Ctx x0 = x0_init;
    Ctx a_l = a;

    Ctx E = cc->EvalMult(a_l, x0);

    E = cc->EvalNegate(E);
    cc->EvalAddInPlace(E, 1.0);

    for (int i = 0; i < iters; ++i) {
        if (dg) { x0 = dg(x0, i); E = dg(E, i); }

        Ctx factor = cc->EvalAdd(E, 1.0);

        x0 = cc->EvalMult(x0, factor);

        cc->EvalSquareInPlace(E);
    }

    return x0;
}

/// @brief Computes a weighted sum of ciphertexts, where the weights are given as plaintexts.
/// @param cc, the crypto context
/// @param cts, the vector of ciphertexts to be summed 
/// @param weights, the vector of plaintext weights corresponding to each ciphertext
/// @return the resulting ciphertext of the weighted sum
/// @todo: this function assumes input cyphertexts are all of the same shape.
///       We should add checks to ensure this is the case, and throw an error if not.
Ctx eval_linear_wsum(const CC& cc,
                     std::vector<Ctx>& cts,
                     const std::vector<double>& weights) {
    Ctx result = cc->EvalMult(cts[0], weights[0]);
    for (size_t i = 1; i < cts.size(); ++i) {
        Ctx term = cc->EvalMult(cts[i], weights[i]);
        cc->EvalAddInPlace(result, term);
    }
    return result;
}

/// @brief Helper function to repeatedly square a ciphertext
/// @param cc, the crypto context
/// @param x, the input ciphertext to be squared
/// @param iters, the number of times to square the ciphertext
/// @return the resulting ciphertext after repeated squaring
Ctx exp_squaring(const CC& cc, Ctx x, int iters,
                 const DepthGuard& dg) {
    for (int i = 0; i < iters; ++i) {
        if (dg) { x = dg(x, i); }
        cc->EvalSquareInPlace(x);
    }
    return x;
}


/// @brief Masks the first active_dim * (slots / active_dim) slots, zeroing the rest.
Ctx mask_slots(const CC& cc, const Ctx& x, int slots, int active_dim) {
    int intRot = slots / active_dim;
    int active = active_dim * intRot;
    if (active >= slots) return x;

    std::vector<double> mask_vec(slots, 0.0);
    for (int i = 0; i < active; ++i) mask_vec[i] = 1.0;
    Ptx mask = cc->MakeCKKSPackedPlaintext(
        mask_vec, /*noiseScaleDeg=*/1, (uint32_t)level_of(x));
    return cc->EvalMult(x, mask);
}

/// @brief compute encrypted average of a vector of ciphertexts
Ctx compute_average(Inference& inf, const Ctx& x_in) {

    const CC& cc = inf.cc();
    const int S  = inf.slots;
    const int hD = inf.size.hidDim;

    Ctx mean = x_in->Clone();
    for (int i = S / hD; i < S; i *= 2) {
        Ctx tmp = cc->EvalRotate(mean, i);
        cc->EvalAddInPlace(mean, tmp);
    }

    return mean;
}

/// @brief Computes the encrypted variance across hidDim slots: (1/N)*sum((x-mean)^2)
/// @param inf, the inference context (provides cc, slots, hidDim)
/// @param x_in, the input ciphertext
/// @param mean, a precomputed mean ciphertext (as returned by compute_average)
/// @return a ciphertext holding the variance replicated across all slots
///
/// Uses realHidDim (unpadded dimension) for scaling so that zero-padded
/// slots (e.g. positions 768-1023 when 768→1024) don't bias the result.
/// The rotation pattern uses hD (padded) because slot layout has period hD.
Ctx compute_variance(Inference& inf, const Ctx& x_in, Ctx mean) {
    const CC& cc  = inf.cc();
    const int S   = inf.slots;
    const int hD  = inf.size.hidDim;           // padded (BSGS block size)
    const int rD  = inf.size.getRealHidDim();  // unpadded (true vector length)

    // Scale x by the real dimension so that (rD*x - mean) = rD*(x - mean/rD)
    Ctx xd = cc->EvalMult(x_in, (double)rD);

    drop_levels(cc, mean, 1);
    Ctx varc = cc->EvalSub(xd, mean); // rD*(x - mean)

    // Zero padded slots before squaring.
    // With padding (rD < hD) the standard mask_slots does nothing because
    // hD evenly divides S.  Build a within-block mask that keeps only the
    // first rD elements out of every hD-sized block (intRot-spaced layout).
    if (rD < hD) {
        const int t = S / hD;           // intRot (replication factor)
        std::vector<double> m(S, 0.0);
        for (int j = 0; j < rD; ++j)    // real elements only
            for (int r = 0; r < t; ++r)  // all replicas within the block
                m[j * t + r] = 1.0;
        Ptx pm = cc->MakeCKKSPackedPlaintext(
            m, /*noiseScaleDeg=*/1, (uint32_t)level_of(varc));
        varc = cc->EvalMult(varc, pm);
    } else {
        varc = mask_slots(cc, varc, S, hD);
    }

    cc->EvalSquareInPlace(varc);

    // Sum across all hD positions (rotation pattern matches intRot-spaced layout)
    for (int i = S / hD; i < S; i *= 2) {
        Ctx tmp = cc->EvalRotate(varc, i);
        cc->EvalAddInPlace(varc, tmp);
    }

    // 1/rD^3 * sum(rD^2*(x-mean)^2) = variance
    double inv_d3 = 1.0 / std::pow((double)rD, 3.0);
    cc->EvalMultInPlace(varc, inv_d3);

    return varc;
}

/// @brief Computes the encrypted variance, deriving the mean internally.
Ctx compute_variance(Inference& inf, const Ctx& x_in) {
    return compute_variance(inf, x_in, compute_average(inf, x_in));
}
