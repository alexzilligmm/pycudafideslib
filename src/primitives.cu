#include "ckks_primitives.h"
#include <cmath>
#include <map>
#include <functional>

static void eval_rescale(const CC& cc, Ctx& x) {
    cc->RescaleInPlace(x);
}

/// @brief Computes the inverse square root of a value using the Newton-Raphson iteration method.
/// @param cc, the crypto context
/// @param x, the input ciphertext for which we want to compute 1/sqrt(x)
/// @param ans_init, the initial guess for 1/sqrt(x)
/// @param iters, the number of iterations to perform
/// @return the i-th newton-raphson approximation of 1/sqrt(x)
/// @todo: this function is instead canonical... if we find out
///        that the goldschmidt method performs the same, we should replace
///        this function with the goldschmidt method, which should be better.
Ctx inv_sqrt_newton(const CC& cc, const Ctx& x, const Ctx& ans_init, int iters) {
    Ctx ans = ans_init;
    int slots = cc->GetRingDimension() / 2;
    
    for (int i = 0; i < iters; ++i) {
        Ctx ansSq = cc->EvalSquare(ans);
        eval_rescale(cc, ansSq);

        match_level(cc, ansSq, ans);
        Ctx ansCu = cc->EvalMult(ansSq, ans);
        eval_rescale(cc, ansCu);

        Ptx pt_neg05 = cc->MakeCKKSPackedPlaintext(std::vector<double>(slots, -0.5), 1, x->GetLevel());
        Ctx halfX = cc->EvalMult(x, pt_neg05);
        eval_rescale(cc, halfX);

        match_level(cc, ansCu, halfX);
        Ctx term1 = cc->EvalMult(ansCu, halfX);
        eval_rescale(cc, term1);

        Ptx pt_15 = cc->MakeCKKSPackedPlaintext(std::vector<double>(slots, 1.5), 1, ans->GetLevel());
        Ctx term2 = cc->EvalMult(ans, pt_15);
        eval_rescale(cc, term2);

        match_level(cc, term1, term2);
        ans = cc->EvalAdd(term1, term2);
    }
    return ans;
}

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
Ctx goldschmidt_inv_sqrt(const CC& cc, const Ctx& x, const Ctx& ans_init, int iters) {
    Ctx ans = ans_init;
    Ctx x_copy = x; 
    
    match_level(cc, x_copy, ans);
    Ctx sqrt_ct = cc->EvalMult(x_copy, ans);
    eval_rescale(cc, sqrt_ct);

    for (int i = 0; i < iters; ++i) {
        match_level(cc, sqrt_ct, ans);
        Ctx res = cc->EvalMult(sqrt_ct, ans);
        eval_rescale(cc, res);

        cc->EvalMultInPlace(res, -0.5);   
        cc->EvalAddInPlace(res, 1.5);     

        match_level(cc, sqrt_ct, res);
        sqrt_ct = cc->EvalMult(sqrt_ct, res);
        eval_rescale(cc, sqrt_ct);

        match_level(cc, ans, res);
        ans = cc->EvalMult(ans, res);
        eval_rescale(cc, ans);
    }
    return ans;
}

/// @brief Computes the inverse of a value using the Newton iteration method.
/// @param cc, the crypto context
/// @param res, the initial guess for the inverse, which will be updated in-place
/// @param dnm, the input ciphertext for which we want to compute the inverse, updated in-place
/// @param iters, the number of iterations to perform
/// @return the i-th Newton approximation of the inverse
Ctx newton_inverse(const CC& cc, const Ctx& res, const Ctx& dnm, int iters) {
    Ctx res_copy = res;

    for (int i = 0; i < iters; ++i) {
        match_level(cc, res_copy, dnm);
        Ctx a = cc->EvalMult(res_copy, dnm);
        cc->EvalNegateInPlace(a);
        cc->EvalAddInPlace(a, 2.0);
        eval_rescale(cc, a);

        match_level(cc, res_copy, a);
        res_copy = cc->EvalMult(res_copy, a);
        eval_rescale(cc, res_copy);
    }
    return res_copy;
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
Ctx goldschmidt_inv(const CC& cc, const Ctx& a, const Ctx& x0_init, int iters) {
    Ctx x0 = x0_init;  
    Ctx a_l = a;      

    match_level(cc, a_l, x0);  
    match_level(cc, x0, a_l);  

    Ctx E = cc->EvalMult(a_l, x0);
    eval_rescale(cc, E);

    E = cc->EvalNegate(E);           
    cc->EvalAddInPlace(E, 1.0);       

    for (int i = 0; i < iters; ++i) {
        Ctx factor = cc->EvalAdd(E, 1.0);

        match_level(cc, x0, factor);
        x0 = cc->EvalMult(x0, factor);
        eval_rescale(cc, x0);

        cc->EvalSquareInPlace(E);
        eval_rescale(cc, E);
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
Ctx exp_squaring(const CC& cc, Ctx x, int iters) {
    for (int i = 0; i < iters; ++i) {
        cc->EvalSquareInPlace(x);
        eval_rescale(cc, x);
    }
    return x;
}
