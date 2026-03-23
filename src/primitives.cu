#include "ckks_primitives.h"
#include <cmath>
#include <functional>

static void eval_rescale(const CC& cc, Ctx& x) {
    cc->RescaleInPlace(x);
}

/// @brief Computes the inverse square root of a value using the Newton-Raphson iteration method.
/// @param cc, the crypto context
/// @param slots, the number of slots in the ciphertext TODO: this parameter is useless???
/// @param x, the input ciphertext for which we want to compute 1/sqrt(x)
/// @param ans_init, the initial guess for 1/sqrt(x)
/// @param iters, the number of iterations to perform
/// @return the i-th newton-raphson approximation of 1/sqrt(x)
/// @todo: this function is instead canonical... if we find out
///        that the goldschmidt method performs the same, we should replace
///        this function with the goldschmidt method, which should be better.
Ctx inv_sqrt_newton(const CC& cc, int slots,
                    const Ctx& x, const Ctx& ans_init, int iters) {
    Ctx ans = ans_init;
    
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
/// @param slots, the number of slots in the ciphertext TODO: this parameter is useless??
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
Ctx goldschmidt_inv_sqrt(const CC& cc, int slots,
                          const Ctx& x, const Ctx& ans_init, int iters) {
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
/// @param one_ct, a ciphertext encrypting the value 1, used for the iteration TODO: this too is useless wtf
/// @param res, the initial guess for the inverse, which will be updated in-place
/// @param dnm, the input ciphertext for which we want to compute the inverse, updated in-place
/// @param iters, the number of iterations to perform
/// @return the i-th Newton approximation of the inverse
/// @todo: differently from the goldschmidt approximation, this function on the other
///        hand assumes that the input ciphertexts can be updated in-place.
///        We should make the two functions consistent, no?
Ctx newton_inverse(const CC& cc, Ctx one_ct, Ctx res, Ctx dnm, int iters) {
    for (int i = 0; i < iters; ++i) {
        cc->EvalSquareInPlace(res);
        eval_rescale(cc, res);

        Ctx tmp = cc->EvalAdd(res, 1.0);

        match_level(cc, dnm, tmp);
        dnm = cc->EvalMult(dnm, tmp);
        eval_rescale(cc, dnm);
    }
    return dnm;
}

/// @brief Computes the inverse of a value using the Goldschmidt iteration method.
/// @param cc, the crypto context
/// @param slots, the number of slots in the ciphertext TODO: this parameter is useless too???
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
Ctx goldschmidt_inv(const CC& cc, int slots,
                    const Ctx& a, const Ctx& x0_init, int iters) {
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


/// @brief Computes the Chebyshev coefficients for approximating a
/// function f on the interval [a, b] with a polynomial of given degree.
/// @param f, the function to approximate
/// @param a, the lower bound of the interval
/// @param b, the upper bound of the interval
/// @param degree, the degree of the approximating polynomial
/// @return a vector containing the Chebyshev coefficients
std::vector<double> chebyshev_coeffs(
        std::function<double(double)> f, double a, double b, int degree) {
    int n = degree + 1;        // 128 nodes for degree=127
    double bma = 0.5 * (b - a);
    double bpa = 0.5 * (b + a);

    // Chebyshev nodes x_k = cos(pi*(k+0.5)/n) * (b-a)/2 + (a+b)/2
    std::vector<double> nodes(n);
    for (int k = 0; k < n; ++k)
        nodes[k] = std::cos(M_PI * (k + 0.5) / n) * bma + bpa;

    std::vector<double> fi(n);
    for (int i = 0; i < n; ++i)
        fi[i] = f(nodes[i]);

    std::vector<double> coeffs(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double u     = (2.0 * nodes[i] - (a + b)) / (b - a);
        double Tprev = 1.0;   // T_0(u)
        double T     = u;     // T_1(u)
        for (int j = 0; j < n; ++j) {
            coeffs[j] += fi[i] * Tprev;
            double Tnext = 2.0 * u * T - Tprev;
            Tprev = T;
            T     = Tnext;
        }
    }

    double two_over_n = 2.0 / n;
    for (int k = 0; k < n; ++k)
        coeffs[k] *= two_over_n;

    return coeffs;
}

/// @brief Evaluates a Chebyshev series at a given point.
/// @param cc, the crypto context 
/// @param x, the input ciphertext at which to evaluate the series
/// @param f, the function to approximate
/// @param a, the lower bound of the interval
/// @param b, the upper bound of the interval 
/// @param degree, the degree of the approximating polynomial
/// @return the ciphertext resulting from evaluating the Chebyshev series at x
/// @todo: this function recomputes the Chebyshev coefficients every time it is called.
///        We need a variant where the chebyshev coefficients are precomputed and
///        passed as input.
Ctx eval_chebyshev(const CC& cc, const Ctx& x,
                   std::function<double(double)> f,
                   double a, double b, int degree) {
    auto coeffs = chebyshev_coeffs(f, a, b, degree);

    double alpha = 2.0 / (b - a);
    double beta  = -(a + b) / (b - a);

    Ctx u;
    if (a == -1.0 && b == 1.0) {
        u = x;
    } else {
        u = cc->EvalMult(x, alpha);  
        cc->RescaleInPlace(u);         
        if (std::abs(beta) > 1e-15)
            cc->EvalAddInPlace(u, beta);
    }

    return cc->EvalChebyshevSeries(u, coeffs, -1.0, 1.0);
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
