#include "ckks_primitives.h"
#include <cmath>
#include <map>
#include <functional>
#include <vector>

static void eval_rescale(const CC& cc, Ctx& x) {
    cc->RescaleInPlace(x);
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

/// @brief Evaluates a Chebyshev series of a function at a given point.
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
Ctx eval_chebyshev_f(const CC& cc, const Ctx& x,
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


/// @brief Converts coefficients from standard monomial basis to Chebyshev basis.
/// @param poly_coeffs Coefficients in standard basis [a0, a1, a2, ..., an]
/// @param a, the lower bound of the interval
/// @param b, the upper bound of the interval
/// @return Coefficients in Chebyshev basis [c0, c1, c2, ..., cn]
std::vector<double> standard_to_chebyshev(const std::vector<double>& poly_coeffs, double a, double b) {
    size_t n = poly_coeffs.size();
    if (n == 0) return {};

    double alpha = (b - a) / 2.0;
    double beta  = (a + b) / 2.0;

    std::vector<double> cheb_coeffs(n, 0.0);

    for (int i = n - 1; i >= 0; --i) {
        std::vector<double> next_cheb(n, 0.0);
        for (size_t j = 0; j < n; ++j) {
            if (cheb_coeffs[j] == 0) continue;

            if (j == 0) {
                next_cheb[1] += cheb_coeffs[j] * alpha;
            } else {
                if (j + 1 < n) next_cheb[j + 1] += 0.5 * cheb_coeffs[j] * alpha;
                next_cheb[j - 1] += 0.5 * cheb_coeffs[j] * alpha;
            }

            next_cheb[j] += cheb_coeffs[j] * beta;
        }
        next_cheb[0] += poly_coeffs[i];
        cheb_coeffs = next_cheb;
    }
    return cheb_coeffs;
}

/// @brief: Evaluates the Chebyshev series of a list of precomputed coefficients at a given point.
/// @param cc, the crypto context
/// @param x, the input ciphertext at which to evaluate the series
/// @param coeffs, the precomputed Chebyshev coefficients for the function to approximate
/// @param a, the lower bound of the interval
/// @param b, the upper bound of the interval
/// @return the ciphertext resulting from evaluating the Chebyshev series at x
Ctx eval_chebyshev(const CC& cc, const Ctx& x, std::vector<double> coeffs, double a, double b) {
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

/// @brief Evaluates a polynomial at a given point using Horner's method.
/// @param cc, the crypto context
/// @param x, the input ciphertext at which to evaluate the polynomial
/// @param coeffs, the coefficients of the polynomial in standard basis [a0, a1, a2, ..., an] representing a0 + a1*x + a2*x^2 + ... + an*x^n
/// @return the ciphertext resulting from evaluating the polynomial at x
/// @brief Evaluates a polynomial at a given point using Horner's method.
/// @param cc, the crypto context
/// @param x, the input ciphertext at which to evaluate the polynomial
/// @param coeffs, the coefficients of the polynomial in standard basis [a0, a1, a2, ..., an] representing a0 + a1*x + a2*x^2 + ... + an*x^n
/// @return the ciphertext resulting from evaluating the polynomial at x
Ctx eval_polynomial(const CC& cc, const Ctx& x, const std::vector<double>& coeffs) {
    size_t n = coeffs.size();
    Ctx result = cc->EvalMult(x, coeffs[n - 1]);
    eval_rescale(cc, result);
    cc->EvalAddInPlace(result, coeffs[n - 2]);

    for (int i = static_cast<int>(n) - 3; i >= 0; --i) {
        Ctx x_cloned = x->Clone();
        match_level(cc, x_cloned, result);

        result = cc->EvalMult(result, x_cloned);
        eval_rescale(cc, result);
        cc->EvalAddInPlace(result, coeffs[i]);
    }

    return result;
}


/// @brief Evaluates a polynomial using the Power-sum tree algorithm. Achieves optimal multiplicative depth of ceil(log2(d+1)) where d is the degree.
/// @param cc, the crypto context
/// @param x, the input ciphertext
/// @param coeffs, coefficients [a0, a1, ..., an] for a0 + a1*x + ... + an*x^n
/// @param pk, the public key (used for encrypting constant-only leaves)
/// @param slots, number of slots
/// @return the ciphertext resulting from evaluating the polynomial at x
Ctx eval_polynomial_ps(const CC& cc, const Ctx& x, const std::vector<double>& coeffs, const PublicKey<DCRTPoly>& pk, size_t slots) {
    int d = static_cast<int>(coeffs.size()) - 1;

    Ctx result = encrypt_const(cc, coeffs[0], slots, pk);

    if (d == 0)
        return result;

    int floor_log2 = std::floor(std::log2(d));

    std::vector<Ctx> powers(floor_log2 + 1);
    powers[0] = x;
    for (int p = 1; p <= floor_log2; ++p) {
        powers[p] = cc->EvalSquare(powers[p - 1]); // squaring triggers rescaling
    }

    for (int p = 1; p < coeffs.size(); ++p) {
        Ctx curr_x = nullptr;
        int pow_idx = 0;
        bool never_rescaled = true;
        for (int i = p; i > 0; i /= 2) {
            if (i % 2 == 1) {
                if (curr_x == nullptr) {
                    curr_x = cc->EvalMult(powers[pow_idx], coeffs[p]);
                    // eval_rescale(cc, curr_x); // maybe we don't need to rescale here?
                } else {
                    match_level(cc, curr_x, powers[pow_idx]);
                    // match_level(cc, powers[pow_idx], curr_x); //curr_x should always be at a level <= powers[pow_idx]
                    curr_x = cc->EvalMult(curr_x, powers[pow_idx]);
                    eval_rescale(cc, curr_x);
                    never_rescaled = false;
                }
            }
            pow_idx++;
        }
        if (never_rescaled) eval_rescale(cc, curr_x); // if we only multiplied with the coefficient, thus we need to rescale before adding to the result

        match_level(cc, result, curr_x);
        // match_level(cc, curr_x, result); //result should always be at a level <= curr_x
        cc->EvalAddInPlace(result, curr_x);
    }

    return result;
}


/// @brief Evaluates a polynomial using the Power-sum tree algorithm without storing intermediate powers. Achieves optimal multiplicative depth of ceil(log2(d+1)) where d is the degree.
/// @param cc, the crypto context
/// @param x, the input ciphertext
/// @param coeffs, coefficients [a0, a1, ..., an] for a0 + a1*x + ... + an*x^n
/// @param pk, the public key (used for encrypting constant-only leaves)
/// @param slots, number of slots
/// @return the ciphertext resulting from evaluating the polynomial at x
Ctx eval_polynomial_computational_ps(const CC& cc, const Ctx& x, const std::vector<double>& coeffs, const PublicKey<DCRTPoly>& pk, size_t slots) {
    Ctx result = encrypt_const(cc, coeffs[0], slots, pk);

    if (coeffs.size() == 1)
        return result;

    for (int p = 1; p < coeffs.size(); ++p) {
        Ctx curr_x = nullptr;
        Ctx running_x = x->Clone();
        int pow_idx = 0;
        bool never_rescaled = true;
        for (int i = p; i > 0; i /= 2) {
            if (i % 2 == 1) {
                if (curr_x == nullptr) {
                    curr_x = cc->EvalMult(running_x, coeffs[p]);
                } else {
                    match_level(cc, curr_x, running_x);
                    curr_x = cc->EvalMult(curr_x, running_x);
                    eval_rescale(cc, curr_x);
                    never_rescaled = false;
                }
            }
            running_x = cc->EvalSquare(running_x); // squaring triggers rescaling
            pow_idx++;
        }
        if (never_rescaled) eval_rescale(cc, curr_x); // if we only multiplied with the coefficient, thus we need to rescale before adding to the result

        match_level(cc, result, curr_x);
        cc->EvalAddInPlace(result, curr_x);
    }

    return result;
}


/// @brief Evaluates a degree-4 polynomial using hard-coded paterson-stockmeyer.
/// @param cc, the crypto context
/// @param x, the input ciphertext
/// @param coeffs, coefficients [a0, a1, ..., an] for a0 + a1*x + ... + an*x^n
/// @param pk, the public key (used for encrypting constant-only leaves)
/// @param slots, number of slots
/// @return the ciphertext resulting from evaluating the polynomial at x
Ctx eval_polynomial_deg4(const CC& cc, const Ctx& x, const std::vector<double>& coeffs) {
    Ctx x2 = cc->EvalSquare(x);
    Ctx result = cc->EvalSquare(x2);

    cc->EvalMultInPlace(result, coeffs[4]);
    eval_rescale(cc, result);

    Ctx A = cc->EvalMult(x, coeffs[1]);
    cc->EvalAddInPlace(A, coeffs[0]);

    Ctx B = cc->EvalMult(x, coeffs[3]);
    cc->EvalAddInPlace(B, coeffs[2]);
    match_level(cc, B, x2);
    B = cc->EvalMult(B, x2);
    eval_rescale(cc, B);

    match_level(cc, A, B);
    A = cc->EvalAdd(A, B);

    match_level(cc, A, result);

    return cc->EvalAdd(A, result);
}

/// @brief Evaluates a rational approximation p(x)/q(x) on a ciphertext.
/// Uses eval_polynomial for numerator/denominator and Goldschmidt iteration for division.
/// @param p_coeffs, numerator polynomial coefficients [p0, p1, ..., pn] (precomputed offline)
/// @param q_coeffs, denominator polynomial coefficients [q0, q1, ..., qm] with q0 = 1.0 (precomputed offline)
/// @param q_min, minimum value of q(x) over the input domain (precomputed offline)
/// @param q_max, maximum value of q(x) over the input domain (precomputed offline)
/// @param gs_iters, number of Goldschmidt iterations for computing 1/q(x)
Ctx eval_rational_approx(const CC& cc, const Ctx& x,
                          const std::vector<double>& p_coeffs,
                          const std::vector<double>& q_coeffs,
                          double q_min, double q_max,
                          const PublicKey<DCRTPoly>& pk,
                          size_t slots,
                          int gs_iters) {
    Ctx p_ct = eval_polynomial(cc, x, p_coeffs);
    Ctx q_ct = eval_polynomial(cc, x, q_coeffs);

    double alpha = 2.0 / (q_min + q_max);
    Ctx x0 = encrypt_const(cc, alpha, slots, pk);

    Ctx inv_q = goldschmidt_inv(cc, q_ct, x0, gs_iters);

    match_level(cc, p_ct, inv_q);
    Ctx result = cc->EvalMult(p_ct, inv_q);
    eval_rescale(cc, result);

    return result;
}

/// @brief Computes the degree-3 Taylor coefficients of 1/sqrt(z) centered at z0.
/// Call once during initialization; pass the result to eval_taylor_inv_sqrt at runtime.
/// Returns coefficients in the shifted basis: f(z) ≈ a0 + a1*(z-z0) + a2*(z-z0)^2 + a3*(z-z0)^3
/// @param z0, the expansion point (typically midpoint of the interval)
/// @return vector {a0, a1, a2, a3}
std::vector<double> taylor_inv_sqrt_coeffs(double z0) {
    double z0_sqrt = std::sqrt(z0);
    double a0 =  1.0 / z0_sqrt;
    double a1 = -1.0 / (2.0 * z0 * z0_sqrt);
    double a2 =  3.0 / (8.0 * z0 * z0 * z0_sqrt);
    double a3 = -5.0 / (16.0 * z0 * z0 * z0 * z0_sqrt);
    return {a0, a1, a2, a3};
}

/// @brief Evaluates the degree-3 Taylor approximation of 1/sqrt(z) around z0.
/// @param coeffs, precomputed Taylor coefficients from taylor_inv_sqrt_coeffs (computed once at init)
/// @param z0, the expansion point used when computing coeffs
Ctx eval_taylor_inv_sqrt(const CC& cc, const Ctx& x,
                          const std::vector<double>& coeffs, double z0) {
    Ctx u = cc->EvalAdd(x, -z0);
    return eval_polynomial(cc, u, coeffs);
}