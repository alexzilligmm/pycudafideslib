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


/// @brief Evaluates a polynomial using the Paterson-Stockmeyer tree algorithm. Achieves optimal multiplicative depth of ceil(log2(d+1)) where d is the degree.
/// @param cc, the crypto context
/// @param x, the input ciphertext
/// @param coeffs, coefficients [a0, a1, ..., an] for a0 + a1*x + ... + an*x^n
/// @param pk, the public key (used for encrypting constant-only leaves)
/// @param slots, number of slots
/// @return the ciphertext resulting from evaluating the polynomial at x
Ctx eval_polynomial_ps(const CC& cc, const Ctx& x, const std::vector<double>& coeffs, const PublicKey<DCRTPoly>& pk, size_t slots) {
    int d = static_cast<int>(coeffs.size()) - 1;

    if (d == 0)
        return encrypt_const(cc, coeffs[0], slots, pk);

    std::map<int, Ctx> powers;
    powers[1] = x;
    for (int p = 1; (1 << p) <= d; ++p) {
        int prev = 1 << (p - 1);
        Ctx sq = cc->EvalMult(powers[prev], powers[prev]);
        eval_rescale(cc, sq);
        powers[1 << p] = sq;
    }

    std::function<Ctx(int, int)> tree_eval = [&](int lo, int hi) -> Ctx {
        int deg = hi - lo;

        if (deg == 0) {
            return encrypt_const(cc, coeffs[lo], slots, pk);
        }

        if (deg == 1) {
            Ctx result = cc->EvalMult(powers.at(1), coeffs[lo + 1]);
            eval_rescale(cc, result);
            cc->EvalAddInPlace(result, coeffs[lo]);
            return result;
        }

        int m = 1;
        while (m * 2 <= deg) m *= 2;

        Ctx low_res = tree_eval(lo, lo + m - 1);

        int high_deg = hi - (lo + m);
        Ctx combined;

        if (high_deg == 0) {
            Ctx xm = powers.at(m);
            combined = cc->EvalMult(xm, coeffs[lo + m]);
            eval_rescale(cc, combined);
        } else {
            Ctx high_res = tree_eval(lo + m, hi);
            Ctx xm = powers.at(m);
            match_level(cc, high_res, xm);
            match_level(cc, xm, high_res);
            combined = cc->EvalMult(xm, high_res);
            eval_rescale(cc, combined);
        }

        match_level(cc, low_res, combined);
        match_level(cc, combined, low_res);
        cc->EvalAddInPlace(combined, low_res);

        return combined;
    };

    return tree_eval(0, d);
}