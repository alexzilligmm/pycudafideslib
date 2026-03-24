#pragma once
#include "fideslib_wrapper.h"
#include "inference.h"
#include <functional>

struct CKKSFHECtx {
    std::shared_ptr<CKKSContext> ckks;  // FIDESlib context + keys
    int slots;                           // N/2

    CC&       cc()       { return ckks->cc; }
    const CC& cc() const { return ckks->cc; }

    PublicKey<DCRTPoly>&  pk() { return ckks->keys.publicKey; }
    PrivateKey<DCRTPoly>& sk() { return ckks->keys.secretKey; }
};


Ctx inv_sqrt_newton(const CC& cc, const Ctx& x, const Ctx& ans_init, int iters);

Ctx goldschmidt_inv_sqrt(const CC& cc, const Ctx& x, const Ctx& ans_init, int iters);

Ctx exp_squaring(const CC& cc, Ctx x, int iters);

Ctx newton_inverse(const CC& cc, const Ctx& res, const Ctx& dnm, int iters);

Ctx goldschmidt_inv(const CC& cc, const Ctx& a, const Ctx& x0_init, int iters);

std::vector<double> chebyshev_coeffs(std::function<double(double)> f,
                                      double a, double b, int degree);

std::vector<double> standard_to_chebyshev(const std::vector<double>& poly_coeffs, double a, double b);

Ctx eval_chebyshev(const CC& cc, const Ctx& x, std::vector<double> coeffs, double a, double b);

Ctx eval_chebyshev_f(const CC& cc, const Ctx& x,
                   std::function<double(double)> f,
                   double a, double b, int degree);


Ctx eval_polynomial(const CC& cc, const Ctx& x, const std::vector<double>& coeffs);

Ctx eval_polynomial_ps(const CC& cc,
                           const Ctx& x,
                           const std::vector<double>& coeffs,
                           const PublicKey<DCRTPoly>& pk,
                           size_t slots);

Ctx eval_linear_wsum(const CC& cc,
                     std::vector<Ctx>& cts,
                     const std::vector<double>& weights);

Ctx compute_average(Inference& inf, const Ctx& x_in);

/// Computes variance, deriving the mean internally.
Ctx compute_variance(Inference& inf, const Ctx& x_in);

/// Computes variance given a precomputed mean (as returned by compute_average).
Ctx compute_variance(Inference& inf, const Ctx& x_in, Ctx mean);
