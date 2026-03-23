#pragma once
#include "fideslib_wrapper.h"
#include <functional>

struct CKKSFHECtx {
    std::shared_ptr<CKKSContext> ckks;  // FIDESlib context + keys
    int slots;                           // N/2

    CC&       cc()       { return ckks->cc; }
    const CC& cc() const { return ckks->cc; }

    PublicKey<DCRTPoly>&  pk() { return ckks->keys.publicKey; }
    PrivateKey<DCRTPoly>& sk() { return ckks->keys.secretKey; }
};


Ctx inv_sqrt_newton(const CC& cc, int slots,
                    const Ctx& x, const Ctx& ans_init, int iters);


Ctx goldschmidt_inv_sqrt(const CC& cc, int slots,
                          const Ctx& x, const Ctx& ans_init, int iters);

Ctx exp_squaring(const CC& cc, Ctx x, int iters);

Ctx newton_inverse(const CC& cc, Ctx res, Ctx dnm, int iters);

Ctx goldschmidt_inv(const CC& cc, int slots,
                    const Ctx& a, const Ctx& x0_init, int iters);

std::vector<double> chebyshev_coeffs(std::function<double(double)> f,
                                      double a, double b, int degree);

std::vector<double> standard_to_chebyshev(const std::vector<double>& poly_coeffs, double a, double b);

Ctx eval_chebyshev(const CC& cc, const Ctx& x, std::vector<double> coeffs, double a, double b);

Ctx eval_chebyshev_f(const CC& cc, const Ctx& x,
                   std::function<double(double)> f,
                   double a, double b, int degree);


Ctx eval_polynomial(const CC& cc, const Ctx& x, const std::vector<double>& coeffs);

Ctx eval_linear_wsum(const CC& cc,
                     std::vector<Ctx>& cts,
                     const std::vector<double>& weights);
