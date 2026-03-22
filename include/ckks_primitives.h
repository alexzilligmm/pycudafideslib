#pragma once
// ckks_primitives.h
// Model-agnostic CKKS numerical primitives.
// Depends only on fideslib_wrapper.h — no LLaMA-specific types.
//
// These are the reusable building blocks for any transformer model:
//   - inv_sqrt_newton      : Newton–Halley iteration for 1/sqrt(x)
//   - goldschmidt_inv_sqrt : Goldschmidt refinement for 1/sqrt(x) after bootstrap
//   - exp_squaring         : (1 + x/N)^N approximation for exp(x)
//   - newton_inverse       : product-series 1/(1-r) for softmax denominator

#include "fideslib_wrapper.h"
#include <functional>

// ── Minimal model-agnostic FHE context ───────────────────────────────────
// Nonlinear ops that are model-independent take this instead of LlamaInference.
// Embed this in your own model struct, or use directly.
struct CKKSFHECtx {
    std::shared_ptr<CKKSContext> ckks;  // FIDESlib context + keys
    int slots;                           // N/2

    CC&       cc()       { return ckks->cc; }
    const CC& cc() const { return ckks->cc; }

    PublicKey<DCRTPoly>&  pk() { return ckks->keys.publicKey; }
    PrivateKey<DCRTPoly>& sk() { return ckks->keys.secretKey; }
};

// ─────────────────────────────────────────────────────────────────────────
// inv_sqrt_newton
//
// Iterates the Newton–Halley step for 1/sqrt(x):
//   ans_{n+1} = ans_n * (1.5 - 0.5 * x * ans_n^2)
//
// Parameters:
//   cc        : CKKS crypto context
//   slots     : number of plaintext slots (for scalar plaintext creation)
//   x         : ciphertext of the value whose reciprocal square root is wanted
//   ans_init  : starting approximation (e.g., linear: ans0 = slope*x + bias)
//   iters     : number of Newton steps (4 gives high accuracy for [0, 100])
//
// Returns ans ≈ 1/sqrt(x) after `iters` refinements.
// Matches Go's NewtonIter(varc, ans, iters).
Ctx inv_sqrt_newton(const CC& cc, int slots,
                    const Ctx& x, const Ctx& ans_init, int iters);

// ─────────────────────────────────────────────────────────────────────────
// goldschmidt_inv_sqrt
//
// Standard Goldschmidt refinement for 1/sqrt(x):
//   Computes sqrt_x = x * ans_init  (≈ sqrt(x)), then iterates:
//     h      = sqrt_x * ans          (≈ 1)
//     corr   = 1.5 - 0.5 * h        (correction factor, → 1 as h → 1)
//     sqrt_x = sqrt_x * corr
//     ans    = ans    * corr
//
// Quadratic convergence: if h = 1 + ε then h_next ≈ 1 - ε².
// Call after a bootstrap to refine the Newton estimate at fresh levels.
//
// Parameters:
//   cc        : CKKS crypto context
//   slots     : number of plaintext slots (for plaintext scalar creation)
//   x         : ciphertext whose 1/sqrt is wanted
//   ans_init  : Newton estimate ≈ 1/sqrt(x)
//   iters     : Goldschmidt iterations (2 is standard after 4 Newton steps)
//
// Returns refined ans ≈ 1/sqrt(x).
// Use (x * ans) as the sqrt, or (input * ans) for layer-norm output.
Ctx goldschmidt_inv_sqrt(const CC& cc, int slots,
                          const Ctx& x, const Ctx& ans_init, int iters);

// ─────────────────────────────────────────────────────────────────────────
// exp_squaring
//
// Approximates exp(z) via (1 + z/N)^N, N = 2^iters.
// The caller is responsible for pre-scaling:
//   x_in = z / 128 + 1    (for iters = 7 + temp, i.e., N = 128 * 2^temp)
// then calls exp_squaring(cc, x_in, 7 + temp).
//
// Returns x after `iters` squarings: x^(2^iters).
// Matches Go's (1+x/128)^(128*2^temp) squaring loop.
Ctx exp_squaring(const CC& cc, Ctx x, int iters);

// ─────────────────────────────────────────────────────────────────────────
// newton_inverse
//
// Computes 1/s using the product series 1/(1-r) = prod_k (1 + r^{2^k}):
//   Start:    res = 1 - s,  dnm = 1 + res
//   Iterate:  res = res^2;  dnm = dnm * (res + 1)
//
// After `iters` iterations, dnm ≈ 1/s provided |r| < 1.
// Used in Softmax and Argmax to invert the attention score sum.
// Matches Go's inverse Newton loop (9 iters for Softmax, 12 for Argmax).
//
// Note: one_ct may be level-dropped by match_level inside the loop.
// Pass a fresh encrypted-one or a const-ciphertext at the right level.
Ctx newton_inverse(const CC& cc, Ctx one_ct, Ctx res, Ctx dnm, int iters);

// ─────────────────────────────────────────────────────────────────────────
// goldschmidt_inv
//
// Goldschmidt iteration for 1/a given an initial approximation x0 ≈ 1/a.
//
// Algorithm (quadratic convergence):
//   d  = a * x0       (≈ 1)
//   e  = 1 - d        (error ≈ 0)
//   each iter:
//     x0 *= (1 + e)   (refine inverse; same factor as Goldschmidt division)
//     e   = e^2       (error squares)
//
// Parameters:
//   cc     : CKKS crypto context
//   slots  : number of plaintext slots (for scalar plaintext creation)
//   a      : ciphertext whose reciprocal is wanted
//   x0_init: starting approximation ≈ 1/a
//   iters  : number of Goldschmidt steps (2–3 is usually sufficient after
//             a good Newton or linear initial guess)
//
// Returns x0 ≈ 1/a after `iters` refinements.
Ctx goldschmidt_inv(const CC& cc, int slots,
                    const Ctx& a, const Ctx& x0_init, int iters);

// ─────────────────────────────────────────────────────────────────────────
// chebyshev_coeffs
//
// Computes n = degree+1 Chebyshev coefficients of f on [a, b], following
// Lattigo's bignum.ChebyshevApproximation / chebyCoeffs algorithm.
//
// Lattigo's Nodes=127 → chebyshevNodes(128) → 128 nodes → 128 coefficients
// for a degree-127 polynomial.  Pass degree = Nodes (e.g. 127 for SiLU).
//
// Normalization: OpenFHE's EvalChebyshevSeries evaluates
//   c[0]/2 + Σ_{k≥1} c[k]·T_k(x)
// so we use (2/n) for ALL k, including k=0.  This is different from Lattigo
// which uses (1/n) for k=0; OpenFHE's EvalChebyshevCoefficients does the
// same (2/n) rescaling so the conventions match end-to-end.
//
// The Chebyshev basis T_k is evaluated via the stable 3-term recurrence
//   T_{k+1} = 2u·T_k − T_{k-1},  T_0=1, T_1=u
// to avoid precision loss from cos(k·arccos(u)) for large k.
//
// Returns a vector of `degree+1` coefficients ready for EvalChebyshevSeries.
std::vector<double> chebyshev_coeffs(std::function<double(double)> f,
                                      double a, double b, int degree);

// ─────────────────────────────────────────────────────────────────────────
// eval_chebyshev
//
// Evaluates f(x) homomorphically via a degree-`degree` Chebyshev
// approximation on [a, b].  Internally calls chebyshev_coeffs then
// EvalChebyshevSeries; the result is decryptable to f(x) ± CKKS noise.
//
// Equivalent to Go's polyeval.Evaluate after ChebyshevApproximation.
Ctx eval_chebyshev(const CC& cc, const Ctx& x,
                   std::function<double(double)> f,
                   double a, double b, int degree);

// ─────────────────────────────────────────────────────────────────────────
// eval_linear_wsum
//
// Computes the weighted sum Σ weights[i] * cts[i] using OpenFHE's
// EvalLinearWSum, which fuses the scalar multiplications and additions into
// a single pass at a fixed noise level.
//
// Parameters:
//   cc      : CKKS crypto context
//   cts     : list of ciphertexts (all must be at the same level / scale)
//   weights : coefficients; must satisfy cts.size() == weights.size()
//
// Returns a single ciphertext at the same level as the inputs (no rescale
// is needed after EvalLinearWSum with FLEXIBLEAUTO scaling mode).
Ctx eval_linear_wsum(const CC& cc,
                     std::vector<Ctx>& cts,
                     const std::vector<double>& weights);
