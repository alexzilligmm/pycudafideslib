#pragma once
#include "fideslib_wrapper.h"
#include "inference.h"
#include <functional>
#include <algorithm>
#include <vector>

struct CKKSFHECtx {
    std::shared_ptr<CKKSContext> ckks;  // FIDESlib context + keys
    int slots;                           // N/2

    CC&       cc()       { return ckks->cc; }
    const CC& cc() const { return ckks->cc; }

    PublicKey<DCRTPoly>&  pk() { return ckks->keys.publicKey; }
    PrivateKey<DCRTPoly>& sk() { return ckks->keys.secretKey; }
};

// ---------------------------------------------------------------------------
// DepthGuard — programmable bootstrap insertion for iterative primitives.
//
// Two modes (can be combined):
//
//   Scheduled   – bootstrap at fixed iteration indices decided by an external
//                 planner (e.g. the Python placement script).
//                 Set `schedule` to a sorted vector of iteration indices.
//
//   Dynamic     – bootstrap when remaining modulus chain levels fall below a
//                 threshold.  Set `total_depth` and `min_remaining`.
//
// Scheduled mode fires first; dynamic acts as a safety net.
// An empty guard (default-constructed) is a no-op.
// ---------------------------------------------------------------------------
struct DepthGuard {
    std::function<Ctx(const Ctx&)> refresh;   // bootstrap / refresh callback
    std::vector<int> schedule;                 // sorted iteration indices
    uint32_t total_depth   = 0;               // modulus chain length
    uint32_t min_remaining = 0;               // 0 → dynamic mode disabled

    explicit operator bool() const { return !!refresh; }

    Ctx operator()(const Ctx& ct, int iter) const {
        if (!refresh) return ct;
        // Scheduled bootstrap
        if (!schedule.empty() &&
            std::binary_search(schedule.begin(), schedule.end(), iter))
            return refresh(ct);
        // Dynamic fallback
        if (min_remaining > 0 && total_depth > 0) {
            uint32_t consumed = level_of(ct);
            if (consumed < total_depth &&
                (total_depth - consumed) < min_remaining)
                return refresh(ct);
        }
        return ct;
    }
};

Ctx inv_sqrt_newton(const CC& cc, const Ctx& x, const Ctx& ans_init, int iters,
                    const DepthGuard& dg = {});

Ctx goldschmidt_inv_sqrt(const CC& cc, const Ctx& x, const Ctx& ans_init, int iters,
                         const DepthGuard& dg = {});

Ctx exp_squaring(const CC& cc, Ctx x, int iters,
                 const DepthGuard& dg = {});

Ctx newton_inverse(const CC& cc, const Ctx& res, Ctx dnm, int iters,
                   const DepthGuard& dg = {});

Ctx goldschmidt_inv(const CC& cc, const Ctx& a, const Ctx& x0_init, int iters,
                    const DepthGuard& dg = {});

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

Ctx eval_polynomial_computational_ps(const CC& cc,
                           const Ctx& x,
                           const std::vector<double>& coeffs,
                           const PublicKey<DCRTPoly>& pk,
                           size_t slots);

Ctx eval_polynomial_deg4(const CC& cc, const Ctx& x, const std::vector<double>& coeffs);

Ctx eval_rational_approx(const CC& cc, const Ctx& x,
                          const std::vector<double>& p_coeffs,
                          const std::vector<double>& q_coeffs,
                          double q_min, double q_max,
                          const PublicKey<DCRTPoly>& pk,
                          size_t slots,
                          int gs_iters);

std::vector<double> taylor_inv_sqrt_coeffs(double z0);

Ctx eval_taylor_inv_sqrt(const CC& cc, const Ctx& x,
                          const std::vector<double>& coeffs, double z0);

Ctx eval_linear_wsum(const CC& cc,
                     std::vector<Ctx>& cts,
                     const std::vector<double>& weights);

/// Masks the first `active_dim * (slots / active_dim)` slots, zeroing the rest.
/// Returns ct unchanged (no copy) when all slots are already active.
/// Cost: 1 level when masking is needed, 0 otherwise.
Ctx mask_slots(const CC& cc, const Ctx& x, int slots, int active_dim);

Ctx compute_average(Inference& inf, const Ctx& x_in);

/// Computes variance, deriving the mean internally.
Ctx compute_variance(Inference& inf, const Ctx& x_in);

/// Computes variance given a precomputed mean (as returned by compute_average).
Ctx compute_variance(Inference& inf, const Ctx& x_in, Ctx mean);
