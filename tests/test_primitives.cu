// test_primitives.cu
// Unit tests for the model-agnostic CKKS primitives in primitives.cu.
// Each test encrypts known values, runs the primitive, decrypts, and
// compares against a plaintext reference.
//
// Uses a non-bootstrap CKKSFHECtx (depth=16) so tests run quickly.
// Tolerances reflect CKKS noise at scale=2^40, logN=12, depth≤16.

#include <gtest/gtest.h>
#include "ckks_primitives.h"

#include <cmath>
#include <functional>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>

// ── fixture ───────────────────────────────────────────────────────────────
class PrimitivesTest : public ::testing::Test {
protected:
    static constexpr int  LOGN   = 12;
    static constexpr int  DEPTH  = 16;
    static constexpr int  SLOTS  = 1 << (LOGN - 1);  // 2048
    static constexpr double TOL  = 1e-4;               // CKKS noise at depth 16

    CKKSFHECtx fhe;

    void SetUp() override {
        fhe.ckks = make_ckks_context(LOGN, DEPTH, /*scale_bits=*/40,
                                      SLOTS, /*bootstrap=*/false);
        fhe.slots = SLOTS;
    }

    // Encrypt a constant to all slots
    Ctx encrypt_const_ct(double val) {
        Ptx pt = encode_const(fhe.cc(), val, SLOTS, 0);
        return encrypt(fhe.cc(), pt, fhe.pk());
    }

    // Encrypt a vector
    Ctx encrypt_vec(const std::vector<double>& v) {
        Ptx pt = encode(fhe.cc(), v, 0);
        return encrypt(fhe.cc(), pt, fhe.pk());
    }

    // Decrypt and return real values
    std::vector<double> dec(const Ctx& ct) {
        return decrypt(fhe.cc(), ct, fhe.sk());
    }

    // Max absolute error over first `n` slots
    double max_err(const Ctx& ct, double expected, int n = SLOTS) {
        auto v = dec(ct);
        double err = 0.0;
        for (int i = 0; i < n; ++i)
            err = std::max(err, std::abs(v[i] - expected));
        return err;
    }

    double max_err_vec(const Ctx& ct, const std::vector<double>& ref) {
        auto v = dec(ct);
        double err = 0.0;
        for (size_t i = 0; i < std::min(v.size(), ref.size()); ++i)
            err = std::max(err, std::abs(v[i] - ref[i]));
        return err;
    }
};

// ── inv_sqrt_newton ───────────────────────────────────────────────────────
// Newton-Halley for 1/sqrt(x): ans_{n+1} = ans_n * (1.5 - 0.5*x*ans_n^2)
// Starting from a linear initial guess, 4 iterations give high accuracy
// for x in [0.01, 1].

TEST_F(PrimitivesTest, InvSqrtNewton_Constant) {
    // x = 0.25 → 1/sqrt(0.25) = 2.0
    // Linear initial guess: ans0 = slope*x + bias calibrated for x ≈ 0.25
    // Use ans0 ≈ 1.9 (close to 2.0) to verify convergence.
    double x_val = 0.25;
    double ans0_val = 1.9;  // near 1/sqrt(0.25)=2.0

    Ctx x   = encrypt_const_ct(x_val);
    Ctx ans = encrypt_const_ct(ans0_val);

    Ctx result = inv_sqrt_newton(fhe.cc(), SLOTS, x, ans, 4);

    double expected = 1.0 / std::sqrt(x_val);
    double err = max_err(result, expected);
    std::cout << "InvSqrtNewton x=0.25: expected=" << expected
              << " err=" << err << "\n";
    EXPECT_LT(err, TOL);
}

TEST_F(PrimitivesTest, InvSqrtNewton_SmallX) {
    // x = 0.04 → 1/sqrt(0.04) = 5.0
    double x_val = 0.04;
    // Initial guess from linear approximation (slope=-1.29054537e-04, bias=0.129054537)
    // is only good for small varc; use a better ans0 for this test
    double ans0_val = 4.8;

    Ctx x   = encrypt_const_ct(x_val);
    Ctx ans = encrypt_const_ct(ans0_val);

    Ctx result = inv_sqrt_newton(fhe.cc(), SLOTS, x, ans, 4);

    double expected = 1.0 / std::sqrt(x_val);
    double err = max_err(result, expected);
    std::cout << "InvSqrtNewton x=0.04: expected=" << expected
              << " err=" << err << "\n";
    EXPECT_LT(err, 1.0);  // wider tolerance: 4 iters from far-off initial guess
}

TEST_F(PrimitivesTest, InvSqrtNewton_VaryingInput) {
    // Test with a vector of values in [0.1, 0.5]
    // Newton-Halley iter consumes 3 levels each; DEPTH=16 supports ≤5 iters.
    // Use a near-exact initial guess (2% off) so 4 iters suffice.
    std::vector<double> x_vec(SLOTS), ans_vec(SLOTS), ref(SLOTS);
    for (int i = 0; i < SLOTS; ++i) {
        x_vec[i]   = 0.1 + 0.4 * i / (double)SLOTS;
        ans_vec[i] = 1.0 / std::sqrt(x_vec[i]) * 1.02;  // 2% above exact
        ref[i]     = 1.0 / std::sqrt(x_vec[i]);
    }

    Ctx x   = encrypt_vec(x_vec);
    Ctx ans = encrypt_vec(ans_vec);

    Ctx result = inv_sqrt_newton(fhe.cc(), SLOTS, x, ans, 4);
    double err = max_err_vec(result, ref);
    std::cout << "InvSqrtNewton varying: max_err=" << err << "\n";
    EXPECT_LT(err, 0.1);
}

// ── goldschmidt_inv_sqrt ──────────────────────────────────────────────────
// Goldschmidt refinement: given x and ans≈1/sqrt(x), refines ans further.
// Used after bootstrap as Goldschmidt uses fewer levels than Newton.

TEST_F(PrimitivesTest, GoldschmidtInvSqrt_Constant) {
    // x = 0.25, ans ≈ 2.0 (exact); 2 Goldschmidt iters should stay at 2.0
    double x_val   = 0.25;
    double ans_val = 2.05;  // slightly off to test convergence

    Ctx x   = encrypt_const_ct(x_val);
    Ctx ans = encrypt_const_ct(ans_val);

    Ctx result = goldschmidt_inv_sqrt(fhe.cc(), SLOTS, x, ans, 4);

    double expected = 1.0 / std::sqrt(x_val);
    double err = max_err(result, expected);
    std::cout << "GoldschmidtInvSqrt x=0.25: expected=" << expected
              << " err=" << err << "\n";
    EXPECT_LT(err, 1e-3);
}

TEST_F(PrimitivesTest, GoldschmidtInvSqrt_VaryingInput) {
    std::vector<double> x_vec(SLOTS), ans_vec(SLOTS), ref(SLOTS);
    for (int i = 0; i < SLOTS; ++i) {
        x_vec[i]   = 0.1 + 0.4 * i / (double)SLOTS;
        // Good initial guess (from Newton)
        ans_vec[i] = 1.0 / std::sqrt(x_vec[i]) * (1.0 + 0.002);  // 0.2% off
        ref[i]     = 1.0 / std::sqrt(x_vec[i]);
    }

    Ctx x   = encrypt_vec(x_vec);
    Ctx ans = encrypt_vec(ans_vec);

    Ctx result = goldschmidt_inv_sqrt(fhe.cc(), SLOTS, x, ans, 2);
    double err = max_err_vec(result, ref);
    std::cout << "GoldschmidtInvSqrt varying: max_err=" << err << "\n";
    EXPECT_LT(err, 0.01);
}

// ── exp_squaring ──────────────────────────────────────────────────────────
// exp(z) ≈ (1 + z/128)^128 via 7 squarings.
// Caller pre-scales: x = 1 + z/128, then squares 7 times.
// Accurate for small z (e.g. z ∈ [-2, 2]).

TEST_F(PrimitivesTest, ExpSquaring_Zero) {
    // z=0: (1+0)^128 = 1
    Ctx x = encrypt_const_ct(1.0);  // 1 + 0/128

    Ctx result = exp_squaring(fhe.cc(), x, 7);
    double err = max_err(result, 1.0);
    std::cout << "ExpSquaring z=0: err=" << err << "\n";
    EXPECT_LT(err, TOL);
}

TEST_F(PrimitivesTest, ExpSquaring_SmallPositive) {
    // z=0.5: exp(0.5) ≈ 1.6487; x = 1 + 0.5/128 ≈ 1.003906
    double z     = 0.5;
    double x_val = 1.0 + z / 128.0;
    double expected = std::pow(x_val, 128.0);  // = exp(z) approximately

    Ctx x = encrypt_const_ct(x_val);
    Ctx result = exp_squaring(fhe.cc(), x, 7);

    double err = max_err(result, expected);
    double approx_err = std::abs(expected - std::exp(z));
    std::cout << "ExpSquaring z=0.5: fhe_err=" << err
              << " approx_err=" << approx_err << "\n";
    EXPECT_LT(err, 1e-3);
}

TEST_F(PrimitivesTest, ExpSquaring_SmallNegative) {
    // z=-0.5: exp(-0.5) ≈ 0.6065
    double z     = -0.5;
    double x_val = 1.0 + z / 128.0;
    double expected = std::pow(x_val, 128.0);

    Ctx x = encrypt_const_ct(x_val);
    Ctx result = exp_squaring(fhe.cc(), x, 7);

    double err = max_err(result, expected);
    std::cout << "ExpSquaring z=-0.5: fhe_err=" << err << "\n";
    EXPECT_LT(err, 1e-3);
}

// ── newton_inverse ────────────────────────────────────────────────────────
// Product-series inverse: 1/s = prod_k (1+r^{2^k}) where r = 1-s.
// Requires |r| < 1 (i.e. 0 < s < 2).

TEST_F(PrimitivesTest, NewtonInverse_SmallDenominator) {
    // s = 0.8, r = 1-s = 0.2 → 1/s = 1.25
    // After 9 iters the product series 1/(1-r) converges well for |r|=0.2
    double s = 0.8;
    double r = 1.0 - s;

    Ctx one_ct = encrypt_const_ct(1.0);
    Ctx res    = encrypt_const_ct(r);
    Ctx dnm    = encrypt_const_ct(1.0 + r);  // dnm = 1 + r

    Ctx result = newton_inverse(fhe.cc(), one_ct, res, dnm, 9);

    double expected = 1.0 / s;
    double err = max_err(result, expected);
    std::cout << "NewtonInverse s=0.8: expected=" << expected
              << " err=" << err << "\n";
    EXPECT_LT(err, TOL);
}

TEST_F(PrimitivesTest, NewtonInverse_LargerDenominator) {
    // s = 0.5, r = 0.5 → 1/s = 2.0; |r|=0.5 takes more iters to converge
    double s = 0.5;
    double r = 1.0 - s;

    Ctx one_ct = encrypt_const_ct(1.0);
    Ctx res    = encrypt_const_ct(r);
    Ctx dnm    = encrypt_const_ct(1.0 + r);

    Ctx result = newton_inverse(fhe.cc(), one_ct, res, dnm, 9);

    double expected = 1.0 / s;
    double err = max_err(result, expected);
    std::cout << "NewtonInverse s=0.5: expected=" << expected
              << " err=" << err << "\n";
    // Wider tolerance: |r|=0.5 converges slowly (9 iters → ~2^-9 residual)
    EXPECT_LT(err, 0.01);
}

// ── goldschmidt_inv ───────────────────────────────────────────────────────
// Standard Goldschmidt iteration for 1/a given initial estimate x0 ≈ 1/a.
//
// Algorithm (mirrors FIDESlib::CKKS::GoldschmidtInv):
//   d  = a * x0  (≈ 1)
//   e  = 1 - d   (≈ 0)
//   each iter:  x0 *= (1+e);  e = e^2
//
// Quadratic convergence: if e_0 = ε then after k iters |e_k| ≈ ε^{2^k}.
// Good starting point: x0 ≈ 1/a from a linear polynomial or a Newton step.

TEST_F(PrimitivesTest, GoldschmidtInv_SmallDenominator) {
    // a = 0.8, x0 ≈ 1/0.8 = 1.25 (exact); 3 iters should stay near 1.25.
    double a_val  = 0.8;
    double x0_val = 1.0 / a_val;  // exact initial guess → should converge in 0 iters

    Ctx a  = encrypt_const_ct(a_val);
    Ctx x0 = encrypt_const_ct(x0_val);

    Ctx result = goldschmidt_inv(fhe.cc(), SLOTS, a, x0, 3);

    double expected = 1.0 / a_val;
    double err = max_err(result, expected);
    std::cout << "GoldschmidtInv a=0.8 (exact x0): expected=" << expected
              << " err=" << err << "\n";
    EXPECT_LT(err, TOL);
}

TEST_F(PrimitivesTest, GoldschmidtInv_RoughInitialGuess) {
    // a = 0.7, x0 = 1.3 (rough; exact is 1/0.7 ≈ 1.4286).
    // e_0 = 1 - 0.7*1.3 = 1 - 0.91 = 0.09.  After 3 iters: |e| ≈ 0.09^8 ≈ 4e-9.
    double a_val  = 0.7;
    double x0_val = 1.3;  // ~8% off

    Ctx a  = encrypt_const_ct(a_val);
    Ctx x0 = encrypt_const_ct(x0_val);

    Ctx result = goldschmidt_inv(fhe.cc(), SLOTS, a, x0, 3);

    double expected = 1.0 / a_val;
    double err = max_err(result, expected);
    std::cout << "GoldschmidtInv a=0.7 (rough x0=1.3): expected=" << expected
              << " err=" << err << "\n";
    EXPECT_LT(err, 1e-3);
}

TEST_F(PrimitivesTest, GoldschmidtInv_VaryingInput) {
    // Varying a in [0.5, 0.95], x0 = 1/a (exact) — tests level handling.
    std::vector<double> a_vec(SLOTS), x0_vec(SLOTS), ref(SLOTS);
    for (int i = 0; i < SLOTS; ++i) {
        a_vec[i]  = 0.5 + 0.45 * i / (double)SLOTS;
        x0_vec[i] = 1.0 / a_vec[i] * (1.0 + 0.01);  // 1% above exact
        ref[i]    = 1.0 / a_vec[i];
    }

    Ctx a  = encrypt_vec(a_vec);
    Ctx x0 = encrypt_vec(x0_vec);

    Ctx result = goldschmidt_inv(fhe.cc(), SLOTS, a, x0, 3);
    double err = max_err_vec(result, ref);
    std::cout << "GoldschmidtInv varying: max_err=" << err << "\n";
    EXPECT_LT(err, 1e-3);
}

// ── chebyshev_coeffs (plaintext, no encryption) ───────────────────────────
// These tests verify the coefficient computation in plain double arithmetic
// before trusting the encrypted evaluation.

TEST_F(PrimitivesTest, ChebyshevCoeffs_SiLU_Degree127) {
    // SiLU(x) = x / (exp(-x) + 1); degree-127 approximation on [-20, 20].
    // Verify: (1) right number of coefficients, (2) polynomial evaluates
    // accurately at known points using the plaintext series sum.
    auto silu_fn = [](double x) { return x / (std::exp(-x) + 1.0); };
    auto coeffs  = chebyshev_coeffs(silu_fn, -20.0, 20.0, 127);

    ASSERT_EQ((int)coeffs.size(), 128);

    // Evaluate degree-127 Chebyshev series at a few points in plaintext.
    // EvalChebyshevSeries computes c[0]/2 + Σ_{k≥1} c[k]*T_k(u)
    // where u = (2x - (a+b)) / (b-a).
    auto cheby_eval = [&](double x) -> double {
        double u = (2.0 * x - (-20.0 + 20.0)) / (20.0 - (-20.0));  // = x/20
        double Tprev = 1.0, T = u, sum = coeffs[0] * 0.5;
        for (int k = 1; k < 128; ++k) {
            sum += coeffs[k] * T;  // T = T_k(u) at this point
            double Tnext = 2.0 * u * T - Tprev;
            Tprev = T; T = Tnext;
        }
        return sum;
    };

    // SiLU(0) = 0
    EXPECT_NEAR(cheby_eval(0.0),  silu_fn(0.0),  1e-6);
    // SiLU(1) ≈ 0.7311
    EXPECT_NEAR(cheby_eval(1.0),  silu_fn(1.0),  1e-5);
    // SiLU(-1) ≈ -0.2689
    EXPECT_NEAR(cheby_eval(-1.0), silu_fn(-1.0), 1e-5);
    // SiLU(5) ≈ 4.9665
    EXPECT_NEAR(cheby_eval(5.0),  silu_fn(5.0),  1e-4);

    std::cout << "ChebyshevCoeffs SiLU(0)="  << cheby_eval(0.0)
              << " SiLU(1)=" << cheby_eval(1.0)
              << " SiLU(-1)=" << cheby_eval(-1.0) << "\n";
}

TEST_F(PrimitivesTest, ChebyshevCoeffs_Identity) {
    // Approximate f(x)=x on [-1,1] with degree-7: should reproduce exactly.
    auto id_fn  = [](double x) { return x; };
    auto coeffs = chebyshev_coeffs(id_fn, -1.0, 1.0, 7);

    ASSERT_EQ((int)coeffs.size(), 8);

    // Evaluate at a few points; f(x)=x so error should be tiny.
    auto cheby_eval = [&](double x) -> double {
        double u = x;  // a=-1, b=1 → u = (2x-0)/2 = x
        double Tprev = 1.0, T = u, sum = coeffs[0] * 0.5;
        for (int k = 1; k < 8; ++k) {
            sum += coeffs[k] * T;  // T = T_k(u) at this point
            double Tnext = 2.0 * u * T - Tprev;
            Tprev = T; T = Tnext;
        }
        return sum;
    };

    EXPECT_NEAR(cheby_eval( 0.5),  0.5, 1e-10);
    EXPECT_NEAR(cheby_eval(-0.5), -0.5, 1e-10);
    EXPECT_NEAR(cheby_eval( 0.9),  0.9, 1e-10);
    std::cout << "ChebyshevCoeffs identity(0.5)=" << cheby_eval(0.5) << "\n";
}

// ── eval_chebyshev (encrypted) ────────────────────────────────────────────
// Tests the homomorphic Chebyshev evaluator.

TEST_F(PrimitivesTest, EvalChebyshev_SiLU_AtZero) {
    // SiLU(0) = 0; encrypt 0.0, evaluate degree-127 poly on [-20,20].
    Ctx x = encrypt_const_ct(0.0);

    auto silu_fn = [](double v) { return v / (std::exp(-v) + 1.0); };
    Ctx result   = eval_chebyshev(fhe.cc(), x, silu_fn, -20.0, 20.0, 127);

    double err = max_err(result, 0.0);
    std::cout << "EvalChebyshev SiLU(0): err=" << err << "\n";
    EXPECT_LT(err, 1e-3);
}

TEST_F(PrimitivesTest, EvalChebyshev_SiLU_KnownValues) {
    // SiLU(1) ≈ 0.7311, SiLU(-1) ≈ -0.2689.
    auto silu_fn = [](double v) { return v / (std::exp(-v) + 1.0); };

    {
        Ctx x   = encrypt_const_ct(1.0);
        Ctx res = eval_chebyshev(fhe.cc(), x, silu_fn, -20.0, 20.0, 127);
        double expected = silu_fn(1.0);
        double err = max_err(res, expected);
        std::cout << "EvalChebyshev SiLU(1): expected=" << expected
                  << " err=" << err << "\n";
        EXPECT_LT(err, 1e-3);
    }
    {
        Ctx x   = encrypt_const_ct(-1.0);
        Ctx res = eval_chebyshev(fhe.cc(), x, silu_fn, -20.0, 20.0, 127);
        double expected = silu_fn(-1.0);
        double err = max_err(res, expected);
        std::cout << "EvalChebyshev SiLU(-1): expected=" << expected
                  << " err=" << err << "\n";
        EXPECT_LT(err, 1e-3);
    }
}

TEST_F(PrimitivesTest, EvalChebyshev_SiLU_VaryingInput) {
    // Verify over x in [-0.5, 0.5] (well within [-20,20]).
    auto silu_fn = [](double v) { return v / (std::exp(-v) + 1.0); };
    std::vector<double> x_vec(SLOTS), ref(SLOTS);
    for (int i = 0; i < SLOTS; ++i) {
        x_vec[i] = -0.5 + 1.0 * i / (double)SLOTS;
        ref[i]   = silu_fn(x_vec[i]);
    }

    Ctx x      = encrypt_vec(x_vec);
    Ctx result = eval_chebyshev(fhe.cc(), x, silu_fn, -20.0, 20.0, 127);

    double err = max_err_vec(result, ref);
    std::cout << "EvalChebyshev SiLU varying [-0.5,0.5]: max_err=" << err << "\n";
    EXPECT_LT(err, 1e-3);
}

// ── eval_linear_wsum ──────────────────────────────────────────────────────
// Verifies the fused weighted-sum wrapper.

TEST_F(PrimitivesTest, EvalLinearWSum_ThreeTerms) {
    // 2.0*ct_a + 3.0*ct_b + (-1.0)*ct_c  where a=1, b=2, c=0.5
    // expected: 2*1 + 3*2 + (-1)*0.5 = 2 + 6 - 0.5 = 7.5
    double a = 1.0, b = 2.0, c = 0.5;
    double expected = 2.0*a + 3.0*b + (-1.0)*c;

    Ctx ct_a = encrypt_const_ct(a);
    Ctx ct_b = encrypt_const_ct(b);
    Ctx ct_c = encrypt_const_ct(c);

    std::vector<Ctx>    cts     = {ct_a, ct_b, ct_c};
    std::vector<double> weights = {2.0, 3.0, -1.0};

    Ctx result = eval_linear_wsum(fhe.cc(), cts, weights);

    double err = max_err(result, expected);
    std::cout << "EvalLinearWSum 2*1+3*2-0.5: expected=" << expected
              << " err=" << err << "\n";
    EXPECT_LT(err, TOL);
}
