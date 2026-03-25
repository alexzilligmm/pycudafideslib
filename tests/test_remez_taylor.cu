#include <gtest/gtest.h>
#include "fideslib_wrapper.h"
#include <cmath>
#include <numeric>
#include "test_basics.h"
#include "ckks_primitives.h"

// ===== Precomputed Remez (3,1) rational coefficients for 1/sqrt(x) =====

// Range [1, 600]
static const std::vector<double> REMEZ_P_1_600 = {
    1.358100629784094e+00,
    4.835792394549090e-02,
   -1.148123197994000e-04,
    1.093999116055408e-07
};
static const std::vector<double> REMEZ_Q_1_600 = {
    1.000000000000000e+00,
    4.885466856794397e-01
};
static constexpr double REMEZ_QMIN_1_600 = 1.488546685679440e+00;
static constexpr double REMEZ_QMAX_1_600 = 2.941280114076638e+02;

// Range [1e-4, 1]
static const std::vector<double> REMEZ_P_1E4_1 = {
    9.710038853872469e+01,
    6.627242874895056e+03,
   -1.290965918127410e+04,
    8.356592957269102e+03
};
static const std::vector<double> REMEZ_Q_1E4_1 = {
    1.000000000000000e+00,
    1.847308811770853e+03
};
static constexpr double REMEZ_QMIN_1E4_1 = 1.184730881177085e+00;
static constexpr double REMEZ_QMAX_1E4_1 = 1.848308811770853e+03;


TEST_F(OpsTest, RemezRationalApprox_1_600) {
    const CC& cc = ctx->cc;
    const size_t slots = cc->GetRingDimension() / 2;

    double test_val = 100.0;
    auto pt = encode(cc, std::vector<double>(slots, test_val));
    auto ct = encrypt(cc, pt, ctx->pk());

    Ctx result = eval_rational_approx(cc, ct,
        REMEZ_P_1_600, REMEZ_Q_1_600,
        REMEZ_QMIN_1_600, REMEZ_QMAX_1_600,
        ctx->pk(), slots, /*gs_iters=*/7);

    auto dec = decrypt(cc, result, ctx->sk());
    double expected = 1.0 / std::sqrt(test_val);

    std::cout << "Remez [1,600] at x=" << test_val << std::endl;
    std::cout << "  Expected: " << expected << std::endl;
    std::cout << "  Got:      " << dec[0] << std::endl;
    std::cout << "  Abs err:  " << std::abs(dec[0] - expected) << std::endl;

    EXPECT_NEAR(dec[0], expected, 0.1);
}

TEST_F(OpsTest, RemezRationalApprox_1e4_1) {
    const CC& cc = ctx->cc;
    const size_t slots = cc->GetRingDimension() / 2;

    double test_val = 0.5;
    auto pt = encode(cc, std::vector<double>(slots, test_val));
    auto ct = encrypt(cc, pt, ctx->pk());

    Ctx result = eval_rational_approx(cc, ct,
        REMEZ_P_1E4_1, REMEZ_Q_1E4_1,
        REMEZ_QMIN_1E4_1, REMEZ_QMAX_1E4_1,
        ctx->pk(), slots, /*gs_iters=*/7);

    auto dec = decrypt(cc, result, ctx->sk());
    double expected = 1.0 / std::sqrt(test_val);

    std::cout << "Remez [1e-4,1] at x=" << test_val << std::endl;
    std::cout << "  Expected: " << expected << std::endl;
    std::cout << "  Got:      " << dec[0] << std::endl;
    std::cout << "  Abs err:  " << std::abs(dec[0] - expected) << std::endl;

    EXPECT_NEAR(dec[0], expected, 0.3);
}

TEST_F(OpsTest, TaylorInvSqrt_1e4_1) {
    const CC& cc = ctx->cc;
    const size_t slots = cc->GetRingDimension() / 2;

    double a = 1e-4, b = 1.0;
    double z0 = (a + b) / 2.0;

    // Precompute Taylor coefficients once (done at model init time)
    std::vector<double> taylor_coeffs = taylor_inv_sqrt_coeffs(z0);

    // Test at a point near z0 where Taylor is accurate
    double test_val = 0.45;
    auto pt = encode(cc, std::vector<double>(slots, test_val));
    auto ct = encrypt(cc, pt, ctx->pk());

    Ctx result = eval_taylor_inv_sqrt(cc, ct, taylor_coeffs, z0);

    auto dec = decrypt(cc, result, ctx->sk());

    // Compute expected Taylor value in plaintext
    double u = test_val - z0;
    double expected_taylor = taylor_coeffs[0] + taylor_coeffs[1]*u
                           + taylor_coeffs[2]*u*u + taylor_coeffs[3]*u*u*u;
    double expected_true = 1.0 / std::sqrt(test_val);

    std::cout << "Taylor inv_sqrt around z0=" << z0 << " at x=" << test_val << std::endl;
    std::cout << "  True 1/sqrt(x):    " << expected_true << std::endl;
    std::cout << "  Taylor (plain):    " << expected_taylor << std::endl;
    std::cout << "  Taylor (HE):       " << dec[0] << std::endl;
    std::cout << "  HE vs plain err:   " << std::abs(dec[0] - expected_taylor) << std::endl;
    std::cout << "  Taylor approx err: " << std::abs(expected_taylor - expected_true) << std::endl;

    // HE result should match plaintext Taylor evaluation closely
    EXPECT_NEAR(dec[0], expected_taylor, 1e-4);
}
