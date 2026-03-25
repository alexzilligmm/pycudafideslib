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

// Plaintext rational evaluation for reference
static double eval_rational_plain(double x,
                                   const std::vector<double>& p,
                                   const std::vector<double>& q) {
    double num = 0.0, den = 0.0;
    double xp = 1.0;
    for (size_t i = 0; i < p.size(); ++i) { num += p[i] * xp; xp *= x; }
    xp = 1.0;
    for (size_t i = 0; i < q.size(); ++i) { den += q[i] * xp; xp *= x; }
    return num / den;
}

TEST_F(OpsTest, RemezRationalApprox_1_600) {
    const CC& cc = ctx->cc;
    const size_t slots = cc->GetRingDimension() / 2;

    const double test_vals[] = { 1.5, 10.0, 100.0, 300.0, 590.0 };

    std::cout << "\n=== Remez (3,1) rational for 1/sqrt(x) over [1, 600] ===\n";
    for (double x_val : test_vals) {
        auto pt = encode(cc, std::vector<double>(slots, x_val));
        auto ct = encrypt(cc, pt, ctx->pk());
        std::cout << "  x=" << x_val << "  input level=" << level_of(ct) << "\n";

        Ctx result = eval_rational_approx(cc, ct,
            REMEZ_P_1_600, REMEZ_Q_1_600,
            REMEZ_QMIN_1_600, REMEZ_QMAX_1_600,
            ctx->pk(), slots, /*gs_iters=*/10);

        std::cout << "          output level=" << level_of(result) << "\n";

        auto dec = decrypt(cc, result, ctx->sk());
        double true_val  = 1.0 / std::sqrt(x_val);
        double remez_val = eval_rational_plain(x_val, REMEZ_P_1_600, REMEZ_Q_1_600);
        double he_val    = dec[0];

        std::cout << "          true=" << true_val
                  << "  remez_plain=" << remez_val
                  << "  HE=" << he_val
                  << "  |HE-true|=" << std::abs(he_val - true_val) << "\n";

        EXPECT_NEAR(he_val, remez_val, 0.15);
    }
}

TEST_F(OpsTest, RemezRationalApprox_1e4_1) {
    const CC& cc = ctx->cc;
    const size_t slots = cc->GetRingDimension() / 2;

    const double test_vals[] = { 1e-3, 0.01, 0.1, 0.5, 0.95 };

    std::cout << "\n=== Remez (3,1) rational for 1/sqrt(x) over [1e-4, 1] ===\n";
    for (double x_val : test_vals) {
        auto pt = encode(cc, std::vector<double>(slots, x_val));
        auto ct = encrypt(cc, pt, ctx->pk());
        std::cout << "  x=" << x_val << "  input level=" << level_of(ct) << "\n";

        Ctx result = eval_rational_approx(cc, ct,
            REMEZ_P_1E4_1, REMEZ_Q_1E4_1,
            REMEZ_QMIN_1E4_1, REMEZ_QMAX_1E4_1,
            ctx->pk(), slots, /*gs_iters=*/10);

        std::cout << "          output level=" << level_of(result) << "\n";

        auto dec = decrypt(cc, result, ctx->sk());
        double true_val  = 1.0 / std::sqrt(x_val);
        double remez_val = eval_rational_plain(x_val, REMEZ_P_1E4_1, REMEZ_Q_1E4_1);
        double he_val    = dec[0];

        std::cout << "          true=" << true_val
                  << "  remez_plain=" << remez_val
                  << "  HE=" << he_val
                  << "  |HE-true|=" << std::abs(he_val - true_val) << "\n";

        EXPECT_NEAR(he_val, remez_val, 0.5);
    }
}

TEST_F(OpsTest, TaylorInvSqrt_1e4_1) {
    const CC& cc = ctx->cc;
    const size_t slots = cc->GetRingDimension() / 2;

    double a = 1e-4, b = 1.0;
    double z0 = (a + b) / 2.0;
    std::vector<double> tc = taylor_inv_sqrt_coeffs(z0);

    const double test_vals[] = { 0.01, 0.2, 0.45, 0.55, 0.9 };

    std::cout << "\n=== Taylor deg-3 for 1/sqrt(x) around z0=" << z0
              << " over [1e-4, 1] ===\n";
    std::cout << "  coeffs: a0=" << tc[0] << " a1=" << tc[1]
              << " a2=" << tc[2] << " a3=" << tc[3] << "\n";

    for (double x_val : test_vals) {
        auto pt = encode(cc, std::vector<double>(slots, x_val));
        auto ct = encrypt(cc, pt, ctx->pk());
        std::cout << "  x=" << x_val << "  input level=" << level_of(ct) << "\n";

        Ctx result = eval_taylor_inv_sqrt(cc, ct, tc, z0);
        std::cout << "          output level=" << level_of(result) << "\n";

        auto dec = decrypt(cc, result, ctx->sk());

        double u = x_val - z0;
        double taylor_plain = tc[0] + tc[1]*u + tc[2]*u*u + tc[3]*u*u*u;
        double true_val = 1.0 / std::sqrt(x_val);
        double he_val   = dec[0];

        std::cout << "          true=" << true_val
                  << "  taylor_plain=" << taylor_plain
                  << "  HE=" << he_val
                  << "  |HE-taylor|=" << std::abs(he_val - taylor_plain)
                  << "  |taylor-true|=" << std::abs(taylor_plain - true_val) << "\n";

        EXPECT_NEAR(he_val, taylor_plain, 1e-3);
    }
}
