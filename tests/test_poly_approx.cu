#include <gtest/gtest.h>
#include "fideslib_wrapper.h"
#include <cmath>
#include <numeric>
#include "test_basics.h" // For OpsTest fixture and context setup
#include "ckks_primitives.h" // For the primitive implementations we're testing

TEST_F(OpsTest, ChebyshevCoefficients) {
    const CC& cc = ctx->cc;
    const size_t slots = cc->GetRingDimension() / 2;

    std::vector<double> F4_coeffs = {0.0, 315.0/128.0, 0.0, -420.0/128.0, 0.0, 378.0/128.0, 0.0, -180.0/128.0, 0.0, 35.0/128.0};

    std::vector<double> G4_coeffs = {0.0, 5850.0/1024.0, 0.0, -34974.0/1024.0, 0.0, 97015.0/1024.0, 0.0, -113492.0/1024.0, 0.0, 46623.0/1024.0};

    double test_val = 0.333;
    auto pt = encode(cc, std::vector<double>(slots, test_val));
    auto ct = encrypt(cc, pt, ctx->pk());
    std::cout << "level of ct: " << level_of(ct) << std::endl;

    std::cout << "level of ct: " << level_of(ct) << std::endl;
    auto binary_res = eval_polynomial_ps(cc, ct, F4_coeffs, ctx->pk(), slots);
    std::cout << "level of binary_res: " << level_of(binary_res) << std::endl;

    std::cout << "level of ct: " << level_of(ct) << std::endl;
    auto baseline = eval_polynomial(cc, ct, F4_coeffs);
    std::cout << "level of baseline: " << level_of(baseline) << std::endl;

    auto chebyshev_coeff = standard_to_chebyshev(F4_coeffs, -5, 5);

    std::cout << "level of ct: " << level_of(ct) << std::endl;
    auto chebyshev_res = eval_chebyshev(cc, ct, chebyshev_coeff, /*a*/-5, /*b*/5);
    std::cout << "level of chebyshev_res: " << level_of(chebyshev_res) << std::endl;


    auto dec_chebyshev = decrypt(cc, chebyshev_res, ctx->sk());
    double chebyshev_homomorphic_res = dec_chebyshev[0];

    auto dec = decrypt(cc, baseline, ctx->sk());
    double homomorphic_res = dec[0];

    double expected_res = 0.0;
    for (int i = static_cast<int>(F4_coeffs.size()) - 1; i >= 0; --i) {
        expected_res = expected_res * test_val + F4_coeffs[i];
    }

    std::cout << "Target Value: " << test_val << std::endl;
    std::cout << "Expected (Plaintext): " << expected_res << std::endl;
    std::cout << "Actual (Homomorphic): " << homomorphic_res << std::endl;
    std::cout << "Chebyshev (Homomorphic): " << chebyshev_homomorphic_res << std::endl;
    std::cout << "Binary (Homomorphic): " << decrypt(cc, binary_res, ctx->sk())[0] << std::endl;
    std::cout << "Absolute Error:       " << std::abs(homomorphic_res - expected_res) << std::endl;
    std::cout << "Chebyshev Absolute Error: " << std::abs(chebyshev_homomorphic_res - expected_res) << std::endl;
    std::cout << "Binary Absolute Error: " << std::abs(decrypt(cc, binary_res, ctx->sk())[0] - expected_res) << std::endl;

    EXPECT_NEAR(homomorphic_res, expected_res, 1e-5);

}