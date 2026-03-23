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

    auto baseline = eval_polynomial(cc, ct, F4_coeffs);

    auto dec = decrypt(cc, baseline, ctx->sk());
    double homomorphic_res = dec[0];





    
    double expected_res = 0.0;
    for (int i = static_cast<int>(F4_coeffs.size()) - 1; i >= 0; --i) {
        expected_res = expected_res * test_val + F4_coeffs[i];
    }

    std::cout << "Target Value: " << test_val << std::endl;
    std::cout << "Expected (Plaintext): " << expected_res << std::endl;
    std::cout << "Actual (Homomorphic): " << homomorphic_res << std::endl;
    std::cout << "Absolute Error:       " << std::abs(homomorphic_res - expected_res) << std::endl;

    EXPECT_NEAR(homomorphic_res, expected_res, 1e-5);

}
