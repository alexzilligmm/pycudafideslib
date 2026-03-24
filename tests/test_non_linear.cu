#include <gtest/gtest.h>
#include "test_non_linear.h"
#include <cmath>
#include <vector>

std::vector<double> linspace(double start, double end, size_t num) {
    std::vector<double> result(num);
    double step = (end - start) / (num - 1);
    for (size_t i = 0; i < num; ++i) {
        result[i] = start + i * step;
    }
    return result;
}

std::vector<double> gelu_expect(const std::vector<double>& xs) {
    std::vector<double> result(xs.size());
    for (size_t i = 0; i < xs.size(); ++i) {
        double x = xs[i];
        result[i] = 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (x + 0.044715 * std::pow(x, 3))));
    }
    return result;
}

TEST_F(NonLinearTest, SignApprox) {
    const CC& cc = inf.cc();
    const int S  = inf.slots;

    const std::vector<double> xs     = {  0.8,  0.3, -0.5, -0.9 };
    const std::vector<double> expect = {  1.0,  1.0, -1.0, -1.0 };

    for (size_t i = 0; i < xs.size(); ++i) {
        std::vector<double> msg(S, xs[i]);
        auto pt = cc->MakeCKKSPackedPlaintext(msg);
        auto ct = cc->Encrypt(inf.fhe->pk(), pt);

        auto result = sign(inf, ct);
        std::cout << "Level after sign: " << level_of(result) << std::endl;
        double got = decrypt(cc, result, inf.fhe->sk())[0];
        std::cout << "sign(" << xs[i] << ") = " << got
                  << "  (expected " << expect[i] << ")\n";

        EXPECT_NEAR(got, expect[i], 0.05);
    }
}

TEST_F(NonLinearTest, LtApprox) {
    const CC& cc = inf.cc();
    const int S  = inf.slots;

    const std::vector<double> xs     = {  0.8,  0.3, -0.5, -0.9 };
    const std::vector<double> expect = {  0.0,  0.0, 1.0, 1.0 };

    for (size_t i = 0; i < xs.size(); ++i) {
        std::vector<double> msg(S, xs[i]);
        auto pt = cc->MakeCKKSPackedPlaintext(msg);
        auto ct = cc->Encrypt(inf.fhe->pk(), pt);

        auto result = lt_function(inf, ct, 0.0, 1.0);

        double got = decrypt(cc, result, inf.fhe->sk())[0];
        std::cout << "lt_function(" << xs[i] << ") = " << got
                  << "  (expected " << expect[i] << ")\n";

        EXPECT_NEAR(got, expect[i], 0.05);
    }
}

TEST_F(NonLinearTest, LtApproxWithBts) {
    const CC& cc = inf.cc();
    const int S  = inf.slots;

    double x_val = 0.6;
    auto pt = cc->MakeCKKSPackedPlaintext(std::vector<double>(S, x_val));
    auto ct = cc->Encrypt(inf.fhe->pk(), pt);

    double thresholds[] = { -4.0, -1.95, 3.0 };
    double expects[]    = {  0.0,  0.0,  1.0 };
    double factor = 0.1;
    Ctx results[3];

    for (int i = 0; i < 3; ++i) {
        results[i] = lt_function(inf, ct, thresholds[i], factor);
        std::cout << "lt(x, " << thresholds[i] << "): level=" << level_of(results[i]);

        double got = decrypt(cc, results[i], inf.fhe->sk())[0];
        std::cout << " val=" << got << " (expected " << expects[i] << ")\n";
        EXPECT_NEAR(got, expects[i], 0.05);

        results[i] = cc->EvalBootstrap(results[i]);
        std::cout << "  post-bts level=" << level_of(results[i]) << "\n";
    }

    match_level(cc, results[0], results[1]);
    Ctx ind = cc->EvalSub(results[2], results[1]); 

    double got = decrypt(cc, ind, inf.fhe->sk())[0];
    std::cout << "\ninterval indicator [-1.95, 3) = " << got << " (expected 1)\n";
    EXPECT_NEAR(got, 1.0, 0.1);
}

TEST_F(NonLinearTest, GeLUApproxWithBts) {
    const CC& cc = inf.cc();
    const int S  = inf.slots;

    const std::vector<double> poly_f0 = {
        -0.5054031199708174,
        -0.42226581151983866,
        -0.11807612951181953,
        -0.011034134030615728
    };
    const std::vector<double> poly_f1 = {
        0.008526321541038084,
        0.5,
        0.3603292692789629,
        0.0,
       -0.037688200365904236,
        0.0,
        0.0018067462606141187
    };

    const double factor = 0.1;

    const std::vector<double> xs = { -3.0, -0.5, 0.0, 1.0, 4.0 };
    const std::vector<double> expect = gelu_expect(xs);

    for (size_t t = 0; t < xs.size(); ++t) {
        auto pt = cc->MakeCKKSPackedPlaintext(std::vector<double>(S, xs[t]));
        auto ct = cc->Encrypt(inf.fhe->pk(), pt);
        std::cout << "\n--- gelu(" << xs[t] << ") ---\n";

        Ctx lt_m4   = lt_function(inf, ct, -4.0,  factor);
        std::cout << "  lt_m4   level=" << level_of(lt_m4) << "\n";
        Ctx lt_m195 = lt_function(inf, ct, -1.95, factor);
        std::cout << "  lt_m195 level=" << level_of(lt_m195) << "\n";
        Ctx lt_p3   = lt_function(inf, ct,  3.0,  factor);
        std::cout << "  lt_p3   level=" << level_of(lt_p3) << "\n";

        lt_m4   = cc->EvalBootstrap(lt_m4);
        lt_m195 = cc->EvalBootstrap(lt_m195);
        lt_p3   = cc->EvalBootstrap(lt_p3);
        std::cout << "  post-bts levels: " << level_of(lt_m4)
                  << ", " << level_of(lt_m195)
                  << ", " << level_of(lt_p3) << "\n";

        match_level(cc, lt_m4, lt_m195);
        match_level(cc, lt_m195, lt_p3);
        Ctx ind_f0  = cc->EvalSub(lt_m195, lt_m4);    // [-4, -1.95)
        Ctx ind_f1  = cc->EvalSub(lt_p3, lt_m195);    // [-1.95, 3)
        Ctx ind_lin = cc->EvalNegate(lt_p3);           // [3, +inf)
        cc->RescaleInPlace(ind_lin);
        cc->EvalAddInPlace(ind_lin, 1.0);

        Ctx p0 = eval_polynomial_ps(cc, ct, poly_f0, inf.fhe->pk(), (size_t)S);
        Ctx p1 = eval_polynomial_ps(cc, ct, poly_f1, inf.fhe->pk(), (size_t)S);
        std::cout << "  poly levels: p0=" << level_of(p0)
                  << " p1=" << level_of(p1) << "\n";

        match_level(cc, ind_f0, p0);
        Ctx term0 = cc->EvalMult(ind_f0, p0);
        cc->RescaleInPlace(term0);

        match_level(cc, ind_f1, p1);
        Ctx term1 = cc->EvalMult(ind_f1, p1);
        cc->RescaleInPlace(term1);

        Ctx x_copy = ct;
        match_level(cc, ind_lin, x_copy);
        Ctx term2 = cc->EvalMult(ind_lin, x_copy);
        cc->RescaleInPlace(term2);

        match_level(cc, term0, term1);
        cc->EvalAddInPlace(term0, term1);
        match_level(cc, term0, term2);
        cc->EvalAddInPlace(term0, term2);

        double got = decrypt(cc, term0, inf.fhe->sk())[0];
        std::cout << "  result=" << got << " (expected " << expect[t] << ")\n";
        EXPECT_NEAR(got, expect[t], 0.15);
    }
}
