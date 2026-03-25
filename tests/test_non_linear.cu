#include <gtest/gtest.h>
#include "test_non_linear.h"
#include "nonlinear.h"
#include <algorithm>
#include <iomanip>
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

    const auto config = GELU_ENCLLM_GPT2;

    const std::vector<double> xs = { -3.0, -0.5, 0.0, 1.0, 4.0 };
    const std::vector<double> expect = gelu_expect(xs);

    for (size_t t = 0; t < xs.size(); ++t) {
        auto pt = cc->MakeCKKSPackedPlaintext(std::vector<double>(S, xs[t]));
        auto ct = cc->Encrypt(inf.fhe->pk(), pt);

        auto out = gelu(inf, ct->Clone(), config);

        double got = decrypt(cc, out, inf.fhe->sk())[0];
        std::cout << "  result=" << got << " (expected " << expect[t] << ")\n";
        EXPECT_NEAR(got, expect[t], 0.15);
    }
}

TEST_F(NonLinearTest, NormApprox) {
    const CC& cc = inf.cc();
    const int S  = inf.slots;

    const double raw[]   = { 1.0, 2.0, 3.0, 4.0, 5.0 };
    const double true_mean = 3.0;
    const double true_std  = std::sqrt(2.0);
    double expect[5];
    for (int j = 0; j < 5; ++j) expect[j] = (raw[j] - true_mean) / true_std;

    std::vector<double> msg(S);
    for (int i = 0; i < S; ++i) msg[i] = raw[i % 5];

    auto pt = cc->MakeCKKSPackedPlaintext(msg);
    auto ct = cc->Encrypt(inf.fhe->pk(), pt);
    std::cout << "Level before norm: " << level_of(ct) << "\n";

    auto out = norm(inf, ct, /*target_level_after_btp=*/14, NORM_ENCLLM_GPT2);
    std::cout << "Level after  norm: " << level_of(out) << "\n";

    auto res = decrypt(cc, out, inf.fhe->sk());

    for (int j = 0; j < 5; ++j) {
        double got = res[j]; 
        std::cout << "  norm(raw[" << j << "]=" << raw[j] << ") = "
                  << got << "  (expected " << expect[j] << ")\n";
        EXPECT_NEAR(got, expect[j], 0.1);
    }
}

TEST_F(NonLinearTest, SoftmaxWithOracleMax) {
    const CC& cc = inf.cc();
    const int S  = inf.slots;           // 2048

    const int seq_dim    = 4;           // 2 sign calls without oracle → fails;
    const int stride     = S / seq_dim; // 512   with oracle → free
    const int r          = 7;
    const int gs_iters   = 10;
    const int target_btp = 14;

    const double inp[] = { -2.0, 0.5, 1.5, -0.5 };
    const double vmax  = *std::max_element(inp, inp + seq_dim);  // 1.5
    double e[4], esum = 0.0;
    for (int i = 0; i < seq_dim; ++i) { e[i] = std::exp(inp[i] - vmax); esum += e[i]; }
    double s[4];
    for (int i = 0; i < seq_dim; ++i) s[i] = e[i] / esum;

    std::vector<double> msg(S);
    for (int i = 0; i < seq_dim; ++i)
        std::fill(msg.begin() + i*stride, msg.begin() + (i+1)*stride, inp[i]);
    auto pt = cc->MakeCKKSPackedPlaintext(msg);
    Ctx x_in = cc->Encrypt(inf.fhe->pk(), pt);
    std::cout << "Encrypt x_in          level=" << level_of(x_in) << "\n";

    Ctx x_max = encrypt_const(cc, vmax, (size_t)S, inf.fhe->pk());
    std::cout << "Encrypt oracle max    level=" << level_of(x_max)
              << "  val=" << decrypt(cc, x_max, inf.fhe->sk())[0] << "\n\n";

    Ctx y = softmax(inf, x_in, target_btp, r, gs_iters, seq_dim, x_max);
    std::cout << "softmax() result      level=" << level_of(y) << "\n";

    auto res = decrypt(cc, y, inf.fhe->sk());
    std::cout << "  result : [" << res[0]          << ", " << res[stride]
              << ", "           << res[2*stride]    << ", " << res[3*stride] << "]\n"
              << "  expect : [" << s[0] << ", " << s[1] << ", " << s[2] << ", " << s[3] << "]\n";

    EXPECT_NEAR(res[0],         s[0], 0.05);
    EXPECT_NEAR(res[stride],    s[1], 0.05);
    EXPECT_NEAR(res[2*stride],  s[2], 0.05);
    EXPECT_NEAR(res[3*stride],  s[3], 0.05);

    double total = res[0] + res[stride] + res[2*stride] + res[3*stride];
    std::cout << "  sum    : " << total << "  (expected 1.0)\n";
    EXPECT_NEAR(total, 1.0, 0.05);
}
