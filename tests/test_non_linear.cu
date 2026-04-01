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

        // Use bootstrap_to (respects total_depth) instead of raw EvalBootstrap
        results[i] = bootstrap_to(inf, results[i], DEPTH);
        std::cout << "  post-bts level=" << level_of(results[i]) << "\n";
    }

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
    const int hD = inf.size.hidDim;          // 1024 (padded)
    const int intRot = S / hD;               // 32 at logN=16

    inf.size.dim = hD;

    const double raw[]   = { 0.1, 0.2, 0.3, 0.4, 0.5 };
    constexpr int nRaw   = 5;

    double sum = 0.0;
    for (int h = 0; h < hD; ++h) sum += raw[h % nRaw];
    double true_mean = sum / hD;

    double sq_sum = 0.0;
    for (int h = 0; h < hD; ++h) {
        double d = raw[h % nRaw] - true_mean;
        sq_sum += d * d;
    }
    double true_var = sq_sum / hD;
    double true_std = std::sqrt(true_var + NORM_ENCLLM_GPT2.epsilon);

    double expect[nRaw];
    for (int j = 0; j < nRaw; ++j) expect[j] = (raw[j] - true_mean) / true_std;

    std::vector<double> msg(S);
    for (int h = 0; h < hD; ++h)
        for (int t = 0; t < intRot; ++t)
            msg[h * intRot + t] = raw[h % nRaw];
    for (int i = hD * intRot; i < S; ++i)
        msg[i] = 999.0;  // garbage — must be masked out by norm()

    auto pt = cc->MakeCKKSPackedPlaintext(msg);
    auto ct = cc->Encrypt(inf.fhe->pk(), pt);
    std::cout << "Level before norm: " << level_of(ct) << "\n";

    auto out = norm(inf, ct, /*target_level_after_btp=*/14, NORM_ENCLLM_GPT2);
    std::cout << "Level after  norm: " << level_of(out) << "\n";

    auto res = decrypt(cc, out, inf.fhe->sk());

    for (int j = 0; j < nRaw; ++j) {
        double got = res[j * intRot];
        std::cout << "  norm(raw[" << j << "]=" << raw[j] << ") = "
                  << got << "  (expected " << expect[j] << ")\n";
        EXPECT_NEAR(got, expect[j], 0.15);
    }
}

TEST_F(NonLinearTest, NormApproxWithPadding) {
    const CC& cc = inf.cc();
    const int S  = inf.slots;             // 32768

    const int paddedHD = 1024;
    const int realHD   = 768;
    inf.size.hidDim    = paddedHD;
    inf.size.dim = realHD;

    const int intRot = S / paddedHD;      // 32

    const double raw[] = { 0.1, 0.2, 0.3, 0.4, 0.5 };
    constexpr int nRaw = 5;

    double sum = 0.0;
    for (int h = 0; h < realHD; ++h) sum += raw[h % nRaw];
    double true_mean = sum / realHD;

    double sq_sum = 0.0;
    for (int h = 0; h < realHD; ++h) {
        double d = raw[h % nRaw] - true_mean;
        sq_sum += d * d;
    }
    double true_var = sq_sum / realHD;
    double true_std = std::sqrt(true_var + NORM_ENCLLM_GPT2.epsilon);

    double expect[nRaw];
    for (int j = 0; j < nRaw; ++j) expect[j] = (raw[j] - true_mean) / true_std;

    std::vector<double> msg(S, 0.0);
    for (int h = 0; h < realHD; ++h)
        for (int t = 0; t < intRot; ++t)
            msg[h * intRot + t] = raw[h % nRaw];

    auto pt = cc->MakeCKKSPackedPlaintext(msg);
    auto ct = cc->Encrypt(inf.fhe->pk(), pt);
    std::cout << "NormWithPadding: S=" << S
              << " paddedHD=" << paddedHD << " realHD=" << realHD
              << " intRot=" << intRot << "\n";
    std::cout << "  true_mean=" << true_mean << " true_var=" << true_var << "\n";
    std::cout << "  Level before norm: " << level_of(ct) << "\n";

    auto out = norm(inf, ct, /*target_level_after_btp=*/14, NORM_ENCLLM_GPT2);
    std::cout << "  Level after  norm: " << level_of(out) << "\n";

    auto res = decrypt(cc, out, inf.fhe->sk());

    double max_err = 0.0;
    for (int j = 0; j < nRaw; ++j) {
        double got = res[j * intRot];
        double err = std::abs(got - expect[j]);
        max_err = std::max(max_err, err);
        std::cout << "  norm(raw[" << j << "]=" << raw[j] << ") = "
                  << got << "  (expected " << expect[j]
                  << ", err=" << err << ")\n";
        EXPECT_NEAR(got, expect[j], 0.2);
    }

    double pad_val = res[realHD * intRot];
    std::cout << "  padded position [" << realHD << "] = " << pad_val << "\n";
    std::cout << "  max_err across real positions = " << max_err << "\n";
}

static void softmax_test_setup(
        const CC& cc, int S, int seq_dim,
        const double* inp,
        std::vector<double>& expected,
        std::vector<double>& msg) {
    int stride = S / seq_dim;
    double vmax = *std::max_element(inp, inp + seq_dim);
    expected.resize(seq_dim);
    double esum = 0.0;
    for (int i = 0; i < seq_dim; ++i) esum += std::exp(inp[i] - vmax);
    for (int i = 0; i < seq_dim; ++i) expected[i] = std::exp(inp[i] - vmax) / esum;

    msg.assign(S, 0.0);
    for (int i = 0; i < seq_dim; ++i)
        std::fill(msg.begin() + i*stride, msg.begin() + (i+1)*stride, inp[i]);
}

static void softmax_check(const std::vector<double>& res,
                           const std::vector<double>& expected,
                           int stride, int seq_dim, double tol) {
    std::cout << "  result : [";
    for (int i = 0; i < seq_dim; ++i)
        std::cout << res[i * stride] << (i < seq_dim-1 ? ", " : "");
    std::cout << "]\n  expect : [";
    for (int i = 0; i < seq_dim; ++i)
        std::cout << expected[i] << (i < seq_dim-1 ? ", " : "");
    std::cout << "]\n";

    double total = 0.0;
    for (int i = 0; i < seq_dim; ++i) {
        EXPECT_NEAR(res[i * stride], expected[i], tol);
        total += res[i * stride];
    }
    std::cout << "  sum    : " << total << "  (expected 1.0)\n";
    EXPECT_NEAR(total, 1.0, tol);
}

TEST_F(NonLinearTest, SoftmaxWithOracleMax) {
    const CC& cc = inf.cc();
    const int S  = inf.slots;

    SoftmaxConfig cfg = SOFTMAX_ENCLLM_GPT2;
    cfg.seq_dim = 4;
    const int stride = S / cfg.seq_dim;

    const double inp[] = { -2.0, 0.5, 1.5, -0.5 };
    std::vector<double> expected, msg;
    softmax_test_setup(cc, S, cfg.seq_dim, inp, expected, msg);

    auto pt = cc->MakeCKKSPackedPlaintext(msg);
    Ctx x_in = cc->Encrypt(inf.fhe->pk(), pt);
    double vmax = *std::max_element(inp, inp + cfg.seq_dim);
    Ctx x_max = encrypt_const(cc, vmax, (size_t)S, inf.fhe->pk());
    std::cout << "Encrypt x_in level=" << level_of(x_in)
              << "  oracle max=" << vmax << "\n";

    Ctx y = softmax(inf, x_in, cfg, x_max);
    std::cout << "softmax() result level=" << level_of(y) << "\n";

    auto res = decrypt(cc, y, inf.fhe->sk());
    softmax_check(res, expected, stride, cfg.seq_dim, 0.05);
}

TEST_F(NonLinearTest, SoftmaxCausalMask) {
    const CC& cc = inf.cc();
    const int S  = inf.slots;

    SoftmaxConfig cfg = SOFTMAX_ENCLLM_GPT2;
    cfg.seq_dim = 4;
    const int stride = S / cfg.seq_dim;

    const double inp[] = { 1.0, 2.0, 3.0, 4.0 };

    const double mask_val = -50.0;
    std::vector<double> mask_vec(S, 0.0);
    std::fill(mask_vec.begin() + 3*stride, mask_vec.begin() + 4*stride, mask_val);

    double vmax = 3.0;  // max of visible positions
    double esum = std::exp(1.0 - vmax) + std::exp(2.0 - vmax) + std::exp(3.0 - vmax);
    std::vector<double> expected = {
        std::exp(1.0 - vmax) / esum,
        std::exp(2.0 - vmax) / esum,
        std::exp(3.0 - vmax) / esum,
        0.0,   // masked out
    };

    std::vector<double> msg(S, 0.0);
    for (int i = 0; i < cfg.seq_dim; ++i)
        std::fill(msg.begin() + i*stride, msg.begin() + (i+1)*stride, inp[i]);

    auto pt = cc->MakeCKKSPackedPlaintext(msg);
    Ctx x_in = cc->Encrypt(inf.fhe->pk(), pt);

    Ctx x_max = encrypt_const(cc, vmax, (size_t)S, inf.fhe->pk());

    Ptx causal_pt = cc->MakeCKKSPackedPlaintext(
        mask_vec, /*noiseScaleDeg=*/1, (uint32_t)level_of(x_in));

    std::cout << "Running softmax with causal mask...\n";
    Ctx y = softmax(inf, x_in, cfg, x_max, causal_pt);
    std::cout << "softmax() result level=" << level_of(y) << "\n";

    auto res = decrypt(cc, y, inf.fhe->sk());

    std::cout << "  result : [";
    for (int i = 0; i < cfg.seq_dim; ++i)
        std::cout << res[i * stride] << (i < cfg.seq_dim-1 ? ", " : "");
    std::cout << "]\n  expect : [";
    for (int i = 0; i < cfg.seq_dim; ++i)
        std::cout << expected[i] << (i < cfg.seq_dim-1 ? ", " : "");
    std::cout << "]\n";

    for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(res[i * stride], expected[i], 0.05);
    EXPECT_NEAR(res[3 * stride], 0.0, 0.01);

    double total = 0.0;
    for (int i = 0; i < 3; ++i) total += res[i * stride];
    std::cout << "  sum(visible) = " << total << "  (expected ~1.0)\n";
    EXPECT_NEAR(total, 1.0, 0.05);
}

TEST_F(NonLinearTest, GeLUNoBtp) {
    const CC& cc = inf.cc();
    const int S  = inf.slots;

    GeluConfig cfg = GELU_ENCLLM_GPT2;
    cfg.bootstrap_indicators = false;  // disable internal bootstraps

    double x_val = 1.0;
    double expect_val = gelu_expect({x_val})[0];

    auto pt = cc->MakeCKKSPackedPlaintext(std::vector<double>(S, x_val));
    auto ct = cc->Encrypt(inf.fhe->pk(), pt);
    std::cout << "Level before gelu (no btp): " << level_of(ct) << "\n";

    auto out = gelu(inf, ct, cfg);
    std::cout << "Level after  gelu (no btp): " << level_of(out) << "\n";

    double got = decrypt(cc, out, inf.fhe->sk())[0];
    std::cout << "  gelu(" << x_val << ") = " << got
              << "  (expected " << expect_val << ")\n";
    EXPECT_NEAR(got, expect_val, 0.15);
}

TEST_F(NonLinearTest, GeLUWithTargetLevel) {
    const CC& cc = inf.cc();
    const int S  = inf.slots;

    GeluConfig cfg = GELU_ENCLLM_GPT2;
    cfg.bootstrap_indicators = true;
    cfg.btp_target_level = 14;  // bootstrap to level 14

    double x_val = -0.5;
    double expect_val = gelu_expect({x_val})[0];

    auto pt = cc->MakeCKKSPackedPlaintext(std::vector<double>(S, x_val));
    auto ct = cc->Encrypt(inf.fhe->pk(), pt);

    auto out = gelu(inf, ct, cfg);
    std::cout << "Level after gelu (btp_target=14): " << level_of(out) << "\n";

    double got = decrypt(cc, out, inf.fhe->sk())[0];
    std::cout << "  gelu(" << x_val << ") = " << got
              << "  (expected " << expect_val << ")\n";
    EXPECT_NEAR(got, expect_val, 0.15);
}
