// test_nonlinear.cu
// Tests SiLU, Softmax, and Norm by comparing FHE outputs to
// plaintext reference implementations.

#include <gtest/gtest.h>
#include "llama.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

Ctx bootstrap_to(const LlamaInference&, const Ctx&, uint32_t);

// ── helpers ───────────────────────────────────────────────────────────────
static double max_abs_err(const std::vector<double>& a,
                           const std::vector<double>& b) {
    double m = 0.0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i)
        m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

static double rel_err(const std::vector<double>& a,
                       const std::vector<double>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        num += (a[i] - b[i]) * (a[i] - b[i]);
        den += b[i] * b[i];
    }
    return std::sqrt(num / (den + 1e-30));
}

// ── plaintext references ──────────────────────────────────────────────────
static std::vector<double> ref_silu(const std::vector<double>& x) {
    std::vector<double> y(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        y[i] = x[i] / (std::exp(-x[i]) + 1.0);
    return y;
}

// Softmax over groups of 8 (matches Go code's SoftmaxPlaintext)
static std::vector<double> ref_softmax(const std::vector<double>& x) {
    std::vector<double> y(x.size()), ex(x.size());
    std::vector<double> sums(8, 0.0);
    for (size_t i = 0; i < x.size(); ++i) {
        ex[i] = std::exp(x[i]);
        sums[i % 8] += ex[i];
    }
    for (size_t i = 0; i < x.size(); ++i)
        y[i] = ex[i] / sums[i % 8];
    return y;
}

// Layer norm over groups of `hidDim`
static std::vector<double> ref_norm(const std::vector<double>& x, int hidDim) {
    int n = (int)x.size();
    std::vector<double> y(n, 0.0);
    int groups = n / hidDim;
    for (int g = 0; g < groups; ++g) {
        double mean = 0.0, var = 0.0;
        for (int j = 0; j < hidDim; ++j)
            mean += x[g * hidDim + j];
        mean /= hidDim;
        for (int j = 0; j < hidDim; ++j) {
            double d = x[g * hidDim + j] - mean;
            var += d * d;
        }
        var  /= hidDim;
        double inv_std = 1.0 / std::sqrt(var + 1e-8);
        for (int j = 0; j < hidDim; ++j)
            y[g * hidDim + j] = (x[g * hidDim + j] - mean) * inv_std;
    }
    return y;
}

// ── fixture ───────────────────────────────────────────────────────────────
class NonlinearTest : public ::testing::Test {
protected:
    static constexpr int LOGN    = 12;
    static constexpr int HID_DIM = 32;    // small for fast tests
    static constexpr int EXP_DIM = 128;
    static constexpr int HEADS   = 4;
    static constexpr int SEQ_LEN = 64;

    LlamaInference llama;

    void SetUp() override {
        llama = make_llama(LOGN, HID_DIM, EXP_DIM, SEQ_LEN, HEADS, /*parallel=*/false);
    }

    Ctx encrypt_msg(const std::vector<double>& msg) {
        return llama.cc()->Encrypt(
            llama.fhe->pk(),
            llama.cc()->MakeCKKSPackedPlaintext(msg));
    }

    std::vector<double> decrypt_msg(const Ctx& ct) {
        return decrypt(llama.cc(), ct, llama.fhe->sk());
    }
};

// ── SiLU ─────────────────────────────────────────────────────────────────
TEST_F(NonlinearTest, SiLU_SmallValues) {
    // Small values in the center of the approximation interval [-20, 20]
    std::vector<double> msg(llama.slots);
    for (int i = 0; i < llama.slots; ++i)
        msg[i] = -2.0 + 4.0 * i / (double)llama.slots;

    Ctx ct  = encrypt_msg(msg);
    Ctx out = silu(llama, ct);

    auto result = decrypt_msg(out);
    auto ref    = ref_silu(msg);

    double err = max_abs_err(result, ref);
    std::cout << "SiLU max_abs_err=" << err << "\n";
    // Chebyshev deg-127 on [-20,20]: expect ≤1e-4 absolute error
    EXPECT_LT(err, 1e-4) << "SiLU approximation too noisy";
}

TEST_F(NonlinearTest, SiLU_AtZero) {
    // SiLU(0) = 0
    std::vector<double> msg(llama.slots, 0.0);
    Ctx ct  = encrypt_msg(msg);
    Ctx out = silu(llama, ct);

    auto result = decrypt_msg(out);
    for (int i = 0; i < llama.slots; ++i)
        EXPECT_NEAR(result[i], 0.0, 1e-4) << "SiLU(0) ≠ 0 at slot " << i;
}

TEST_F(NonlinearTest, SiLU_Positive_Approx_Identity) {
    // For large x, SiLU(x) ≈ x
    std::vector<double> msg(llama.slots, 10.0);
    Ctx ct  = encrypt_msg(msg);
    Ctx out = silu(llama, ct);

    auto result = decrypt_msg(out);
    auto ref    = ref_silu(msg);

    double err = max_abs_err(result, ref);
    EXPECT_LT(err, 0.1) << "SiLU(10) error too high: " << err;
}

// ── Softmax ───────────────────────────────────────────────────────────────
TEST_F(NonlinearTest, Softmax_SumsToOne) {
    // Softmax output should sum to ~1 within each group of 8
    std::vector<double> msg(llama.slots);
    for (int i = 0; i < llama.slots; ++i)
        msg[i] = 0.1 * (i % 8);  // gentle gradient

    Ctx ct  = encrypt_msg(msg);
    Ctx out = softmax(llama, ct, /*btp_level=*/14, /*temp=*/0);

    auto result = decrypt_msg(out);

    // Check group sums ≈ 1
    const int group = 8;
    int n_groups = llama.slots / group;
    for (int g = 0; g < std::min(n_groups, 10); ++g) {
        double sum = 0.0;
        for (int j = 0; j < group; ++j)
            sum += result[g * group + j];
        EXPECT_NEAR(sum, 1.0, 0.01)
            << "Group " << g << " sum=" << sum;
    }
}

TEST_F(NonlinearTest, Softmax_MaxSlot) {
    // Input where one slot dominates: softmax should concentrate there
    std::vector<double> msg(llama.slots, 0.0);
    // Group 0: slot 3 has value 5, others have 0
    msg[3] = 5.0;

    Ctx ct  = encrypt_msg(msg);
    Ctx out = softmax(llama, ct, 14, 0);

    auto result = decrypt_msg(out);
    auto ref    = ref_softmax(msg);

    double err = max_abs_err(
        std::vector<double>(result.begin(), result.begin() + 8),
        std::vector<double>(ref.begin(),    ref.begin()    + 8));

    std::cout << "Softmax MaxSlot err=" << err << "\n";
    EXPECT_LT(err, 0.05);
}

// ── Norm ─────────────────────────────────────────────────────────────────
TEST_F(NonlinearTest, Norm_MeanZero) {
    // After layer norm, each group should have mean ≈ 0
    std::vector<double> msg(llama.slots);
    for (int i = 0; i < llama.slots; ++i)
        msg[i] = std::sin(2.0 * M_PI * i / (double)llama.slots);

    Ctx ct  = encrypt_msg(msg);
    Ctx out = norm(llama, ct, /*btp_level=*/9);

    auto result = decrypt_msg(out);
    const int hD = llama.size.hidDim;

    // Check mean of first group ≈ 0
    double mean = 0.0;
    for (int j = 0; j < hD; ++j) mean += result[j];
    mean /= hD;
    EXPECT_NEAR(mean, 0.0, 0.05)
        << "Norm output mean should be ≈ 0, got " << mean;
}

TEST_F(NonlinearTest, Norm_MatchesReference) {
    std::vector<double> msg(llama.slots);
    for (int i = 0; i < llama.slots; ++i)
        msg[i] = 1.0 + 0.1 * i;

    Ctx ct  = encrypt_msg(msg);
    Ctx out = norm(llama, ct, 9);

    auto result = decrypt_msg(out);
    auto ref    = ref_norm(msg, llama.size.hidDim);

    double rel = rel_err(result, ref);
    std::cout << "Norm relative error: " << rel << "\n";
    EXPECT_LT(rel, 0.05)
        << "Norm output diverges from reference: rel=" << rel;
}

TEST_F(NonlinearTest, Norm_UnitVariance) {
    // After norm the variance per group should be ≈ 1
    std::vector<double> msg(llama.slots);
    for (int i = 0; i < llama.slots; ++i)
        msg[i] = 3.0 * std::sin(i);  // arbitrary signal

    Ctx ct  = encrypt_msg(msg);
    Ctx out = norm(llama, ct, 9);

    auto result = decrypt_msg(out);
    const int hD = llama.size.hidDim;

    // Variance of first group
    double mean = 0.0;
    for (int j = 0; j < hD; ++j) mean += result[j];
    mean /= hD;
    double var = 0.0;
    for (int j = 0; j < hD; ++j) {
        double d = result[j] - mean;
        var += d * d;
    }
    var /= hD;

    // Due to CKKS approximation errors, allow variance in [0.8, 1.2]
    EXPECT_NEAR(var, 1.0, 0.2)
        << "Norm output variance should be ≈ 1, got " << var;
}
