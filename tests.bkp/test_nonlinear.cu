// test_nonlinear.cu
// Tests SiLU, Softmax, and Norm by comparing FHE outputs to
// plaintext reference implementations.
//
// Reference functions (ref_silu, ref_softmax, ref_norm) mirror the Go
// plaintext functions SiLUPlaintext, SoftmaxPlaintext, NormPlaintext
// in cachemir-go/nonlinear.go.
//
// ── Slot grouping conventions ─────────────────────────────────────────────
//
// Norm:
//   Go's NormPlaintext uses STRIDED groups with stride intRot = slots/hidDim.
//   Group g ∈ [0, intRot) contains slots {g, g+intRot, g+2*intRot, ...,
//   g+(hidDim-1)*intRot}.  The FHE rotation pattern matches this exactly:
//   rotations by intRot, 2*intRot, 4*intRot, ... build the broadcast sum.
//
// Softmax:
//   Go's SoftmaxPlaintext groups by i % 8 (the sum of all exp within each
//   residue class mod 8 should equal 1 after normalization).  The FHE
//   rotation steps {1024/j | j=1,2,4,...,128} = {1024,512,...,8} are a
//   dyadic tree that sums all slots sharing the same i % 8 residue.

#include <gtest/gtest.h>
#include "llama.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

Ctx bootstrap_to(LlamaInference&, const Ctx&, uint32_t);

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
// These mirror Go's SiLUPlaintext, SoftmaxPlaintext, NormPlaintext exactly.

// Go: SiLUPlaintext — elementwise x/(exp(-x)+1)
static std::vector<double> ref_silu(const std::vector<double>& x) {
    std::vector<double> y(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        y[i] = x[i] / (std::exp(-x[i]) + 1.0);
    return y;
}

// Go: SoftmaxPlaintext — softmax over strided groups (i % 8)
// Group g contains all slots j where j % 8 == g.
// sum[g] = Σ exp(x[j]) for j%8==g.  result[i] = exp(x[i]) / sum[i%8].
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

// Go: NormPlaintext — x / sqrt(var(x)) per STRIDED group.
// intRot = n / hidDim.  Group g ∈ [0,intRot) contains slots
// {g, g+intRot, g+2*intRot, ..., g+(hidDim-1)*intRot}.
// result[i] = x[i] / sqrt( variance_of_group(i % intRot) )
// Note: no mean subtraction in the final multiply — the mean is only
// used internally to compute the variance (matches Go and the FHE impl).
static std::vector<double> ref_norm(const std::vector<double>& x, int hidDim) {
    int n      = (int)x.size();
    int intRot = n / hidDim;  // stride = number of groups
    std::vector<double> result(n, 0.0);

    for (int g = 0; g < intRot; ++g) {
        // mean over group g
        double mean = 0.0;
        for (int j = 0; j < hidDim; ++j)
            mean += x[g + j * intRot];
        mean /= hidDim;

        // variance over group g
        double var = 0.0;
        for (int j = 0; j < hidDim; ++j) {
            double d = x[g + j * intRot] - mean;
            var += d * d;
        }
        var /= hidDim;

        double inv_std = 1.0 / std::sqrt(var + 1e-8);
        for (int j = 0; j < hidDim; ++j)
            result[g + j * intRot] = x[g + j * intRot] * inv_std;
    }
    return result;
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
        return encrypt(llama.cc(), llama.cc()->MakeCKKSPackedPlaintext(msg), llama.fhe->pk());
    }

    std::vector<double> decrypt_msg(const Ctx& ct) {
        return decrypt(llama.cc(), ct, llama.fhe->sk());
    }
};

// ── SiLU ─────────────────────────────────────────────────────────────────
// NOTE on CKKS bootstrap-mode message range:
// The LlamaInference context uses bootstrap-mode CKKS (59-bit scaling,
// HEStd_NotSet).  OpenFHE's Decode() sanity check rejects messages whose
// magnitude is inconsistent with the scaling after non-trivial FHE
// operations.  Empirically (matching test_bootstrap.cu which uses
// scale=0.5), encoded values must satisfy |m| ≤ 0.5 for a
// freshly-encrypted ciphertext evaluated through EvalChebyshevSeries.
// Tests therefore use inputs/outputs in [-0.5, 0.5].
// CKKS noise for logN=12 bootstrap-mode parameters is ~1e-3; tolerances
// are set accordingly.

TEST_F(NonlinearTest, SiLU_SmallValues) {
    // Inputs in [-0.5, 0.5] → SiLU output in [-0.19, 0.31], within range.
    std::vector<double> msg(llama.slots);
    for (int i = 0; i < llama.slots; ++i)
        msg[i] = -0.5 + 1.0 * i / (double)llama.slots;

    Ctx ct  = encrypt_msg(msg);
    Ctx out = silu(llama, ct);

    auto result = decrypt_msg(out);
    auto ref    = ref_silu(msg);

    double err = max_abs_err(result, ref);
    std::cout << "SiLU max_abs_err=" << err << "\n";
    EXPECT_LT(err, 1e-3) << "SiLU approximation too noisy";
}

TEST_F(NonlinearTest, SiLU_AtZero) {
    // SiLU(0) = 0; tolerance 1e-3 matches bootstrap-mode noise floor.
    std::vector<double> msg(llama.slots, 0.0);
    Ctx ct  = encrypt_msg(msg);
    Ctx out = silu(llama, ct);

    auto result = decrypt_msg(out);
    double err = max_abs_err(result, std::vector<double>(llama.slots, 0.0));
    std::cout << "SiLU(0) max_abs_err=" << err << "\n";
    EXPECT_LT(err, 1e-3) << "SiLU(0) error too high";
}

TEST_F(NonlinearTest, SiLU_Positive_Approx_Identity) {
    // SiLU(0.4) ≈ 0.239; output bounded within CKKS-safe range.
    std::vector<double> msg(llama.slots, 0.4);
    Ctx ct  = encrypt_msg(msg);
    Ctx out = silu(llama, ct);

    auto result = decrypt_msg(out);
    auto ref    = ref_silu(msg);

    double err = max_abs_err(result, ref);
    std::cout << "SiLU(0.4) max_abs_err=" << err << "\n";
    EXPECT_LT(err, 1e-3) << "SiLU(0.4) error too high: " << err;
}

// ── Softmax ───────────────────────────────────────────────────────────────

// The FHE softmax normalizes over strided groups (i % 8).
// Group g contains all slots j where j % 8 == g.
// After normalization, Σ result[j] for j%8==g should be ≈ 1.
TEST_F(NonlinearTest, Softmax_SumsToOne) {
    // Inputs scaled to stay within bootstrap-mode CKKS safe range.
    std::vector<double> msg(llama.slots);
    for (int i = 0; i < llama.slots; ++i)
        msg[i] = 0.05 * (i % 8);  // [0, 0.35] — within CKKS-safe range

    Ctx ct  = encrypt_msg(msg);
    Ctx out = softmax(llama, ct, /*btp_level=*/14, /*temp=*/0);

    auto result = decrypt_msg(out);

    // For each residue class g = 0..7, sum over all slots j where j%8==g.
    // Each such sum should be ≈ 1.
    for (int g = 0; g < 8; ++g) {
        double sum = 0.0;
        for (int j = g; j < llama.slots; j += 8)
            sum += result[j];
        EXPECT_NEAR(sum, 1.0, 0.1)
            << "Strided group " << g << " sum=" << sum;
    }
}

TEST_F(NonlinearTest, Softmax_MaxSlot) {
    // Input where slot 3 dominates within its group (i % 8 == 3).
    // Value 0.4 keeps output bounded within CKKS-safe range.
    std::vector<double> msg(llama.slots, 0.0);
    msg[3] = 0.4;

    Ctx ct  = encrypt_msg(msg);
    Ctx out = softmax(llama, ct, 14, 0);

    auto result = decrypt_msg(out);
    auto ref    = ref_softmax(msg);

    // Compare first 64 slots (8 groups × 8 slots)
    double err = max_abs_err(
        std::vector<double>(result.begin(), result.begin() + 64),
        std::vector<double>(ref.begin(),    ref.begin()    + 64));

    std::cout << "Softmax MaxSlot err=" << err << "\n";
    EXPECT_LT(err, 0.1);
}

// ── Norm ─────────────────────────────────────────────────────────────────
// The FHE norm uses STRIDED groups (stride intRot = slots/hidDim).
// Group g ∈ [0, intRot) contains slots {g, g+intRot, g+2*intRot, ...}.

TEST_F(NonlinearTest, Norm_MeanZero) {
    // sin is symmetric around each half-period; with hidDim evenly spaced
    // points at stride intRot in [0,2π), the group mean is 0.
    // Scale to 0.3 to stay within CKKS-safe range.
    std::vector<double> msg(llama.slots);
    for (int i = 0; i < llama.slots; ++i)
        msg[i] = 0.3 * std::sin(2.0 * M_PI * i / (double)llama.slots);

    Ctx ct  = encrypt_msg(msg);
    Ctx out = norm(llama, ct, /*btp_level=*/9);

    auto result = decrypt_msg(out);
    const int hD     = llama.size.hidDim;
    const int intRot = llama.slots / hD;

    // Check mean of strided group 0: {result[0], result[intRot], ..., result[(hD-1)*intRot]}
    double mean = 0.0;
    for (int j = 0; j < hD; ++j)
        mean += result[j * intRot];
    mean /= hD;
    EXPECT_NEAR(mean, 0.0, 0.1)
        << "Norm output mean of group 0 should be ≈ 0, got " << mean;
}

TEST_F(NonlinearTest, Norm_MatchesReference) {
    // Matches Go's NormPlaintext exactly (strided grouping, no mean subtraction).
    // Input scaled to 0.3 to stay within bootstrap-mode CKKS-safe range.
    std::vector<double> msg(llama.slots);
    for (int i = 0; i < llama.slots; ++i)
        msg[i] = 0.3 * std::sin((double)i);

    Ctx ct  = encrypt_msg(msg);
    Ctx out = norm(llama, ct, 9);

    auto result = decrypt_msg(out);
    auto ref    = ref_norm(msg, llama.size.hidDim);

    double rel = rel_err(result, ref);
    std::cout << "Norm relative error: " << rel << "\n";
    EXPECT_LT(rel, 0.1)
        << "Norm output diverges from NormPlaintext reference: rel=" << rel;
}

TEST_F(NonlinearTest, Norm_UnitVariance) {
    // After norm, variance within each strided group should be ≈ 1.
    // var(result_g) = var(x_g) / var(x_g) = 1 (since result[i]=x[i]/sqrt(var_g)).
    std::vector<double> msg(llama.slots);
    for (int i = 0; i < llama.slots; ++i)
        msg[i] = 0.3 * std::sin(i);

    Ctx ct  = encrypt_msg(msg);
    Ctx out = norm(llama, ct, 9);

    auto result = decrypt_msg(out);
    const int hD     = llama.size.hidDim;
    const int intRot = llama.slots / hD;

    // Check variance of strided group 0
    double mean = 0.0;
    for (int j = 0; j < hD; ++j)
        mean += result[j * intRot];
    mean /= hD;

    double var = 0.0;
    for (int j = 0; j < hD; ++j) {
        double d = result[j * intRot] - mean;
        var += d * d;
    }
    var /= hD;

    EXPECT_NEAR(var, 1.0, 0.3)
        << "Norm output variance of group 0 should be ≈ 1, got " << var;
}

// ── Go-compatibility tests ────────────────────────────────────────────────
// These tests explicitly verify that the FHE output matches the Go
// plaintext reference functions (SiLUPlaintext, SoftmaxPlaintext,
// NormPlaintext from cachemir-go/nonlinear.go).
// The FHE result should agree with the Go plaintext up to CKKS noise.

TEST_F(NonlinearTest, GoCompat_SiLU_MatchesSiLUPlaintext) {
    // Go: SiLUPlaintext(x) = x/(exp(-x)+1) elementwise.
    // FHE uses a deg-127 Chebyshev approximation on [-20,20].
    // Bootstrap-mode CKKS requires |output| ≤ 0.5; use x ∈ [-0.5, 0.5].
    std::vector<double> msg(llama.slots);
    for (int i = 0; i < llama.slots; ++i)
        msg[i] = -0.5 + 1.0 * i / (double)llama.slots;

    Ctx ct  = encrypt_msg(msg);
    Ctx out = silu(llama, ct);

    auto fhe_result = decrypt_msg(out);
    auto go_ref     = ref_silu(msg);  // mirrors Go's SiLUPlaintext

    double err = max_abs_err(fhe_result, go_ref);
    std::cout << "GoCompat SiLU max_abs_err=" << err << "\n";
    EXPECT_LT(err, 1e-3)
        << "FHE SiLU disagrees with Go SiLUPlaintext by " << err;
}

TEST_F(NonlinearTest, GoCompat_Softmax_MatchesSoftmaxPlaintext) {
    // Go: SoftmaxPlaintext groups by i%8, normalizes exp within each group.
    // FHE: same grouping via rotation-sum with steps {1024,512,...,8}.
    // Scale to 0.2 to keep values within bootstrap-mode CKKS-safe range.
    std::vector<double> msg(llama.slots);
    for (int i = 0; i < llama.slots; ++i)
        msg[i] = 0.2 * std::sin(0.1 * i);

    Ctx ct  = encrypt_msg(msg);
    Ctx out = softmax(llama, ct, 14, 0);

    auto fhe_result = decrypt_msg(out);
    auto go_ref     = ref_softmax(msg);  // mirrors Go's SoftmaxPlaintext

    double rel = rel_err(fhe_result, go_ref);
    std::cout << "GoCompat Softmax rel_err=" << rel << "\n";
    EXPECT_LT(rel, 0.1)
        << "FHE Softmax disagrees with Go SoftmaxPlaintext by " << rel;
}

TEST_F(NonlinearTest, GoCompat_Norm_MatchesNormPlaintext) {
    // Go: NormPlaintext groups by i%intRot (strided), computes x/sqrt(var_g).
    // FHE: same algorithm — rotate-sum for mean/var, Newton+Goldschmidt for inv_sqrt.
    // Scale to 0.3 to stay within bootstrap-mode CKKS-safe range.
    std::vector<double> msg(llama.slots);
    for (int i = 0; i < llama.slots; ++i)
        msg[i] = 0.3 * std::cos(0.05 * i) + 0.1 * std::sin(0.2 * i);

    Ctx ct  = encrypt_msg(msg);
    Ctx out = norm(llama, ct, 9);

    auto fhe_result = decrypt_msg(out);
    auto go_ref     = ref_norm(msg, llama.size.hidDim);  // mirrors Go's NormPlaintext

    double rel = rel_err(fhe_result, go_ref);
    std::cout << "GoCompat Norm rel_err=" << rel << "\n";
    EXPECT_LT(rel, 0.1)
        << "FHE Norm disagrees with Go NormPlaintext by " << rel;
}
