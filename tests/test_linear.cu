#include <gtest/gtest.h>
#include "gpt2.h"
#include "fideslib_wrapper.h"
#include <cmath>
#include <vector>
#include <random>

static constexpr int HD = 1024;
static constexpr int FD = 4096;
using V = std::vector<double>;
using Mat = std::vector<V>;

static int pmod(int a, int n) { return ((a % n) + n) % n; }

class LinearTest : public ::testing::Test {
protected:
    static Inference inf;

    static void SetUpTestSuite() {
        if (inf.fhe) return;
        inf = make_gpt2(16, HD, FD, 4, 16, false, false);
    }

    int S() const { return inf.slots; }
    const CC& cc() const { return inf.fhe->cc; }

    V dec(const Ctx& ct) {
        return decrypt(cc(), ct, inf.fhe->sk());
    }

    // ── Paper encoding: square W ∈ ℝ^{d×d} ──
    //   p_{j,k}[i] = W[(i/t + i%t + j·t) % d,  (i/t − k·t·r_i) % d]
    //   where t = S/d, r_i = inRot
    void install_square(const std::string& wname, const Mat& W, int d) {
        int s = S();
        int t = s / d;
        int inRot  = (int)std::sqrt((double)((long long)d * d / (2 * s)));
        int outRot = (int)((long long)d * d / ((long long)s * inRot));
        int nPt = inRot * outRot;

        inf.w[wname].clear();
        for (int idx = 0; idx < nPt; ++idx) {
            int j = idx % inRot;
            int k = idx / inRot;
            V sl(s);
            for (int i = 0; i < s; ++i) {
                int row = pmod(i / t + i % t + j * t, d);
                int col = pmod(i / t - k * t * inRot, d);
                sl[i] = W[row][col];
            }
            inf.w[wname].push_back(cc()->MakeCKKSPackedPlaintext(sl));
        }
    }

    // ── Paper encoding: up  W ∈ ℝ^{d × αd} ──
    //   p_{j,k}[i] = W[(i/t + i%t + j·t) % d,
    //                   (⌊i/t⌋ + (i/t')%α · d − k·t·r_i) % (αd)]
    //   where t = S/d, t' = S/(αd)
    void install_up(const std::string& wname, const Mat& W,
                    int d, int alpha_d) {
        int s = S();
        int alpha = alpha_d / d;
        int t  = s / d;
        int tp = s / alpha_d;
        int inRot  = (int)std::sqrt((double)((long long)d * alpha_d / (2 * s)));
        int outRot = (int)((long long)d * alpha_d / ((long long)s * inRot));
        int nPt = inRot * outRot;

        inf.w[wname].clear();
        for (int idx = 0; idx < nPt; ++idx) {
            int j = idx % inRot;
            int k = idx / inRot;
            V sl(s);
            for (int i = 0; i < s; ++i) {
                int row = pmod(i / t + i % t + j * t, d);
                int col = pmod(i / t + (i / tp % alpha) * d
                               - k * t * inRot, alpha_d);
                sl[i] = W[row][col];
            }
            inf.w[wname].push_back(cc()->MakeCKKSPackedPlaintext(sl));
        }
    }

    // ── Paper encoding: down  W ∈ ℝ^{αd × d} ──
    //   p_{j,k}[i] = W[(⌊i/t⌋ + (i/t')%α·d + i%t + j·t) % (αd),
    //                   (i/t − k·t·r_i) % d]
    //   where t = S/d, t' = S/(αd)
    void install_down(const std::string& wname, const Mat& W,
                      int alpha_d, int d) {
        int s = S();
        int alpha = alpha_d / d;
        int t  = s / d;
        int tp = s / alpha_d;
        int inRot  = (int)std::sqrt((double)((long long)d * alpha_d / (2 * s)));
        int outRot = (int)((long long)d * alpha_d / ((long long)s * inRot));
        int nPt = inRot * outRot;

        inf.w[wname].clear();
        for (int idx = 0; idx < nPt; ++idx) {
            int j = idx % inRot;
            int k = idx / inRot;
            V sl(s);
            for (int i = 0; i < s; ++i) {
                int row = pmod(i / t + (i / tp % alpha) * d
                               + i % t + j * t, alpha_d);
                int col = pmod(i / t - k * t * inRot, d);
                sl[i] = W[row][col];
            }
            inf.w[wname].push_back(cc()->MakeCKKSPackedPlaintext(sl));
        }
    }

    // ── Input: square / up  (d-dimensional, stride t = S/d) ──
    Ctx enc_simple(const V& x, int d) {
        int s = S(), t = s / d;
        V sl(s, 0.0);
        for (int m = 0; m < d; ++m) sl[m * t] = x[m];
        auto pt = cc()->MakeCKKSPackedPlaintext(sl);
        return cc()->Encrypt(inf.fhe->pk(), pt);
    }

    // ── Input: down  (αd-dimensional, interleaved stride t' = S/(αd)) ──
    //   c^x[m·t'] = x[⌊m/α⌋ + (m%α)·d]    for m = 0 … αd−1
    Ctx enc_down(const V& x, int alpha_d, int d) {
        int s = S();
        int alpha = alpha_d / d;
        int tp = s / alpha_d;
        V sl(s, 0.0);
        for (int m = 0; m < alpha_d; ++m) {
            int idx = m / alpha + (m % alpha) * d;
            sl[m * tp] = x[idx];
        }
        auto pt = cc()->MakeCKKSPackedPlaintext(sl);
        return cc()->Encrypt(inf.fhe->pk(), pt);
    }

    // ── Output extraction ──

    V extract_simple(const V& raw, int d) {
        int t = S() / d;
        V y(d);
        for (int m = 0; m < d; ++m) y[m] = raw[m * t];
        return y;
    }

    V extract_up(const V& raw, int d, int alpha_d) {
        int alpha = alpha_d / d;
        int tp = S() / alpha_d;
        V y(alpha_d);
        for (int m = 0; m < alpha_d; ++m) {
            int idx = m / alpha + (m % alpha) * d;
            y[idx] = raw[m * tp];
        }
        return y;
    }

    // ── Reference: y = x · W  (left multiply) ──
    static V vmm(const V& x, const Mat& W) {
        int d_in = (int)x.size(), d_out = (int)W[0].size();
        V y(d_out, 0.0);
        for (int i = 0; i < d_in; ++i)
            for (int j = 0; j < d_out; ++j)
                y[j] += x[i] * W[i][j];
        return y;
    }

    void run(const char* wname, int d_in, int d_out) {
        std::mt19937 rng(d_in * 1000 + d_out);
        std::uniform_real_distribution<double> wdist(-0.01, 0.01);
        std::uniform_real_distribution<double> xdist(-0.1, 0.1);

        Mat W(d_in, V(d_out));
        for (auto& row : W)
            for (auto& v : row) v = wdist(rng);

        if (d_in == d_out)
            install_square(wname, W, d_in);
        else if (d_in < d_out)
            install_up(wname, W, d_in, d_out);
        else
            install_down(wname, W, d_in, d_out);

        for (int trial = 0; trial < 3; ++trial) {
            V x(d_in);
            for (auto& v : x) v = xdist(rng);
            V expected = vmm(x, W);

            Ctx ct_in = (d_in > d_out)
                ? enc_down(x, d_in, d_out)
                : enc_simple(x, d_in);

            Ctx ct_out = linear_interleaved(inf, ct_in, wname, d_in, d_out);
            V raw = dec(ct_out);

            V actual = (d_in < d_out)
                ? extract_up(raw, d_in, d_out)
                : extract_simple(raw, d_out);

            double max_err = 0.0;
            for (int i = 0; i < d_out; ++i)
                max_err = std::max(max_err,
                                   std::abs(actual[i] - expected[i]));
            EXPECT_LT(max_err, 0.5)
                << wname << " " << d_in << "x" << d_out
                << " trial " << trial << " max_err=" << max_err;
        }
    }
};

Inference LinearTest::inf = {};

TEST_F(LinearTest, Square) { run("q",    HD, HD); }
TEST_F(LinearTest, Up)     { run("up",   HD, FD); }
TEST_F(LinearTest, Down)   { run("down", FD, HD); }
