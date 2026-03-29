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

    // ── Correct encoding for d-1 preprocessing ──
    //
    // After d-1 preprocessing with input dim d_pre and K_pre = S/d_pre:
    //   c'x[K_pre*i + s] = x[(i + s * stride) % d_pre]
    //   where stride = d_pre / K_pre
    //
    // BSGS diagonal t = g*inRot + b.  At slot q:
    //   row = sigma_in((q + b*intRot) % S)
    //   col = output_index(m)  where m = ((q - g*inRot*intRot) % S) / K_post
    //
    // sigma_in(p) = (p/K_pre + (p%K_pre) * stride) % d_pre

    void install_weights(const std::string& wname, const Mat& W,
                         int d_in, int d_out) {
        int s = S();
        int expand = (d_in > d_out) ? -1 : (d_in < d_out) ? 1 : 0;

        int d_pre  = (expand >= 0) ? d_in : d_in;   // always d_in
        int K_pre  = s / d_pre;
        int stride = d_pre / K_pre;

        int intRot = s / std::max(d_in, d_out);
        int K_post;
        if (expand <= 0)
            K_post = s / d_out;
        else
            K_post = s / d_out;  // always S/d_out

        int inRot, outRot;
        if (expand == 0) {
            inRot  = (int)std::sqrt((double)((long long)d_in * d_in / (2 * s)));
            outRot = (int)((long long)d_in * d_in / ((long long)s * inRot));
        } else {
            inRot  = (int)std::sqrt((double)((long long)d_in * d_out / (2 * s)));
            outRot = (int)((long long)d_in * d_out / ((long long)s * inRot));
        }
        int nPt = inRot * outRot;
        int alpha = (expand == 0) ? 1 : std::max(d_in, d_out) / std::min(d_in, d_out);

        inf.w[wname].clear();
        for (int t = 0; t < nPt; ++t) {
            int b = t % inRot;
            int g = t / inRot;
            V sl(s, 0.0);
            for (int q = 0; q < s; ++q) {
                int q_x = (q + b * intRot) % s;
                int row_raw = pmod(q_x / K_pre + (q_x % K_pre) * stride, d_pre);

                int row;
                if (expand == -1)
                    row = row_raw / alpha + (row_raw % alpha) * d_out;
                else
                    row = row_raw;

                int q_shifted = (q + s - g * inRot * intRot) % s;
                int m = q_shifted / K_post;

                int col;
                if (expand == 1)
                    col = m / alpha + (m % alpha) * d_in;
                else
                    col = m % d_out;

                sl[q] = W[row][col];
            }
            inf.w[wname].push_back(cc()->MakeCKKSPackedPlaintext(sl));
        }
    }

    // ── Input: square / up  (d_in-dimensional, stride K = S/d_in) ──
    Ctx enc_simple(const V& x, int d) {
        int s = S(), t = s / d;
        V sl(s, 0.0);
        for (int m = 0; m < d; ++m) sl[m * t] = x[m];
        auto pt = cc()->MakeCKKSPackedPlaintext(sl);
        return cc()->Encrypt(inf.fhe->pk(), pt);
    }

    // ── Input: down  (d_in-dimensional, interleaved stride K' = S/d_in) ──
    //   c^x[m·K'] = x[m/α + (m%α)·d_out]    for m = 0 … d_in−1
    Ctx enc_down(const V& x, int d_in, int d_out) {
        int s = S();
        int alpha = d_in / d_out;
        int kp = s / d_in;
        V sl(s, 0.0);
        for (int m = 0; m < d_in; ++m) {
            int idx = m / alpha + (m % alpha) * d_out;
            sl[m * kp] = x[idx];
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

    V extract_up(const V& raw, int d_in, int d_out) {
        int alpha = d_out / d_in;
        int kp = S() / d_out;
        V y(d_out);
        for (int m = 0; m < d_out; ++m) {
            int idx = m / alpha + (m % alpha) * d_in;
            y[idx] = raw[m * kp];
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
        std::normal_distribution<double> wdist(0.0, 1.0 / std::sqrt((double)d_in));
        std::normal_distribution<double> xdist(0.0, 1.0);

        Mat W(d_in, V(d_out));
        for (auto& row : W)
            for (auto& v : row) v = wdist(rng);

        install_weights(wname, W, d_in, d_out);

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
            EXPECT_LT(max_err, 0.01)
                << wname << " " << d_in << "x" << d_out
                << " trial " << trial << " max_err=" << max_err;
        }
    }
};

Inference LinearTest::inf = {};

TEST_F(LinearTest, Square) { run("q",    HD, HD); }
TEST_F(LinearTest, Up)     { run("up",   HD, FD); }
TEST_F(LinearTest, Down)   { run("down", FD, HD); }
