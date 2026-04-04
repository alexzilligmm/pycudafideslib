#include "gpt2.h"
#include "attention.h"
#include "nonlinear.h"
#include "ckks_primitives.h"
#include <iostream>
#include <random>
#include <cmath>

static std::vector<double> ref_matmul(const std::vector<double>& x,
                                       const std::vector<std::vector<double>>& W) {
    int d_in  = (int)x.size();
    int d_out = (int)W[0].size();
    std::vector<double> y(d_out, 0.0);
    for (int j = 0; j < d_out; ++j)
        for (int i = 0; i < d_in; ++i)
            y[j] += x[i] * W[i][j];
    return y;
}

static std::vector<double> ref_attention_softmax(
        const std::vector<std::vector<double>>& qkt, int N, int d, int H,
        double given_max, double mask_max) {
    int t  = N / d;
    int tH = t * H;
    int nk = (int)qkt.size();

    std::vector<std::vector<double>> soft(nk, std::vector<double>(H, 0.0));
    for (int h = 0; h < H; ++h) {
        double sum = 0.0;
        for (int k = 0; k < nk; ++k) {
            soft[k][h] = std::exp(qkt[k][h] - given_max);
            sum += soft[k][h];
        }
        for (int k = 0; k < nk; ++k)
            soft[k][h] /= sum;
    }

    std::vector<double> encoded(N, 0.0);
    for (int k = 0; k < nk; ++k)
        for (int h = 0; h < H; ++h)
            encoded[k / t * tH + h * t + k % t] = soft[k][h];
    return encoded;
}

int main() {
    const int logN = 16, d = 1024, d_exp = 4096, H = 16, nk = 5;
    int S = 1 << (logN - 1);
    int t = S / d, tH = t * H;

    auto extra_rots = compute_gpt2_rot_indices(S, d, d_exp, H, 1);
    std::cout << "Creating CKKS context (logN=" << logN << ")...\n";

    Inference inf;
    inf.fhe = make_ckks_context(logN, 13, 41, 0, true, 50, 53, {4,3}, 0, 192, 3, 15, extra_rots);
    inf.slots = S;  inf.total_depth = 28;  inf.bench_mode = false;
    inf.size.hidDim = d;  inf.size.numHeads = H;
    prepare_mha_masks(inf);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-0.01, 0.01);
    const SoftmaxConfig& cfg = SOFTMAX_ATTN_GPT2;

    std::vector<std::vector<double>> W_Q(d, std::vector<double>(d));
    std::vector<std::vector<double>> W_K(d, std::vector<double>(d));
    for (auto& row : W_Q) for (auto& v : row) v = dist(rng);
    for (auto& row : W_K) for (auto& v : row) v = dist(rng);
    inf.w["q"] = encode_weight_matrix(inf, rearrange_qkv_weights(W_Q, H), d, d);
    inf.w["k"] = encode_weight_matrix(inf, rearrange_qkv_weights(W_K, H), d, d);

    auto make_vec = [&]() {
        std::vector<double> v(d);
        for (auto& x : v) x = dist(rng);
        double n = 0; for (auto x : v) n += x*x; n = std::sqrt(n);
        for (auto& x : v) x /= n;
        return v;
    };

    for (int i = 0; i < nk; ++i) {
        Ctx xe = encode_linear_input(inf, make_vec(), d, d);
        cache_k_push(inf, linear(inf, xe, "k", d, d));
    }
    Ctx qd = qkt(inf, linear(inf, encode_linear_input(inf, make_vec(), d, d), "q", d, d));
    int plevel = level_of(qd);
    std::cout << "Pipeline level=" << plevel << "\n";

    double given_max = 50.0;
    double mask_max  = std::max(given_max, std::pow(2.0, cfg.exp_r - 1));

    std::vector<std::vector<double>> gt_qkt(nk, std::vector<double>(H));
    {
        std::uniform_real_distribution<double> top(0.0, 1.0), rest(3.0, 15.0);
        std::uniform_int_distribution<int> dom(0, nk - 1);
        for (int h = 0; h < H; ++h) {
            int d_idx = dom(rng);
            for (int k = 0; k < nk; ++k)
                gt_qkt[k][h] = given_max - (k == d_idx ? top(rng) : rest(rng));
        }
    }

    std::vector<double> scores_msg(S, 0.0);
    for (int k = 0; k < nk; ++k)
        for (int h = 0; h < H; ++h)
            scores_msg[k / t * tH + h * t + k % t] = gt_qkt[k][h];

    Ptx spt = inf.cc()->MakeCKKSPackedPlaintext(scores_msg);
    Ctx sct = inf.cc()->Encrypt(spt, inf.fhe->pk());
    for (int i = 0; i < plevel; ++i) inf.cc()->EvalMultInPlace(sct, 1.0);

    std::cout << "\n[softmax] Testing (peaked, level=" << level_of(sct) << ") ... " << std::flush;
    Ctx soft = attention_softmax(inf, sct, nk, given_max);
    auto raw = decrypt(inf.cc(), soft, inf.fhe->sk());
    auto gt  = ref_attention_softmax(gt_qkt, S, d, H, given_max, mask_max);

    double max_err = 0.0;
    for (int k = 0; k < nk; ++k)
        for (int h = 0; h < H; ++h) {
            int idx = k / t * tH + h * t + k % t;
            max_err = std::max(max_err, std::abs(raw[idx] - gt[idx]));
        }

    bool ok = max_err < 1e-2;
    std::cout << (ok ? "PASS" : "FAIL") << "  max_err=" << max_err << "\n";
    std::cout << (ok ? "ALL PASSED" : "SOME FAILURES") << "\n";
    return ok ? 0 : 1;
}