#include "gpt2.h"
#include "attention.h"
#include <iostream>
#include <random>
#include <cmath>
#include <iomanip>

static std::vector<double> encode_mha(const std::vector<double>& v,
                                       int N, int d, int H) {
    int t = N / d, tH = t * H;
    std::vector<double> msg(N, 0.0);
    for (int r = 0; r < d; ++r)
        msg[(r / H) * tH + (r % H) * t] = v[r];
    return msg;
}

int main() {
    const int logN = 16, d = 1024, d_exp = 4096, H = 16, nk = 5;
    int S = 1 << (logN - 1);
    int t = S / d, tH = t * H, d_head = d / H;

    auto extra_rots = compute_gpt2_rot_indices(S, d, d_exp, H, 1);
    std::cout << "Creating CKKS context (logN=" << logN << ")...\n";

    Inference inf;
    inf.fhe = make_ckks_context(logN, 13, 41, 0, true, 50, 53, {4,3}, 0, 192, 3, 15, extra_rots);
    inf.slots       = S;
    inf.total_depth = 13 + 15;
    inf.bench_mode  = false;
    inf.size.hidDim   = d;
    inf.size.numHeads = H;
    prepare_mha_masks(inf);
    prepare_vcache(inf);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-0.05, 0.05);

    std::cout << "\n[vcache] Pushing " << nk << " values ... " << std::flush;
    std::vector<std::vector<double>> V_ref(nk, std::vector<double>(d));
    for (int k = 0; k < nk; ++k) {
        for (auto& x : V_ref[k]) x = dist(rng);
        auto msg = encode_mha(V_ref[k], S, d, H);
        Ctx ct = encrypt(inf.cc(),
                         inf.cc()->MakeCKKSPackedPlaintext(msg),
                         inf.fhe->pk());
        cache_v_push(inf, ct);
    }
    std::cout << "done (v_count=" << inf.v_count << ")\n";

    std::vector<std::vector<double>> soft(nk, std::vector<double>(H));
    for (int h = 0; h < H; ++h) {
        double sum = 0.0;
        for (int k = 0; k < nk; ++k) {
            soft[k][h] = 0.1 + std::abs(dist(rng));
            sum += soft[k][h];
        }
        for (int k = 0; k < nk; ++k)
            soft[k][h] /= sum;
    }

    std::vector<double> scores_msg(S, 0.0);
    for (int k = 0; k < nk; ++k)
        for (int h = 0; h < H; ++h)
            scores_msg[(k / t) * tH + h * t + (k % t)] = soft[k][h];

    Ctx scores_ct = encrypt(inf.cc(),
                            inf.cc()->MakeCKKSPackedPlaintext(scores_msg),
                            inf.fhe->pk());

    std::cout << "[vcache] Computing softmax @ V ... " << std::flush;
    Ctx out = softmax_v(inf, scores_ct);
    auto raw = decrypt(inf.cc(), out, inf.fhe->sk());

    double max_err = 0.0;
    for (int ld = 0; ld < d_head; ++ld)
        for (int h = 0; h < H; ++h) {
            double expected = 0.0;
            for (int k = 0; k < nk; ++k)
                expected += soft[k][h] * V_ref[k][ld * H + h];
            int idx = ld * tH + h * t;
            double err = std::abs(raw[idx] - expected);
            if (err > max_err) max_err = err;
        }

    bool ok = max_err < 1e-2;
    std::cout << (ok ? "PASS" : "FAIL") << "  max_err=" << max_err << "\n";
    std::cout << (ok ? "ALL PASSED" : "SOME FAILURES") << "\n";
    return ok ? 0 : 1;
}
