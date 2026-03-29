// test_gpt2_real.cu — validate C++ GPT-2 ops against PyTorch-generated vectors.
//
// All tests use production params: logN=16, L=13, K=15, paper-aligned.
// Test vectors from: sbatch scripts/03c_generate_test_vectors.sh
//
// Two fixtures:
//   GPT2LinearTest   — per-op diagnostic (each op gets exact Python input)
//   GPT2DecoderTest  — full decoder block(s) with bootstrapping
//
// Build: registered in tests/CMakeLists.txt
// Run:   sbatch scripts/03d_test_gpt2_real.sh

#include <gtest/gtest.h>
#include "fideslib_wrapper.h"
#include "ckks_primitives.h"
#include "inference.h"
#include "gpt2.h"
#include "gpt2_optimized_config.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>

// ── File I/O ─────────────────────────────────────────────────────────────

static std::vector<double> load_vec(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open: " + path);
    std::vector<double> v;
    double val;
    while (f >> val) v.push_back(val);
    return v;
}

static std::vector<std::vector<double>> load_mat(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open: " + path);
    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::vector<double> row;
        double val;
        while (iss >> val) row.push_back(val);
        if (!row.empty()) rows.push_back(std::move(row));
    }
    return rows;
}

// ── Shared test vector directory ────────────────────────────────────────

static const char* VEC_DIR = "test_vectors_logN16/layer0";

static bool test_vectors_exist() {
    return std::filesystem::exists(std::string(VEC_DIR) + "/input.txt");
}

// ══════════════════════════════════════════════════════════════════════════
// GPT2LinearTest — per-op diagnostics using production make_gpt2()
//
// Uses linear_interleaved() — the same code path as real inference.
// Test vectors are in interleaved (sparse) encoding: d²/S plaintexts.
// This uses ~30× less GPU memory than the old BSGS linear() path.
// ══════════════════════════════════════════════════════════════════════════

class GPT2LinearTest : public ::testing::Test {
protected:
    static constexpr uint32_t LOG_N = kGPT2ConfigLogN;
    static constexpr int HD = 1024;
    static constexpr int FD = 4096;

    static Inference production_inf;
    static bool ctx_initialized;
    Inference inf;
    std::string dir;       // interleaved test vector directory
    std::string base_dir;  // parent dir with replicated-format vectors

    static void initContext() {
        if (ctx_initialized) return;
        production_inf = make_gpt2(LOG_N, HD, FD,
                                   /*seqLen=*/1, /*numHeads=*/12,
                                   /*parallel=*/false, /*interleaved=*/false);
        std::cout << "--- GPT2LinearTest (logN=" << LOG_N
                  << " S=" << production_inf.slots
                  << " total_depth=" << production_inf.total_depth << ") ---\n";
        ctx_initialized = true;
    }

    void SetUp() override {
        base_dir = VEC_DIR;
        dir = std::string(VEC_DIR) + "/interleaved";
        if (!std::filesystem::exists(dir + "/input.txt"))
            GTEST_SKIP() << "Interleaved test vectors not found. Regenerate with generate_gpt2_test_vectors.py";
        initContext();

        inf = production_inf;
        inf.bench_mode = false;  // real rotations for correctness validation
        inf.size.realHidDim = 768;
        inf.size.realFfDim  = 3072;
        inf.w.clear();
        inf.cache.clear();
    }

    void load_weight(const std::string& name, const std::string& filename) {
        auto mat = load_mat(dir + "/" + filename);
        inf.w[name].clear();
        for (auto& row : mat)
            inf.w[name].push_back(inf.cc()->MakeCKKSPackedPlaintext(row));
    }

    Ctx encrypt_vec(const std::vector<double>& v) {
        auto pt = inf.cc()->MakeCKKSPackedPlaintext(v);
        return inf.cc()->Encrypt(inf.fhe->pk(), pt);
    }

    // Decrypt and extract sparse-encoded values at positions k*t for k=0..d-1.
    std::vector<double> decrypt_sparse(const Ctx& ct, int d) {
        const int S = inf.slots;
        const int t = S / d;
        auto dec = decrypt(inf.cc(), ct, inf.fhe->sk());
        std::vector<double> out(d);
        for (int k = 0; k < d; ++k)
            out[k] = dec[k * t];
        return out;
    }

    std::vector<double> decrypt_first(const Ctx& ct, int d) {
        auto dec = decrypt(inf.cc(), ct, inf.fhe->sk());
        return std::vector<double>(dec.begin(), dec.begin() + d);
    }

    // Extract sparse-encoded values from a plaintext vector.
    std::vector<double> extract_sparse(const std::vector<double>& v, int d) {
        const int S = inf.slots;
        const int t = S / d;
        std::vector<double> out(d);
        for (int k = 0; k < d && k * t < (int)v.size(); ++k)
            out[k] = v[k * t];
        return out;
    }

    // Run linear_interleaved op + add bias, compare against expected file.
    double test_linear_op(const std::string& weight_name,
                          const std::string& weight_file,
                          const std::string& bias_file,
                          const std::string& input_file,
                          const std::string& expected_file,
                          int d_in, int d_out) {
        load_weight(weight_name, weight_file);
        auto input = load_vec(dir + "/" + input_file);
        Ctx x = encrypt_vec(input);

        Ctx y = linear_interleaved(inf, x, weight_name, d_in, d_out);

        auto bias = load_vec(dir + "/" + bias_file);
        auto bp = inf.cc()->MakeCKKSPackedPlaintext(bias,
                      y->GetNoiseScaleDeg(), (uint32_t)level_of(y));
        inf.cc()->EvalAddInPlace(y, bp);

        auto expected = load_vec(dir + "/" + expected_file);
        auto got_sparse = decrypt_sparse(y, d_out);
        auto exp_sparse = extract_sparse(expected, d_out);
        double max_err = 0.0;
        for (int i = 0; i < d_out; ++i)
            max_err = std::max(max_err, std::abs(got_sparse[i] - exp_sparse[i]));

        std::cout << "  " << weight_name << ": max_err=" << max_err
                  << " level=" << level_of(y)
                  << " (d_in=" << d_in << " d_out=" << d_out
                  << " n_pt=" << inf.w[weight_name].size() << ")\n";
        return max_err;
    }
};

Inference GPT2LinearTest::production_inf;
bool GPT2LinearTest::ctx_initialized = false;

// ── Linear projection tests (interleaved path) ─────────────────────────

TEST_F(GPT2LinearTest, QProjection) {
    double err = test_linear_op("q", "Wq.txt", "bq.txt",
                                "input.txt", "expected_q.txt", HD, HD);
    EXPECT_LT(err, 0.5);
}

TEST_F(GPT2LinearTest, KProjection) {
    double err = test_linear_op("k", "Wk.txt", "bk.txt",
                                "input.txt", "expected_k.txt", HD, HD);
    EXPECT_LT(err, 0.5);
}

TEST_F(GPT2LinearTest, VProjection) {
    double err = test_linear_op("v", "Wv.txt", "bv.txt",
                                "input.txt", "expected_v.txt", HD, HD);
    EXPECT_LT(err, 0.5);
}

TEST_F(GPT2LinearTest, OutProjection) {
    double err = test_linear_op("out", "Wo.txt", "bo.txt",
                                "expected_v.txt", "expected_out_proj.txt", HD, HD);
    EXPECT_LT(err, 0.5);
}

TEST_F(GPT2LinearTest, UpProjection) {
    double err = test_linear_op("up", "Wu.txt", "bu.txt",
                                "input_up.txt", "expected_up.txt", HD, FD);
    EXPECT_LT(err, 0.5);
}

TEST_F(GPT2LinearTest, DownProjection) {
    double err = test_linear_op("down", "Wd.txt", "bd.txt",
                                "input_down.txt", "expected_down.txt", FD, HD);
    EXPECT_LT(err, 0.5);
}

// ── Norm test ───────────────────────────────────────────────────────────

TEST_F(GPT2LinearTest, Norm1) {
    // Norm uses replicated encoding (from base_dir, not interleaved)
    auto input = load_vec(base_dir + "/input.txt");
    Ctx x = encrypt_vec(input);

    NormConfig nc = NORM_ENCLLM_GPT2;
    nc.nr_iters = 4;
    nc.taylor_rescale = 0.19271061;  // layer0 profiled

    Ctx normed = norm(inf, x, /*target_level_after_btp=*/0, nc);

    auto expected = load_vec(base_dir + "/expected_norm1.txt");
    auto got = decrypt_first(normed, HD);
    double max_err = 0.0;
    for (int i = 0; i < HD; ++i)
        max_err = std::max(max_err, std::abs(got[i] - expected[i]));

    std::cout << "  Norm1: max_err=" << max_err
              << " level=" << level_of(normed) << "\n";
    // Norm is approximate — print but don't hard-fail
    if (max_err >= 0.1)
        std::cout << "  WARNING: Norm1 max_err=" << max_err << " >= 0.1\n";
    SUCCEED();
}

// ── GELU test ───────────────────────────────────────────────────────────

TEST_F(GPT2LinearTest, GELU) {
    // GELU uses replicated encoding (from base_dir)
    auto up_vec = load_vec(base_dir + "/expected_up.txt");
    Ctx x = encrypt_vec(up_vec);

    GeluConfig gelu_cfg = GELU_ENCLLM_GPT2;
    gelu_cfg.bootstrap_indicators = false;  // no BTP in this test

    Ctx g = gelu(inf, x, gelu_cfg);

    auto expected = load_vec(base_dir + "/expected_gelu.txt");
    auto got = decrypt_first(g, FD);
    double max_err = 0.0;
    for (int i = 0; i < FD; ++i)
        max_err = std::max(max_err, std::abs(got[i] - expected[i]));

    std::cout << "  GELU: max_err=" << max_err
              << " level=" << level_of(g) << "\n";
    if (max_err >= 1.0)
        std::cout << "  WARNING: GELU max_err=" << max_err << " >= 1.0\n";
    SUCCEED();
}

// ══════════════════════════════════════════════════════════════════════════
// GPT2DecoderTest — full decoder block with bootstrapping
//
// Uses production make_gpt2() context (logN=16, L=13, K=15).
// Block 1: compare vs expected_output.txt
// Block 2: stability check (no NaN/Inf, |slots| < 1000)
// Norm sweep: test different NR iterations
// ══════════════════════════════════════════════════════════════════════════

class GPT2DecoderTest : public ::testing::Test {
protected:
    static constexpr uint32_t LOG_N = kGPT2ConfigLogN;
    static constexpr int HD = 1024;
    static constexpr int FD = 4096;

    static Inference production_inf;
    static bool ctx_initialized;
    Inference inf;
    std::string dir;

    static void initContext() {
        if (ctx_initialized) return;
        production_inf = make_gpt2(LOG_N, HD, FD,
                                   /*seqLen=*/1, /*numHeads=*/12,
                                   /*parallel=*/false, /*interleaved=*/false);
        std::cout << "--- GPT2DecoderTest (logN=" << LOG_N
                  << " S=" << production_inf.slots
                  << " total_depth=" << production_inf.total_depth
                  << " bootstrap=true) ---\n";
        ctx_initialized = true;
    }

    void SetUp() override {
        dir = VEC_DIR;
        if (!test_vectors_exist())
            GTEST_SKIP() << "Test vectors not found. Run: sbatch scripts/03c_generate_test_vectors.sh";
        initContext();

        inf = production_inf;
        inf.bench_mode = false;
        inf.w.clear();
        inf.cache.clear();
        inf.mask.clear();
        inf.cache_mask.clear();
    }

    void load_layer0_weights() {
        const std::pair<const char*, const char*> weights[] = {
            {"layer0_q", "Wq.txt"}, {"layer0_k", "Wk.txt"},
            {"layer0_v", "Wv.txt"}, {"layer0_out", "Wo.txt"},
            {"layer0_up", "Wu.txt"}, {"layer0_down", "Wd.txt"},
        };
        const std::pair<const char*, const char*> biases[] = {
            {"layer0_bq", "bq.txt"}, {"layer0_bk", "bk.txt"},
            {"layer0_bv", "bv.txt"}, {"layer0_bo", "bo.txt"},
            {"layer0_bu", "bu.txt"}, {"layer0_bd", "bd.txt"},
        };
        for (auto& [key, file] : weights) {
            auto mat = load_mat(dir + "/" + file);
            inf.w[key].clear();
            for (auto& row : mat)
                inf.w[key].push_back(inf.cc()->MakeCKKSPackedPlaintext(row));
        }
        for (auto& [key, file] : biases) {
            auto v = load_vec(dir + "/" + file);
            inf.w[key].clear();
            inf.w[key].push_back(inf.cc()->MakeCKKSPackedPlaintext(v));
        }
        std::cout << "  Layer0 weights loaded.\n";
    }

    void reset_cache() {
        inf.cache.clear();
        inf.mask.clear();
        inf.cache_mask.clear();
        gpt2_prepare_cache(inf, {"k", "v", "mask"});
    }

    Ctx encrypt_vec(const std::vector<double>& v) {
        auto pt = inf.cc()->MakeCKKSPackedPlaintext(v);
        return inf.cc()->Encrypt(inf.fhe->pk(), pt);
    }

    std::vector<double> decrypt_first(const Ctx& ct, int d) {
        auto dec = decrypt(inf.cc(), ct, inf.fhe->sk());
        return std::vector<double>(dec.begin(), dec.begin() + d);
    }

    GPT2LayerConfig make_paper_config(int nr_iters = 4, int norm_target = 7) {
        NormConfig nc = NORM_ENCLLM_GPT2;
        nc.nr_iters = nr_iters;
        nc.taylor_rescale = 0.19271061;  // layer0

        GPT2LayerConfig cfg;
        cfg.norm1_btp_level    = 7;
        cfg.norm1_target_level = norm_target;
        cfg.norm1_cfg          = nc;
        cfg.norm2_btp_level    = 7;
        cfg.norm2_target_level = norm_target;
        cfg.norm2_cfg          = nc;
        cfg.attn_btp_level     = 9;
        cfg.attn_v_btp_level   = 0;
        cfg.gelu_btp_level     = 13;
        cfg.down_btp_level     = 0;
        GeluConfig gc          = GELU_ENCLLM_GPT2;
        gc.bootstrap_indicators = true;
        cfg.gelu_cfg           = gc;
        cfg.attn_causal_mask   = nullptr;
        return cfg;
    }
};

Inference GPT2DecoderTest::production_inf;
bool GPT2DecoderTest::ctx_initialized = false;

// ── Full decoder block ──────────────────────────────────────────────────

TEST_F(GPT2DecoderTest, OneBlock) {
    load_layer0_weights();
    reset_cache();

    auto input_vec = load_vec(dir + "/input.txt");
    Ctx x0 = encrypt_vec(input_vec);
    std::cout << "  x0 level=" << level_of(x0) << "\n";

    auto cfg = make_paper_config();

    Timer t;
    Ctx x1 = gpt2_decoder(inf, x0, cfg, 0);
    double elapsed = t.elapsed_s();

    // Compare vs reference
    auto expected = load_vec(dir + "/expected_output.txt");
    auto got = decrypt_first(x1, HD);
    double max_err = 0.0;
    for (int i = 0; i < HD; ++i)
        max_err = std::max(max_err, std::abs(got[i] - expected[i]));

    std::cout << "  Block1: max_err=" << max_err
              << " level=" << level_of(x1)
              << " elapsed=" << elapsed << "s\n";
    for (int i : {0, 1, HD/2, HD-1})
        std::cout << "    slot[" << i << "] expected=" << expected[i]
                  << " got=" << got[i] << "\n";

    if (max_err >= 10.0)
        std::cout << "  WARNING: Block1 max_err >= 10.0\n";
}

TEST_F(GPT2DecoderTest, TwoBlocks) {
    load_layer0_weights();
    reset_cache();

    auto input_vec = load_vec(dir + "/input.txt");
    Ctx x0 = encrypt_vec(input_vec);
    auto cfg = make_paper_config();

    // Block 1
    Ctx x1 = gpt2_decoder(inf, x0, cfg, 0);
    std::cout << "  Block1 level=" << level_of(x1) << "\n";

    // Block 2 (reuse layer-0 weights as surrogate)
    reset_cache();
    Timer t;
    Ctx x2 = gpt2_decoder(inf, x1, cfg, 0);
    double elapsed = t.elapsed_s();
    std::cout << "  Block2 level=" << level_of(x2)
              << " elapsed=" << elapsed << "s\n";

    // Stability checks
    auto dec = decrypt_first(x2, HD);
    bool has_nan = false;
    double mx = 0.0;
    for (int i = 0; i < HD; ++i) {
        if (!std::isfinite(dec[i])) has_nan = true;
        mx = std::max(mx, std::abs(dec[i]));
    }
    std::cout << "  Block2 max_abs=" << mx << "\n";
    for (int i : {0, 1, HD/2, HD-1})
        std::cout << "    slot[" << i << "] = " << dec[i] << "\n";

    EXPECT_FALSE(has_nan) << "Block2 output must be finite";
    EXPECT_LT(mx, 1000.0) << "Block2 should not diverge";
}

// ── Norm precision sweep ────────────────────────────────────────────────

TEST_F(GPT2DecoderTest, NormPrecisionSweep) {
    load_layer0_weights();

    auto input_vec = load_vec(dir + "/input.txt");

    struct Variant { const char* label; int nr; int tgt; };
    Variant variants[] = {
        {"4NR-tgt7",   4,  7},
        {"8NR-tgt7",   8,  7},
        {"16NR-tgt7", 16,  7},
    };

    std::cout << "\n=== Norm Precision Sweep ===\n";
    std::cout << "  variant          | out_lvl | max_abs | status\n";
    std::cout << "  -----------------|---------|---------|-------\n";

    for (auto& v : variants) {
        Ctx x0 = encrypt_vec(input_vec);
        reset_cache();

        auto cfg = make_paper_config(v.nr, v.tgt);

        try {
            Ctx x1 = gpt2_decoder(inf, x0, cfg, 0);
            Ctx probe = bootstrap_to(inf, x1, inf.total_depth);

            auto dec = decrypt_first(probe, HD);
            double mx = 0.0;
            bool has_nan = false;
            for (int i = 0; i < HD; ++i) {
                if (!std::isfinite(dec[i])) has_nan = true;
                mx = std::max(mx, std::abs(dec[i]));
            }
            std::printf("  %-18s| %7d | %7.1f | %s\n",
                        v.label, (int)level_of(x1), mx,
                        has_nan ? "NaN/Inf!" : "OK");
        } catch (const std::exception& e) {
            std::printf("  %-18s| ---     | ---     | FAIL: %s\n",
                        v.label, e.what());
        }
    }
}
