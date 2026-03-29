// main.cu – Benchmark harness for measuring per-operation FHE latencies.
//
// GPT-2 only.  All linear layers use CacheMIR interleaved packing.
//
// Usage:
//   ./cuda_cachemir -test QKV     -level 5  -logN 12
//   ./cuda_cachemir -test GELU    -level 8  -logN 12
//   ./cuda_cachemir -test Softmax -level 16 -btpLevel 12 -logN 12
//   ./cuda_cachemir -test Norm    -level 5  -btpLevel 12 -logN 12
//   ./cuda_cachemir -test Decoder -logN 12  -hidDim 256 -ffDim 1024

#include "gpt2.h"
#include "gpt2_optimized_config.h"
#include "nonlinear.h"
#include "ckks_primitives.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <stdexcept>
#include <random>
#include <chrono>

struct Flags {
    int logN = 12;
    std::string test = "Decoder";
    int level = 5;
    int btpLevel = 15;
    int hidDim = 256;
    int ffDim = 1024;
    int realHidDim = 0;  // 0 → same as hidDim (no padding)
    int realFfDim  = 0;  // 0 → same as ffDim  (no padding)
    int seqLen = 512;
    int numHeads = 32;
    bool parallel = true;
    // For Classify test: comma-separated candidate token IDs
    // Default: two common GPT-2 BPE tokens for "positive" / "negative"
    std::vector<int> candidates = {3967, 4633};
    // Path to real weights directory (from prepare_gpt2_weights.py)
    std::string weights;
    int numLayers = 12;
    // For Generate test: prompt token IDs and number of tokens to generate
    std::vector<int> prompt_tokens;
    int numGen = 1;
    int maxLevel = 0;  // 0 = use total_depth (for BenchAll)
    std::string resumeFile;  // BenchAll: skip ops already in this file
};

static Flags parse(int argc, char** argv) {
    Flags f;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto nxt = [&]() {
            if (++i >= argc) throw std::runtime_error("Missing value for " + s);
            return std::string(argv[i]);
        };
        if      (s == "-logN"     || s == "--logN")     f.logN     = std::stoi(nxt());
        else if (s == "-test"     || s == "--test")     f.test     = nxt();
        else if (s == "-level"    || s == "--level")    f.level    = std::stoi(nxt());
        else if (s == "-btpLevel" || s == "--btpLevel") f.btpLevel = std::stoi(nxt());
        else if (s == "-hidDim"   || s == "--hidDim")   f.hidDim   = std::stoi(nxt());
        else if (s == "-ffDim"    || s == "--ffDim"  ||
                 s == "-expDim"   || s == "--expDim")   f.ffDim    = std::stoi(nxt());
        else if (s == "-realHidDim" || s == "--realHidDim") f.realHidDim = std::stoi(nxt());
        else if (s == "-realFfDim"  || s == "--realFfDim")  f.realFfDim  = std::stoi(nxt());
        else if (s == "-seqLen"   || s == "--seqLen")   f.seqLen   = std::stoi(nxt());
        else if (s == "-numHeads" || s == "--numHeads") f.numHeads = std::stoi(nxt());
        else if (s == "-parallel"   || s == "--parallel")   f.parallel = (nxt() != "false");
        else if (s == "-model"      || s == "--model")      nxt(); // ignored, always GPT-2
        else if (s == "-weights"    || s == "--weights")    f.weights   = nxt();
        else if (s == "-numLayers"  || s == "--numLayers")  f.numLayers = std::stoi(nxt());
        else if (s == "-candidates" || s == "--candidates") {
            std::string csv = nxt();
            f.candidates.clear();
            std::stringstream ss(csv);
            std::string tok;
            while (std::getline(ss, tok, ','))
                if (!tok.empty()) f.candidates.push_back(std::stoi(tok));
        }
        else if (s == "-prompt" || s == "--prompt") {
            std::string csv = nxt();
            f.prompt_tokens.clear();
            std::stringstream ss(csv);
            std::string tok;
            while (std::getline(ss, tok, ','))
                if (!tok.empty()) f.prompt_tokens.push_back(std::stoi(tok));
        }
        else if (s == "-numGen" || s == "--numGen") f.numGen = std::stoi(nxt());
        else if (s == "-maxLevel" || s == "--maxLevel") f.maxLevel = std::stoi(nxt());
        else if (s == "-resumeFile" || s == "--resumeFile") f.resumeFile = nxt();
    }
    return f;
}

// ── Helpers ─────────────────────────────────────────────────────────────

static std::mt19937_64 rng(42);

static Ctx make_ct_at_level(Inference& inf, int target_level) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> msg(inf.slots);
    for (auto& v : msg) v = dist(rng);
    std::cout << "  [make_ct] encoding plaintext (slots=" << inf.slots << ")..." << std::flush;
    Ptx pt = inf.cc()->MakeCKKSPackedPlaintext(msg);
    std::cout << " encrypting..." << std::flush;
    Ctx ct = inf.cc()->Encrypt(inf.fhe->pk(), pt);
    std::cout << " done (level=" << level_of(ct) << ")" << std::endl;
    if (target_level > 0) {
        std::cout << "  [make_ct] reducing level " << level_of(ct)
                  << " → " << target_level << "..." << std::flush;
        reduce_to_level(inf.cc(), ct, (uint32_t)target_level, inf.slots);
        std::cout << " done (level=" << level_of(ct) << ")" << std::endl;
    }
    return ct;
}

static Ptx make_rand_pt(Inference& inf) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> msg(inf.slots);
    for (auto& v : msg) v = dist(rng);
    return inf.cc()->MakeCKKSPackedPlaintext(msg);
}

static Ctx make_rand_ct(Inference& inf) {
    Ptx pt = make_rand_pt(inf);
    return inf.cc()->Encrypt(inf.fhe->pk(), pt);
}

struct BenchTimer {
    std::chrono::steady_clock::time_point t0;
    void start() { t0 = std::chrono::steady_clock::now(); }
    double stop() const {
        using namespace std::chrono;
        return duration<double>(steady_clock::now() - t0).count();
    }
};

// Print in the exact format run_bench.sh expects
static void print_consumed(double secs, int in_level, int out_level) {
    std::printf("Consumed %f seconds with input level %d and output level %d\n",
                secs, in_level, out_level);
}

static void print_consumed(double secs, int level) {
    std::printf("Consumed %f seconds with level %d\n", secs, level);
}

// ── Per-op benchmarks ──────────────────────────────────────────────────

static void bench_qkv(Inference& inf, int level) {
    const int hD = inf.size.hidDim;
    gpt2_prepare_weights(inf, {"q"});
    Ctx x = make_ct_at_level(inf, level);
    int in_lvl = (int)level_of(x);

    BenchTimer t; t.start();
    Ctx q = linear_interleaved(inf, x, "q", hD, hD);
    double elapsed = t.stop();

    print_consumed(elapsed, in_lvl, (int)level_of(q));
}

static void bench_cache(Inference& inf, int level) {
    gpt2_prepare_cache(inf, {"k", "v", "mask"});
    Ctx k = make_ct_at_level(inf, level);
    Ctx v = make_ct_at_level(inf, level);
    int in_lvl = (int)level_of(k);

    BenchTimer t; t.start();
    cache_kv(inf, k, v);
    double elapsed = t.stop();

    print_consumed(elapsed, in_lvl, in_lvl);
}

static void bench_qk_t(Inference& inf, int level) {
    gpt2_prepare_cache(inf, {"k", "v", "mask"});
    Ctx q = make_ct_at_level(inf, level);
    int in_lvl = (int)level_of(q);

    BenchTimer t; t.start();
    Ctx s = qk_transpose(inf, q);
    double elapsed = t.stop();

    print_consumed(elapsed, in_lvl, (int)level_of(s));
}

static void bench_attn_v(Inference& inf, int level) {
    gpt2_prepare_cache(inf, {"k", "v", "mask"});
    Ctx s = make_ct_at_level(inf, level);
    int in_lvl = (int)level_of(s);

    BenchTimer t; t.start();
    Ctx o = attn_v(inf, s);
    double elapsed = t.stop();

    print_consumed(elapsed, in_lvl, (int)level_of(o));
}

static void bench_out(Inference& inf, int level) {
    const int hD = inf.size.hidDim;
    gpt2_prepare_weights(inf, {"out"});
    Ctx x = make_ct_at_level(inf, level);
    int in_lvl = (int)level_of(x);

    BenchTimer t; t.start();
    Ctx y = linear_interleaved(inf, x, "out", hD, hD);
    double elapsed = t.stop();

    print_consumed(elapsed, in_lvl, (int)level_of(y));
}

static void bench_up(Inference& inf, int level) {
    const int hD = inf.size.hidDim, fD = inf.size.ffDim;
    gpt2_prepare_weights(inf, {"up"});
    Ctx x = make_ct_at_level(inf, level);
    int in_lvl = (int)level_of(x);

    BenchTimer t; t.start();
    Ctx y = linear_interleaved(inf, x, "up", hD, fD);
    double elapsed = t.stop();

    print_consumed(elapsed, in_lvl, (int)level_of(y));
}

static void bench_down(Inference& inf, int level) {
    const int hD = inf.size.hidDim, fD = inf.size.ffDim;
    gpt2_prepare_weights(inf, {"down"});
    Ctx x = make_ct_at_level(inf, level);
    int in_lvl = (int)level_of(x);

    BenchTimer t; t.start();
    Ctx y = linear_interleaved(inf, x, "down", fD, hD);
    double elapsed = t.stop();

    print_consumed(elapsed, in_lvl, (int)level_of(y));
}

static void bench_gelu(Inference& inf, int level) {
    Ctx x = make_ct_at_level(inf, level);
    int in_lvl = (int)level_of(x);

    BenchTimer t; t.start();
    Ctx y = gelu(inf, x, GELU_ENCLLM_GPT2);
    double elapsed = t.stop();

    print_consumed(elapsed, in_lvl, (int)level_of(y));
}

static void bench_ctmult(Inference& inf, int level) {
    const int N = 10;  // average over N multiplications
    std::vector<Ctx> cts;
    for (int i = 0; i < N + 1; ++i)
        cts.push_back(make_ct_at_level(inf, level));

    BenchTimer t; t.start();
    for (int i = 0; i < N; ++i) {
        Ctx r = inf.cc()->EvalMult(cts[i], cts[i + 1]);
        inf.cc()->RescaleInPlace(r);
    }
    double elapsed = t.stop();

    print_consumed(elapsed / N, level);
}

static void bench_bootstrap(Inference& inf, int level) {
    Ctx x = make_ct_at_level(inf, level);
    int in_lvl = (int)level_of(x);

    BenchTimer t; t.start();
    inf.cc()->EvalBootstrap(x);
    double elapsed = t.stop();

    print_consumed(elapsed, in_lvl, (int)level_of(x));
}

static void bench_softmax(Inference& inf, int level, int btpLevel) {
    Ctx x = make_ct_at_level(inf, level);
    int in_lvl = (int)level_of(x);

    BenchTimer t; t.start();
    Ctx y = softmax_cachemir(inf, x, btpLevel, 0);
    double elapsed = t.stop();

    print_consumed(elapsed, in_lvl, (int)level_of(y));
}

static void bench_norm(Inference& inf, int level, int btpLevel) {
    Ctx x = make_ct_at_level(inf, level);
    int in_lvl = (int)level_of(x);

    BenchTimer t; t.start();
    Ctx y = norm(inf, x, btpLevel);
    double elapsed = t.stop();

    print_consumed(elapsed, in_lvl, (int)level_of(y));
}

// ── GPT-2 decoder/model benchmarks ─────────────────────────────────────

static void bench_decoder_gpt2(Inference& inf) {
    gpt2_prepare_weights(inf, {"q", "k", "v", "out", "up", "down"});
    gpt2_prepare_cache(inf, {"k", "v", "mask"});
    Ctx x = make_rand_ct(inf);
    int in_lvl = (int)level_of(x);

    const auto& opt_cfg = gpt2_optimized_config();
    const auto& lcfg = opt_cfg.layers.empty()
        ? GPT2LayerConfig{}
        : opt_cfg.layers[0];

    std::cout << "Weights + cache prepared. Starting decoder..." << std::endl;
    BenchTimer t; t.start();
    Ctx y = gpt2_decoder(inf, x, lcfg);
    double elapsed = t.stop();

    print_consumed(elapsed, in_lvl, (int)level_of(y));
}

static void bench_model_gpt2(Inference& inf) {
    gpt2_prepare_weights(inf, {"q", "k", "v", "out", "up", "down"});
    gpt2_prepare_cache(inf, {"k", "v", "mask"});
    Ctx x = make_rand_ct(inf);

    std::cout << "Evaluating full GPT-2 model (optimized config)...\n";
    BenchTimer t; t.start();
    Ctx y = gpt2_model(inf, x, gpt2_optimized_config());
    double elapsed = t.stop();

    std::printf("Consumed %f seconds for the whole model.\n", elapsed);
}

// ── Zero-shot classification ───────────────────────────────────────────

static void bench_classify_gpt2(Inference& inf,
                                 const std::vector<int>& candidates) {
    std::cout << "\n=== GPT-2 Zero-shot Classification Benchmark ===\n";

    gpt2_prepare_weights(inf, {"q", "k", "v", "out", "up", "down"});
    gpt2_prepare_cache(inf, {"k", "v", "mask"});
    Ctx x = make_rand_ct(inf);
    int in_lvl = (int)level_of(x);

    BenchTimer t; t.start();
    std::vector<Ctx> logits = gpt2_classify(
        inf, x, candidates, -1, gpt2_optimized_config());
    double elapsed = t.stop();

    std::printf("Consumed %f seconds. Output: %zu encrypted logit(s).\n",
                elapsed, logits.size());
}

// ── BenchAll resume support ────────────────────────────────────────────

static std::set<std::string> load_done_sections(const std::string& path) {
    std::set<std::string> done;
    if (path.empty()) return done;
    std::ifstream f(path);
    if (!f.is_open()) return done;
    std::string prev_header;
    std::string line;
    while (std::getline(f, line)) {
        if (line.substr(0, 4) == "--- " && line.size() > 7 &&
            line.substr(line.size() - 4) == " ---") {
            prev_header = line;
        } else if (line.substr(0, 8) == "Consumed" && !prev_header.empty()) {
            done.insert(prev_header);
            prev_header.clear();
        }
    }
    std::cout << "[resume] " << done.size() << " sections already done in "
              << path << std::endl;
    return done;
}

// ── BenchAll: single-context sweep of all ops at all levels ─────────────

static void bench_all_gpt2(Inference& inf, int max_level,
                           const std::string& resumeFile = "") {
    auto done = load_done_sections(resumeFile);

    // Level 1: FIDESlib heap corruption — always start from 2.
    constexpr int MIN_LEVEL = 2;

    // Disable OMP: FIDESlib GPU kernels not thread-safe.
    const bool was_parallel = inf.parallel;
    inf.parallel = false;

    const char* basic_ops[] = {"QKV", "QK_T", "AttnV", "Up", "Down", "Cache", "CtMult"};
    const int   basic_min[] = {  2,      3,      3,      2,     2,      2,       2    };
    using BenchFn = void(*)(Inference&, int);
    BenchFn basic_fns[] = {
        bench_qkv, bench_qk_t, bench_attn_v, bench_up, bench_down,
        bench_cache, bench_ctmult
    };

    auto header = [](const char* op, const char* tag, int val) {
        std::ostringstream ss;
        ss << "--- " << op << " @ " << tag << " " << val << " ---";
        return ss.str();
    };
    int skipped = 0;

    for (int i = 0; i < 7; ++i) {
        const int op_min = std::max(MIN_LEVEL, basic_min[i]);
        for (int level = op_min; level <= max_level; ++level) {
            std::string hdr = header(basic_ops[i], "level", level);
            if (done.count(hdr)) { ++skipped; continue; }
            std::cout << hdr << std::endl;
            try {
                basic_fns[i](inf, level);
            } catch (const std::exception& e) {
                std::cout << "FAIL: " << e.what() << "\n";
            } catch (...) {
                std::cout << "FAIL: unknown exception\n";
            }
        }
    }

    // GELU
    for (int level = MIN_LEVEL; level <= max_level; ++level) {
        std::string hdr = header("GELU", "level", level);
        if (done.count(hdr)) { ++skipped; continue; }
        std::cout << hdr << std::endl;
        try { bench_gelu(inf, level); }
        catch (const std::exception& e) { std::cout << "FAIL: " << e.what() << std::endl; }
        catch (...) { std::cout << "FAIL: unknown exception" << std::endl; }
    }

    // Bootstrap
    for (int level = MIN_LEVEL; level <= max_level; ++level) {
        std::string hdr = header("Bootstrap", "level", level);
        if (done.count(hdr)) { ++skipped; continue; }
        std::cout << hdr << std::endl;
        try { bench_bootstrap(inf, level); }
        catch (const std::exception& e) { std::cout << "FAIL: " << e.what() << std::endl; }
        catch (...) { std::cout << "FAIL: unknown exception" << std::endl; }
    }

    // Softmax (sweeps btpLevel, input always at max_level)
    for (int btp = MIN_LEVEL; btp <= max_level; ++btp) {
        std::string hdr = header("Softmax", "btpLevel", btp);
        if (done.count(hdr)) { ++skipped; continue; }
        std::cout << hdr << std::endl;
        try { bench_softmax(inf, max_level, btp); }
        catch (const std::exception& e) { std::cout << "FAIL: " << e.what() << std::endl; }
        catch (...) { std::cout << "FAIL: unknown exception" << std::endl; }
    }

    // Norm
    for (int level = MIN_LEVEL; level <= max_level; ++level) {
        std::string hdr = header("SqrtNt", "level", level);
        if (done.count(hdr)) { ++skipped; continue; }
        std::cout << hdr << std::endl;
        try { bench_norm(inf, level, max_level); }
        catch (const std::exception& e) { std::cout << "FAIL: " << e.what() << std::endl; }
        catch (...) { std::cout << "FAIL: unknown exception" << std::endl; }
    }
    constexpr int NORM_MIN_BTP = 7;
    for (int btp = NORM_MIN_BTP; btp <= max_level; ++btp) {
        std::string hdr = header("SqrtGold", "btpLevel", btp);
        if (done.count(hdr)) { ++skipped; continue; }
        std::cout << hdr << std::endl;
        try { bench_norm(inf, max_level, btp); }
        catch (const std::exception& e) { std::cout << "FAIL: " << e.what() << std::endl; }
        catch (...) { std::cout << "FAIL: unknown exception" << std::endl; }
    }

    if (skipped > 0)
        std::cout << "[resume] skipped " << skipped << " already-done sections" << std::endl;

    inf.parallel = was_parallel;
    std::cout << "=== BenchAll complete ===\n";
}

// ── Real-weight inference ────────────────────────────────────────────────

static Ctx make_input_from_prompt(Inference& inf, const std::vector<int>& prompt_tokens) {
    const int S = inf.slots;

    bool has_ptx_wte = inf.w.count("wte") && !inf.w.at("wte").empty();
    bool has_ptx_wpe = inf.w.count("wpe") && !inf.w.at("wpe").empty();
    bool has_raw_wte = inf.raw_w.count("wte") && !inf.raw_w.at("wte").empty();
    bool has_raw_wpe = inf.raw_w.count("wpe") && !inf.raw_w.at("wpe").empty();

    if (!prompt_tokens.empty() && (has_ptx_wte || has_raw_wte) && (has_ptx_wpe || has_raw_wpe)) {
        int last_pos = (int)prompt_tokens.size() - 1;
        int last_tok = prompt_tokens[last_pos];
        std::vector<double> h(S, 0.0);

        if (has_raw_wte && has_raw_wpe) {
            const auto& wte_vals = inf.raw_w["wte"][last_tok];
            const auto& wpe_vals = inf.raw_w["wpe"][last_pos];
            for (int j = 0; j < S && j < (int)wte_vals.size(); ++j)
                h[j] = wte_vals[j] + wpe_vals[j];
        } else {
            auto wte_vals = inf.w["wte"][last_tok]->GetRealPackedValue();
            auto wpe_vals = inf.w["wpe"][last_pos]->GetRealPackedValue();
            for (int j = 0; j < S && j < (int)wte_vals.size(); ++j)
                h[j] = wte_vals[j] + wpe_vals[j];
        }

        std::cout << "[CLIENT] Embedding token " << last_tok
                  << " at position " << last_pos << std::endl;
        auto pt = inf.cc()->MakeCKKSPackedPlaintext(h);
        return inf.cc()->Encrypt(inf.fhe->pk(), pt);
    }
    std::cout << "[CLIENT] No prompt/embeddings — using random input\n";
    return make_rand_ct(inf);
}

static void real_classify_gpt2(Inference& inf, const std::string& weight_dir,
                                int num_layers, const std::vector<int>& candidates,
                                const std::vector<int>& prompt_tokens) {
    std::cout << "\n=== GPT-2 Real-Weight Classification ===\n";
    gpt2_load_weights(inf, weight_dir, num_layers);
    gpt2_prepare_cache(inf, {"k", "v", "mask"});

    Ctx x = make_input_from_prompt(inf, prompt_tokens);
    std::cout << "Input encrypted. Running " << num_layers << "-layer model + classify...\n";

    BenchTimer t; t.start();
    std::vector<Ctx> logits = gpt2_classify(
        inf, x, candidates, -1, gpt2_optimized_config());
    double elapsed = t.stop();
    std::printf("Classification done in %f seconds.\n", elapsed);

    // [CLIENT] Decrypt logits and determine predicted class
    std::cout << "\n[CLIENT] Decrypting logits:\n";
    int best_idx = 0;
    double best_logit = -1e30;
    for (int i = 0; i < (int)logits.size(); ++i) {
        auto dec    = decrypt(inf.cc(), logits[i], inf.fhe->sk());
        double logit = dec[0];
        std::printf("  candidate %d: logit=%.6f\n", candidates[i], logit);
        if (logit > best_logit) { best_logit = logit; best_idx = i; }
    }
    std::printf("[CLIENT] Predicted class token: %d\n", candidates[best_idx]);
}

static void real_generate_gpt2(Inference& inf, const std::string& weight_dir,
                                int num_layers, const std::vector<int>& prompt_tokens,
                                int num_gen) {
    std::cout << "\n=== GPT-2 Real-Weight Text Generation ===\n";
    gpt2_load_weights(inf, weight_dir, num_layers);
    gpt2_prepare_cache(inf, {"k", "v", "mask"});

    auto gen = gpt2_generate(inf, prompt_tokens, num_gen, gpt2_optimized_config());

    std::cout << "\n[CLIENT] Generated token IDs: [";
    for (int i = 0; i < (int)gen.size(); ++i)
        std::cout << (i ? "," : "") << gen[i];
    std::cout << "]" << std::endl;
}

static void real_model_gpt2(Inference& inf, const std::string& weight_dir,
                             int num_layers) {
    std::cout << "\n=== GPT-2 Real-Weight Full Model ===\n";
    gpt2_load_weights(inf, weight_dir, num_layers);
    gpt2_prepare_cache(inf, {"k", "v", "mask"});

    Ctx x = make_rand_ct(inf);
    std::cout << "Input encrypted. Running " << num_layers << "-layer model...\n";

    BenchTimer t; t.start();
    Ctx y = gpt2_model(inf, x, gpt2_optimized_config());
    double elapsed = t.stop();

    std::printf("Model done in %f seconds. Output level=%d\n",
                elapsed, (int)level_of(y));
}

static void real_diag_model_gpt2(Inference& inf, const std::string& weight_dir,
                                  int num_layers, const std::vector<int>& prompt_tokens) {
    std::cout << "\n=== GPT-2 Diagnostic Model (layer-by-layer decrypt) ===\n";
    gpt2_load_weights(inf, weight_dir, num_layers);
    gpt2_prepare_cache(inf, {"k", "v", "mask"});

    Ctx x = make_input_from_prompt(inf, prompt_tokens);
    gpt2_diag_model(inf, x, num_layers);
}

// ── Main ────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    Flags f = parse(argc, argv);

    Inference inf = make_gpt2(f.logN, f.hidDim, f.ffDim, f.seqLen, f.numHeads, f.parallel);

    if (f.realHidDim > 0) inf.size.realHidDim = f.realHidDim;
    if (f.realFfDim  > 0) inf.size.realFfDim  = f.realFfDim;
    std::cout << "Initialization finished! slots=" << inf.slots
              << "  hidDim=" << inf.size.hidDim
              << " (real=" << inf.size.getRealHidDim() << ")"
              << "  ffDim=" << inf.size.ffDim
              << " (real=" << inf.size.getRealFfDim() << ")" << std::endl;

    const std::string& T = f.test;

    // SmokeTest: encrypt→mult→bootstrap at various levels
    if (T == "SmokeTest") {
        std::cout << "=== SmokeTest (logN=" << inf.logN << " slots=" << inf.slots
                  << " total_depth=" << inf.total_depth << ") ===" << std::endl;
        std::vector<double> msg(inf.slots, 1.5);

        // 1. Basic encrypt/mult/decrypt
        std::cout << "\n--- Phase 1: Basic operations ---" << std::endl;
        Ptx pt = inf.cc()->MakeCKKSPackedPlaintext(msg);
        Ctx ct = inf.cc()->Encrypt(inf.fhe->pk(), pt);
        std::cout << "  Encrypted (level=" << level_of(ct) << ")" << std::endl;

        std::vector<double> ones(inf.slots, 1.0);
        Ptx pt_one = inf.cc()->MakeCKKSPackedPlaintext(ones, 1, level_of(ct));
        inf.cc()->EvalMultInPlace(ct, pt_one);
        inf.cc()->RescaleInPlace(ct);
        auto vals = decrypt(inf.cc(), ct, inf.fhe->sk());
        std::cout << "  After mult+rescale: level=" << level_of(ct)
                  << " val[0]=" << vals[0] << " err=" << std::abs(vals[0]-1.5) << std::endl;

        // 2. reduce_to_level sweep
        std::cout << "\n--- Phase 2: reduce_to_level sweep ---" << std::endl;
        for (int tgt : {1, 3, 5, 10, 13}) {
            Ctx fresh = inf.cc()->Encrypt(inf.fhe->pk(), pt);
            std::cout << "  Reducing level 0 → " << tgt << "..." << std::flush;
            reduce_to_level(inf.cc(), fresh, (uint32_t)tgt, inf.slots);
            auto v = decrypt(inf.cc(), fresh, inf.fhe->sk());
            std::cout << " level=" << level_of(fresh) << " err=" << std::abs(v[0]-1.5) << std::endl;
        }

        // 3. Bootstrap from different input levels
        std::cout << "\n--- Phase 3: Bootstrap precision vs input level ---" << std::endl;
        for (int input_level : {1, 5, 10, 13, 20, 25}) {
            if ((uint32_t)input_level >= inf.total_depth) continue;
            Ctx c = inf.cc()->Encrypt(inf.fhe->pk(), pt);
            if (input_level > 0)
                reduce_to_level(inf.cc(), c, (uint32_t)input_level, inf.slots);
            std::cout << "  Bootstrap from level " << level_of(c) << "..." << std::flush;
            Ctx b = inf.cc()->EvalBootstrap(c);
            std::cout << " output_level=" << level_of(b) << std::flush;
            try {
                auto vb = decrypt(inf.cc(), b, inf.fhe->sk());
                double berr = std::abs(vb[0] - 1.5);
                std::cout << " val=" << vb[0] << " err=" << berr
                          << " bits=" << (berr > 0 ? -std::log2(berr/1.5) : 99) << std::endl;
            } catch (const std::exception& e) {
                std::cout << " DECODE FAIL: " << e.what() << std::endl;
            }
        }

        std::cout << "\n=== SmokeTest complete ===" << std::endl;
        return 0;
    }

    // BenchAll: single context, sweep all ops at all levels
    if (T == "BenchAll") {
        int ml = f.maxLevel > 0 ? f.maxLevel : (int)inf.total_depth;
        bench_all_gpt2(inf, ml, f.resumeFile);
        return 0;
    }

    // Per-op benchmarks
    if      (T == "QKV")       bench_qkv(inf, f.level);
    else if (T == "Cache")     bench_cache(inf, f.level);
    else if (T == "QK_T")      bench_qk_t(inf, f.level);
    else if (T == "AttnV")     bench_attn_v(inf, f.level);
    else if (T == "Out")       bench_out(inf, f.level);
    else if (T == "Up")        bench_up(inf, f.level);
    else if (T == "Down")      bench_down(inf, f.level);
    else if (T == "GELU")      bench_gelu(inf, f.level);
    else if (T == "CtMult")    bench_ctmult(inf, f.level);
    else if (T == "Bootstrap") bench_bootstrap(inf, f.level);
    else if (T == "Softmax")   bench_softmax(inf, f.level, f.btpLevel);
    else if (T == "Norm")      bench_norm(inf, f.level, f.btpLevel);
    // Full pipeline benchmarks
    else if (T == "Decoder") {
        if (!f.weights.empty()) {
            gpt2_load_weights(inf, f.weights, f.numLayers);
            gpt2_prepare_cache(inf, {"k", "v", "mask"});
            Ctx x = make_rand_ct(inf);
            int in_lvl = (int)level_of(x);
            BenchTimer t; t.start();
            Ctx y = gpt2_decoder(inf, x, GPT2LayerConfig{}, 0);
            double elapsed = t.stop();
            print_consumed(elapsed, in_lvl, (int)level_of(y));
        } else {
            bench_decoder_gpt2(inf);
        }
    }
    else if (T == "Model") {
        if (!f.weights.empty())
            real_model_gpt2(inf, f.weights, f.numLayers);
        else
            bench_model_gpt2(inf);
    }
    else if (T == "DiagModel") {
        if (f.weights.empty()) {
            std::cerr << "DiagModel requires -weights <dir>\n";
            return 1;
        }
        real_diag_model_gpt2(inf, f.weights, f.numLayers, f.prompt_tokens);
    }
    else if (T == "Classify") {
        if (!f.weights.empty())
            real_classify_gpt2(inf, f.weights, f.numLayers, f.candidates, f.prompt_tokens);
        else
            bench_classify_gpt2(inf, f.candidates);
    }
    else if (T == "Generate") {
        if (!f.weights.empty())
            real_generate_gpt2(inf, f.weights, f.numLayers, f.prompt_tokens, f.numGen);
        else {
            gpt2_prepare_weights(inf, {"q","k","v","out","up","down"});
            gpt2_prepare_cache(inf, {"k","v","mask"});
            gpt2_generate(inf, {}, f.numGen, gpt2_optimized_config());
        }
    }
    else if (T == "RealModel") {
        if (f.weights.empty()) {
            std::cerr << "RealModel requires -weights <dir>\n";
            return 1;
        }
        real_model_gpt2(inf, f.weights, f.numLayers);
    }
    else {
        std::cerr << "Unknown test: " << T << "\n"
                  << "Available: QKV Cache QK_T AttnV Out Up Down\n"
                  << "           GELU CtMult Bootstrap Softmax Norm\n"
                  << "           Decoder Model DiagModel Classify Generate RealModel\n"
                  << "           BenchAll SmokeTest\n"
                  << "Options:\n"
                  << "  -weights <dir>       load real weights from prepare_gpt2_weights.py\n"
                  << "  -numLayers N         number of decoder layers (default 12)\n"
                  << "  -prompt tok1,tok2... prompt token IDs for Generate/Classify\n"
                  << "  -numGen N            tokens to generate (default 1)\n"
                  << "  -candidates tok,...  candidate token IDs for Classify (default: 3967,4633)\n";
        return 1;
    }

    return 0;
}
