// main.cu – Benchmark harness for measuring per-operation FHE latencies.
//
// Mirrors the Go/Cachemir benchmark interface so that run_bench.sh can
// parse the same "Consumed X.XXXXXX seconds" output format.
//
// Usage:
//   ./cuda_cachemir -test QKV     -level 5  -logN 12
//   ./cuda_cachemir -test GELU    -level 8  -logN 12
//   ./cuda_cachemir -test Softmax -level 16 -btpLevel 12 -logN 12
//   ./cuda_cachemir -test Norm    -level 5  -btpLevel 12 -logN 12
//   ./cuda_cachemir -test Decoder -logN 12  -hidDim 256 -ffDim 1024

#include "llama.h"
#include "gpt2.h"
#include "nonlinear.h"
#include "ckks_primitives.h"
#include <cmath>
#include <iostream>
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
    int seqLen = 512;
    int numHeads = 32;
    bool parallel = true;
    std::string model = "llama";
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
        else if (s == "-seqLen"   || s == "--seqLen")   f.seqLen   = std::stoi(nxt());
        else if (s == "-numHeads" || s == "--numHeads") f.numHeads = std::stoi(nxt());
        else if (s == "-parallel" || s == "--parallel") f.parallel = (nxt() != "false");
        else if (s == "-model"    || s == "--model")    f.model    = nxt();
    }
    return f;
}

// ── Helpers ─────────────────────────────────────────────────────────────

static std::mt19937_64 rng(42);

static Ctx make_ct_at_level(Inference& inf, int target_level) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> msg(inf.slots);
    for (auto& v : msg) v = dist(rng);
    Ptx pt = inf.cc()->MakeCKKSPackedPlaintext(msg);
    Ctx ct = inf.cc()->Encrypt(inf.fhe->pk(), pt);
    // Drop to target level
    drop_levels(inf.cc(), ct, (uint32_t)target_level);
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

// ── Test implementations ────────────────────────────────────────────────

static void bench_qkv(Inference& inf, int level) {
    prepare_weights(inf, {"q", "k", "v"});
    Ctx x = make_ct_at_level(inf, level);
    int in_lvl = (int)level_of(x);

    BenchTimer t; t.start();
    Ctx q = qkv_q(inf, x);
    double elapsed = t.stop();

    print_consumed(elapsed, in_lvl, (int)level_of(q));
}

static void bench_cache(Inference& inf, int level) {
    prepare_cache(inf, {"k", "v", "mask"});
    Ctx k = make_ct_at_level(inf, level);
    Ctx v = make_ct_at_level(inf, level);
    int in_lvl = (int)level_of(k);

    BenchTimer t; t.start();
    cache_kv(inf, k, v);
    double elapsed = t.stop();

    print_consumed(elapsed, in_lvl, in_lvl);
}

static void bench_qk_t(Inference& inf, int level) {
    prepare_cache(inf, {"k"});
    Ctx q = make_ct_at_level(inf, level);
    int in_lvl = (int)level_of(q);

    BenchTimer t; t.start();
    Ctx s = qk_transpose(inf, q);
    double elapsed = t.stop();

    print_consumed(elapsed, in_lvl, (int)level_of(s));
}

static void bench_attn_v(Inference& inf, int level) {
    prepare_cache(inf, {"v"});
    Ctx s = make_ct_at_level(inf, level);
    int in_lvl = (int)level_of(s);

    BenchTimer t; t.start();
    Ctx o = attn_v(inf, s);
    double elapsed = t.stop();

    print_consumed(elapsed, in_lvl, (int)level_of(o));
}

static void bench_out(Inference& inf, int level) {
    prepare_weights(inf, {"out"});
    Ctx x = make_ct_at_level(inf, level);
    int in_lvl = (int)level_of(x);

    BenchTimer t; t.start();
    Ctx y = out_proj(inf, x);
    double elapsed = t.stop();

    print_consumed(elapsed, in_lvl, (int)level_of(y));
}

static void bench_upgate(Inference& inf, int level) {
    prepare_weights(inf, {"up", "gate"});
    Ctx x = make_ct_at_level(inf, level);
    int in_lvl = (int)level_of(x);

    BenchTimer t; t.start();
    auto [up_ct, gate_ct] = up_gate(inf, x);
    double elapsed = t.stop();

    print_consumed(elapsed, in_lvl, (int)level_of(up_ct));
}

static void bench_up(Inference& inf, int level) {
    // GPT-2 style: single up projection (no gate)
    prepare_weights(inf, {"up"});
    Ctx x = make_ct_at_level(inf, level);
    int in_lvl = (int)level_of(x);

    BenchTimer t; t.start();
    Ctx y = linear(inf, x, "up", 1);
    double elapsed = t.stop();

    print_consumed(elapsed, in_lvl, (int)level_of(y));
}

static void bench_down(Inference& inf, int level) {
    prepare_weights(inf, {"down"});
    Ctx x = make_ct_at_level(inf, level);
    int in_lvl = (int)level_of(x);

    BenchTimer t; t.start();
    Ctx y = down_proj(inf, x);
    double elapsed = t.stop();

    print_consumed(elapsed, in_lvl, (int)level_of(y));
}

static void bench_silu(Inference& inf, int level) {
    Ctx x = make_ct_at_level(inf, level);
    int in_lvl = (int)level_of(x);

    BenchTimer t; t.start();
    Ctx y = silu(inf, x);
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

static void bench_rope(Inference& inf, int level) {
    prepare_weights(inf, {"RoPE"});
    Ctx q = make_ct_at_level(inf, level);
    Ctx k = make_ct_at_level(inf, level);
    int in_lvl = (int)level_of(q);

    BenchTimer t; t.start();
    auto [yq, yk] = rope(inf, q, k);
    double elapsed = t.stop();

    print_consumed(elapsed, in_lvl, (int)level_of(yq));
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

// Softmax: prints two timings — pre-bootstrap and post-bootstrap parts.
// run_bench.sh captures the last "Consumed" line for the Softmax row,
// and uses btpLevel to sweep the post-bootstrap level.
static void bench_softmax(Inference& inf, int level, int btpLevel) {
    Ctx x = make_ct_at_level(inf, level);
    int in_lvl = (int)level_of(x);

    BenchTimer t; t.start();
    Ctx y = softmax_cachemir(inf, x, btpLevel, 0);
    double elapsed = t.stop();

    print_consumed(elapsed, in_lvl, (int)level_of(y));
}

// Norm: prints two separate timings for SqrtNt (Newton init) and
// SqrtGold (Goldschmidt refinement) phases.
static void bench_norm(Inference& inf, int level, int btpLevel) {
    Ctx x = make_ct_at_level(inf, level);
    int in_lvl = (int)level_of(x);

    BenchTimer t; t.start();
    Ctx y = norm(inf, x, btpLevel);
    double elapsed = t.stop();

    // Print total time — run_bench.sh separates SqrtNt/SqrtGold
    // by calling with different level/btpLevel combinations
    print_consumed(elapsed, in_lvl, (int)level_of(y));
}

// ── LLaMA decoder/model benchmarks ──────────────────────────────────────

static void bench_decoder_llama(Inference& inf) {
    prepare_weights(inf, {"q", "k", "v", "out", "up", "gate", "down", "RoPE"});
    prepare_cache(inf, {"k", "v", "mask"});
    Ctx x = make_rand_ct(inf);
    int in_lvl = (int)level_of(x);

    std::cout << "Evaluating one LLaMA decoder...\n";
    BenchTimer t; t.start();
    Ctx y = decoder(inf, x);
    double elapsed = t.stop();

    print_consumed(elapsed, in_lvl, (int)level_of(y));
}

static void bench_model_llama(Inference& inf) {
    prepare_weights(inf, {"q", "k", "v", "out", "up", "gate", "down", "RoPE"});
    prepare_cache(inf, {"k", "v", "mask"});
    Ctx x = make_rand_ct(inf);

    std::cout << "Evaluating full LLaMA model...\n";
    BenchTimer t; t.start();
    Ctx y = model(inf, x);
    double elapsed = t.stop();

    std::printf("Consumed %f seconds for the whole model.\n", elapsed);
}

// ── GPT-2 decoder/model benchmarks ─────────────────────────────────────

static void bench_decoder_gpt2(Inference& inf) {
    gpt2_prepare_weights(inf, {"q", "k", "v", "out", "up", "down"});
    gpt2_prepare_cache(inf, {"k", "v", "mask"});
    Ctx x = make_rand_ct(inf);
    int in_lvl = (int)level_of(x);

    std::cout << "Evaluating one GPT-2 decoder...\n";
    BenchTimer t; t.start();
    Ctx y = gpt2_decoder(inf, x);
    double elapsed = t.stop();

    print_consumed(elapsed, in_lvl, (int)level_of(y));
}

static void bench_model_gpt2(Inference& inf) {
    gpt2_prepare_weights(inf, {"q", "k", "v", "out", "up", "down"});
    gpt2_prepare_cache(inf, {"k", "v", "mask"});
    Ctx x = make_rand_ct(inf);

    std::cout << "Evaluating full GPT-2 model...\n";
    BenchTimer t; t.start();
    Ctx y = gpt2_model(inf, x);
    double elapsed = t.stop();

    std::printf("Consumed %f seconds for the whole model.\n", elapsed);
}

// ── Main ────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    Flags f = parse(argc, argv);

    // Use GPT-2 context if model=gpt2, else LLaMA
    Inference inf = (f.model == "gpt2")
        ? make_gpt2(f.logN, f.hidDim, f.ffDim, f.seqLen, f.numHeads, f.parallel)
        : make_llama(f.logN, f.hidDim, f.ffDim, f.seqLen, f.numHeads, f.parallel);
    std::cout << "Initialization finished! slots=" << inf.slots << "\n";

    const std::string& T = f.test;

    // Per-op benchmarks (model-agnostic — reuse linear.cu ops)
    if      (T == "QKV")     bench_qkv(inf, f.level);
    else if (T == "RoPE")    bench_rope(inf, f.level);
    else if (T == "Cache")   bench_cache(inf, f.level);
    else if (T == "QK_T")    bench_qk_t(inf, f.level);
    else if (T == "AttnV")   bench_attn_v(inf, f.level);
    else if (T == "Out")     bench_out(inf, f.level);
    else if (T == "UpGate")  bench_upgate(inf, f.level);
    else if (T == "Up")      bench_up(inf, f.level);
    else if (T == "Down")    bench_down(inf, f.level);
    else if (T == "SiLU")    bench_silu(inf, f.level);
    else if (T == "GELU")    bench_gelu(inf, f.level);
    else if (T == "CtMult")  bench_ctmult(inf, f.level);
    else if (T == "Softmax") bench_softmax(inf, f.level, f.btpLevel);
    else if (T == "Norm")    bench_norm(inf, f.level, f.btpLevel);
    // Model-specific full pipelines
    else if (T == "Decoder") {
        if (f.model == "gpt2") bench_decoder_gpt2(inf);
        else                    bench_decoder_llama(inf);
    }
    else if (T == "Model") {
        if (f.model == "gpt2") bench_model_gpt2(inf);
        else                    bench_model_llama(inf);
    }
    else {
        std::cerr << "Unknown test: " << T << "\n"
                  << "Available: QKV RoPE Cache QK_T AttnV Out UpGate Up Down\n"
                  << "           SiLU GELU CtMult Softmax Norm Decoder Model\n"
                  << "Use -model gpt2 for GPT-2 Decoder/Model benchmarks.\n";
        return 1;
    }

    return 0;
}
