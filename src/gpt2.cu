// gpt2.cu – GPT-2 inference using CacheMIR interleaved packing.
//
// GPT-2 (small) vs LLaMA:
//   - 12 layers (not 32)
//   - No RoPE (learned position embeddings, handled at input)
//   - No gated MLP: Up -> GELU -> Down (no gate, no elem_mult)
//   - GELU activation instead of SiLU
//   - Pre-norm (LayerNorm before attention and MLP)
//   - Biases on all linear layers (absorbed into weight plaintexts)
//
// All linear layers use CacheMIR interleaved packing (linear_interleaved).
// Junk slots from interleaved output are fused/cleaned by the next
// element-wise multiplication (paper §3.2 "Fused Ciphertext Extraction").

#include "gpt2.h"
#include "gpt2_optimized_config.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <set>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <omp.h>

// ── Memory tracking ────────────────────────────────────────────────────
static size_t get_rss_mb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.compare(0, 6, "VmRSS:") == 0) {
            size_t kb = 0;
            sscanf(line.c_str(), "VmRSS: %zu kB", &kb);
            return kb / 1024;
        }
    }
    return 0;
}

// ── Rotation index computation (interleaved only) ──────────────────────

static int find_bsgs_inRot(int d) {
    int r = (int)std::sqrt((double)d);
    while (r > 1 && d % r != 0) --r;
    return r;
}

std::vector<int32_t> compute_gpt2_rot_indices(
    int S, int hidDim, int ffDim, int numHeads, int seqLen)
{
    std::set<int32_t> indices;

    // ── CacheMIR interleaved linear rotations ───────────────────────
    auto add_interleaved_rots = [&](int d_in, int d_out) {
        const int t_in  = S / d_in;
        const int t_out = S / d_out;
        const int n_pt  = d_in * d_out / S;

        int r_i = std::max(1, (int)std::sqrt((double)n_pt));
        while (n_pt % r_i != 0 && r_i > 1) --r_i;

        // Baby step: max for all cases (matches linear.cu)
        int t_baby = std::max(t_in, t_out);
        if (t_baby > 1) {
            indices.insert(t_baby);
            indices.insert(-t_baby);
        }

        int gs = std::max(t_in, t_out) * r_i;
        if (gs != t_baby) {
            indices.insert(gs);
            indices.insert(-gs);
        }

        for (int step = 1; step < t_in; step *= 2) {
            int rv = step * (d_in - 1);
            indices.insert(rv % S);
            indices.insert(-(rv % S));
        }
    };

    add_interleaved_rots(hidDim, hidDim);    // Q/K/V/Out
    add_interleaved_rots(hidDim, ffDim);     // Up
    add_interleaved_rots(ffDim, hidDim);     // Down

    // ── Attention rotations ─────────────────────────────────────────
    int headDim = hidDim / numHeads;
    int intRot  = S / hidDim;

    int inner_step = numHeads * S / hidDim;
    for (int j = 1; j < headDim; j *= 2) {
        int rot_val = inner_step * j;
        indices.insert(rot_val);
        indices.insert(-rot_val);
    }

    int space_qk = S * S / (seqLen * hidDim);
    for (int i = 1; i < hidDim * seqLen / S; ++i)
        indices.insert(S - space_qk * i);

    int space_av = S * numHeads / hidDim;
    int av_inRot = find_bsgs_inRot(headDim);
    int av_outRot = av_inRot * 2;
    indices.insert(space_av);
    indices.insert(-space_av);
    for (int i = 1; i < av_outRot; ++i) {
        indices.insert(i * space_av * av_inRot);
        indices.insert(-(i * space_av * av_inRot));
    }

    for (int idx = 0; idx < intRot; ++idx) {
        indices.insert(idx);
        indices.insert(-idx);
    }
    for (int mid = 0; mid <= seqLen / intRot; ++mid) {
        int rot_idx = mid * S * numHeads / hidDim;
        for (int t = 0; t < intRot; ++t) {
            indices.insert(t + rot_idx);
            indices.insert(-(t + rot_idx));
        }
    }

    indices.erase(0);
    return std::vector<int32_t>(indices.begin(), indices.end());
}

// ── Context setup ───────────────────────────────────────────────────────

Inference make_gpt2(int logN, int hidDim, int ffDim,
                    int seqLen, int numHeads, bool parallel,
                    bool bench) {
    Inference inf;
    inf.size     = {hidDim, ffDim, numHeads, seqLen, 0, 0};
    inf.logN     = logN;
    inf.parallel = parallel;

    std::cout << "Creating CKKS context for GPT-2 (logN=" << logN << ")...\n";
    uint32_t btp_slots = (uint32_t)(1 << (logN - 1));
    int S = (int)btp_slots;

    constexpr int kDepth    = 13;
    constexpr int kBtpExtra = 15;

    int btp_sbits = kGPT2BtpScaleBits;
    int first_mod = kGPT2FirstModBits;

    auto extra_rots = compute_gpt2_rot_indices(S, hidDim, ffDim, numHeads, seqLen);
    std::cout << "  Extra rotation indices: " << extra_rots.size() << "\n";

    inf.fhe         = make_ckks_context(logN, kDepth, /*scale_bits=*/41,
                                        btp_slots, /*bootstrap=*/true,
                                        /*btp_scale_bits=*/btp_sbits,
                                        /*first_mod_bits=*/first_mod,
                                        /*level_budget_in=*/{4, 3},
                                        /*batch_size=*/0,
                                        /*h_weight=*/192,
                                        /*num_large_digits=*/3,
                                        /*btp_depth_overhead=*/kBtpExtra,
                                        /*extra_rot_steps=*/extra_rots);
    inf.slots       = (int)btp_slots;
    inf.total_depth = kDepth + kBtpExtra;
    inf.bench_mode  = bench;
    std::cout << "  slots=" << inf.slots
              << "  bench_mode=" << inf.bench_mode
              << "  GPU context loaded.\n";
    return inf;
}

// ── Weight / cache preparation ──────────────────────────────────────────

static std::mt19937_64 gpt2_rng_(42);

static Ptx gpt2_rand_plaintext(Inference& inf) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> msg(inf.slots);
    for (auto& v : msg) v = dist(gpt2_rng_);
    return inf.cc()->MakeCKKSPackedPlaintext(msg);
}

static Ctx gpt2_rand_ciphertext(Inference& inf) {
    std::uniform_real_distribution<double> dist(-20.0, 0.0);
    std::vector<double> msg(inf.slots);
    for (auto& v : msg) v = dist(gpt2_rng_);
    Ptx pt = inf.cc()->MakeCKKSPackedPlaintext(msg);
    return inf.cc()->Encrypt(inf.fhe->pk(), pt);
}

void gpt2_prepare_weights(Inference& inf, const std::vector<std::string>& names) {
    std::cout << "Preparing GPT-2 weights (interleaved)..." << std::endl;
    const int hD = inf.size.hidDim, fD = inf.size.ffDim, S = inf.slots;
    for (const auto& n : names) {
        int cnt;
        // Interleaved: n_weights = d_in * d_out / S
        if      (n == "q" || n == "k" || n == "v" || n == "out") cnt = hD * hD / S;
        else if (n == "up" || n == "down")                        cnt = hD * fD / S;
        else throw std::runtime_error("Unknown GPT-2 weight: " + n);

        inf.w[n].clear();
        for (int i = 0; i < cnt; ++i)
            inf.w[n].push_back(gpt2_rand_plaintext(inf));
    }
    std::cout << "GPT-2 weights ready." << std::endl;
}

void gpt2_prepare_cache(Inference& inf, const std::vector<std::string>& names) {
    std::cout << "Preparing GPT-2 cache..." << std::endl;
    const int hD = inf.size.hidDim, seqL = inf.size.seqLen;
    const int nH = inf.size.numHeads, S = inf.slots;
    for (const auto& n : names) {
        if (n == "k") {
            inf.cache["k"].clear();
            int k_slots = std::max(1, hD * seqL / S);
            for (int i = 0; i < k_slots; ++i)
                inf.cache["k"].push_back(gpt2_rand_ciphertext(inf));
            inf.mask["k"] = gpt2_rand_plaintext(inf);
        } else if (n == "v") {
            inf.cache["v"].clear();
            for (int i = 0; i < hD / nH; ++i)
                inf.cache["v"].push_back(gpt2_rand_ciphertext(inf));
            inf.mask["v"] = gpt2_rand_plaintext(inf);
        } else if (n == "mask") {
            inf.cache_mask.clear();
            for (int i = 0; i < hD / nH; ++i)
                inf.cache_mask.push_back(gpt2_rand_plaintext(inf));
        } else {
            throw std::runtime_error("Unknown GPT-2 cache: " + n);
        }
    }
    std::cout << "GPT-2 cache ready." << std::endl;
}

// ── Weight loading from files ───────────────────────────────────────────

static std::tuple<int, int, std::vector<double>> read_npy(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("Cannot open npy file: " + path);

    char magic[6];
    f.read(magic, 6);
    if (magic[0] != '\x93' || std::string(magic+1, 5) != "NUMPY")
        throw std::runtime_error("Not a .npy file: " + path);

    uint8_t major, minor;
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);

    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t hl;
        f.read(reinterpret_cast<char*>(&hl), 2);
        header_len = hl;
    } else {
        f.read(reinterpret_cast<char*>(&header_len), 4);
    }

    std::string header(header_len, '\0');
    f.read(&header[0], header_len);

    auto shape_pos = header.find("'shape':");
    if (shape_pos == std::string::npos)
        throw std::runtime_error("No shape in npy header: " + path);
    auto paren_open = header.find('(', shape_pos);
    auto paren_close = header.find(')', paren_open);
    std::string shape_str = header.substr(paren_open + 1, paren_close - paren_open - 1);

    int rows = 0, cols = 1;
    auto comma = shape_str.find(',');
    rows = std::stoi(shape_str.substr(0, comma));
    if (comma != std::string::npos) {
        std::string rest = shape_str.substr(comma + 1);
        while (!rest.empty() && (rest[0] == ' ' || rest[0] == ',')) rest.erase(0, 1);
        if (!rest.empty())
            cols = std::stoi(rest);
    }

    size_t total = (size_t)rows * cols;
    std::vector<double> data(total);
    f.read(reinterpret_cast<char*>(data.data()), total * sizeof(double));
    return {rows, cols, std::move(data)};
}

static std::vector<std::vector<double>> read_weight_file_txt(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open weight file: " + path);
    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::vector<double> vals;
        double v;
        while (iss >> v) vals.push_back(v);
        if (!vals.empty()) rows.push_back(std::move(vals));
    }
    return rows;
}

static std::vector<double> tile_to_slots(const double* core, int d, int S) {
    std::vector<double> full(S);
    for (int start = 0; start < S; start += d)
        std::memcpy(&full[start], core, d * sizeof(double));
    return full;
}

static std::vector<Ptx> load_weight_matrix(
        const CC& cc, const std::string& txt_path, int S = 0) {
    std::string npy_path = txt_path.substr(0, txt_path.size() - 4) + ".npy";
    if (std::filesystem::exists(npy_path)) {
        auto [nrows, ncols, data] = read_npy(npy_path);
        std::vector<Ptx> pts;
        pts.reserve(nrows);
        for (int r = 0; r < nrows; ++r) {
            const double* row_ptr = data.data() + (size_t)r * ncols;
            if (S > 0 && ncols < S) {
                auto full = tile_to_slots(row_ptr, ncols, S);
                pts.push_back(cc->MakeCKKSPackedPlaintext(full));
            } else {
                std::vector<double> row(row_ptr, row_ptr + ncols);
                pts.push_back(cc->MakeCKKSPackedPlaintext(row));
            }
        }
        std::cout << "  loaded " << npy_path << " (" << nrows
                  << " diags, core=" << ncols << " tiled to " << S << ")\n";
        return pts;
    }
    auto rows = read_weight_file_txt(txt_path);
    std::vector<Ptx> pts;
    pts.reserve(rows.size());
    for (auto& row : rows)
        pts.push_back(cc->MakeCKKSPackedPlaintext(row));
    return pts;
}

static std::vector<std::vector<double>> load_raw_weight_matrix(
        const std::string& txt_path, int S = 0) {
    std::string npy_path = txt_path.substr(0, txt_path.size() - 4) + ".npy";
    if (std::filesystem::exists(npy_path)) {
        auto [nrows, ncols, data] = read_npy(npy_path);
        std::vector<std::vector<double>> result;
        result.reserve(nrows);
        for (int r = 0; r < nrows; ++r) {
            const double* rp = data.data() + (size_t)r * ncols;
            if (S > 0 && ncols < S)
                result.push_back(tile_to_slots(rp, ncols, S));
            else
                result.emplace_back(rp, rp + ncols);
        }
        return result;
    }
    return read_weight_file_txt(txt_path);
}

static Ptx load_single_plaintext(const CC& cc, const std::string& txt_path) {
    std::string npy_path = txt_path.substr(0, txt_path.size() - 4) + ".npy";
    if (std::filesystem::exists(npy_path)) {
        auto [nrows, ncols, data] = read_npy(npy_path);
        return cc->MakeCKKSPackedPlaintext(data);
    }
    auto rows = read_weight_file_txt(txt_path);
    if (rows.empty())
        throw std::runtime_error("Empty weight file: " + txt_path);
    return cc->MakeCKKSPackedPlaintext(rows[0]);
}

// ── Per-layer weight streaming ──────────────────────────────────────────

static void load_layer_weights(Inference& inf, int layer_idx) {
    const CC& cc = inf.cc();
    const int S = inf.slots;
    const std::string& dir = inf.weight_dir;
    std::string lp = "layer" + std::to_string(layer_idx) + "_";
    auto path = [&](const std::string& n) { return dir + "/" + n; };

    auto try_load_bias = [&](const std::string& fpath) -> std::vector<Ptx> {
        std::string npy = fpath.substr(0, fpath.size() - 4) + ".npy";
        if (std::filesystem::exists(npy) || std::ifstream(fpath).good())
            return { load_single_plaintext(cc, fpath) };
        return {};
    };

    Timer t;
    inf.w[lp + "q"]    = load_weight_matrix(cc, path(lp + "Wq.txt"), S);
    inf.w[lp + "k"]    = load_weight_matrix(cc, path(lp + "Wk.txt"), S);
    inf.w[lp + "v"]    = load_weight_matrix(cc, path(lp + "Wv.txt"), S);
    inf.w[lp + "out"]  = load_weight_matrix(cc, path(lp + "Wo.txt"), S);
    inf.w[lp + "up"]   = load_weight_matrix(cc, path(lp + "Wu.txt"), S);
    inf.w[lp + "down"] = load_weight_matrix(cc, path(lp + "Wd.txt"), S);

    inf.w[lp + "bq"] = try_load_bias(path(lp + "bq.txt"));
    inf.w[lp + "bk"] = try_load_bias(path(lp + "bk.txt"));
    inf.w[lp + "bv"] = try_load_bias(path(lp + "bv.txt"));
    inf.w[lp + "bo"] = try_load_bias(path(lp + "bo.txt"));
    inf.w[lp + "bu"] = try_load_bias(path(lp + "bu.txt"));
    inf.w[lp + "bd"] = try_load_bias(path(lp + "bd.txt"));

    std::cout << "  [stream] layer " << layer_idx << " loaded ("
              << t.elapsed_s() << " s, RSS=" << get_rss_mb() << " MB)\n";
}

static void free_layer_weights(Inference& inf, int layer_idx) {
    std::string lp = "layer" + std::to_string(layer_idx) + "_";
    for (const auto& name : {"q","k","v","out","up","down",
                              "bq","bk","bv","bo","bu","bd"}) {
        inf.w.erase(lp + name);
        inf.w.erase(std::string(name));
    }
}

static void activate_layer_weights(Inference& inf, int layer_idx) {
    std::string lp = "layer" + std::to_string(layer_idx) + "_";
    for (const auto& name : {"q","k","v","out","up","down",
                              "bq","bk","bv","bo","bu","bd"}) {
        auto it = inf.w.find(lp + name);
        if (it != inf.w.end())
            inf.w[name] = it->second;
    }
}

void gpt2_load_weights(Inference& inf, const std::string& dir, int num_layers) {
    inf.weight_dir = dir;
    inf.bench_mode = false;

    std::cout << "Loading GPT-2 weights from " << dir << " ...\n";
    const int hD = inf.size.hidDim, fD = inf.size.ffDim, S = inf.slots;
    int ptx_per_layer = 4 * (hD * hD / S) + (hD * fD / S) + (fD * hD / S);
    double mb_per_ptx = (double)(S * 2 * 8 * 21) / (1024.0 * 1024.0);
    std::cout << "  " << ptx_per_layer << " Ptx/layer (~"
              << (int)(ptx_per_layer * mb_per_ptx / 1024) << " GB/layer)\n";
    std::cout << "  Per-layer weights will be streamed on-demand.\n";

    auto path = [&](const std::string& name) { return dir + "/" + name; };

    // Embeddings (raw doubles — encoded on-demand to avoid OOM)
    std::string wte_npy = path("wte.npy"), wpe_npy = path("wpe.npy");
    std::string wte_txt = path("wte.txt"), wpe_txt = path("wpe.txt");
    if (std::filesystem::exists(wte_npy) || std::ifstream(wte_txt).good()) {
        inf.raw_w["wte"] = load_raw_weight_matrix(wte_txt, S);
        std::cout << "  wte: " << inf.raw_w["wte"].size() << " token embeddings\n";
    }
    if (std::filesystem::exists(wpe_npy) || std::ifstream(wpe_txt).good()) {
        inf.raw_w["wpe"] = load_raw_weight_matrix(wpe_txt, S);
        std::cout << "  wpe: " << inf.raw_w["wpe"].size() << " position embeddings\n";
    }

    std::string lm_txt = path("lm_head.txt");
    if (std::filesystem::exists(path("lm_head.npy")) || std::ifstream(lm_txt).good()) {
        inf.raw_w["lm_head"] = load_raw_weight_matrix(lm_txt, S);
        std::cout << "  lm_head: " << inf.raw_w["lm_head"].size() << " rows\n";
    }

    std::cout << "GPT-2 weights loaded (streamed, RSS=" << get_rss_mb() << " MB)\n";
}

// ── GPT-2 Decoder ───────────────────────────────────────────────────────
//
// Pipeline (per layer), matching Go reference structure:
//
//   1. LayerNorm1 (gamma absorbed into W columns)
//   2. Q/K/V = linear_interleaved(x, name, hD, hD) + bias
//   3. Cache K,V
//   4. QK^T → bootstrap → Softmax → AttnV → Out + bias
//   5. Residual: x = x_in + attn_out
//   6. LayerNorm2 (gamma absorbed into W columns)
//   7. Up = linear_interleaved(x, "up", hD, fD) + bias
//   8. bootstrap → GELU
//   9. Down = linear_interleaved(x, "down", fD, hD) + bias
//  10. Residual: x = x + mlp_out

static bool has_bias(const Inference& inf, const std::string& name) {
    auto it = inf.w.find(name);
    return it != inf.w.end() && !it->second.empty();
}

Ctx gpt2_decoder(Inference& inf, const Ctx& x_in, const GPT2LayerConfig& cfg,
                 int layer_idx) {
    const CC& cc = inf.cc();
    const int hD = inf.size.hidDim;
    const int fD = inf.size.ffDim;

    std::cout << "=== GPT-2 Decoder (layer=" << layer_idx
              << ") x_in level=" << level_of(x_in) << " ===\n";
    Timer t;

    if (layer_idx >= 0)
        activate_layer_weights(inf, layer_idx);

    // 1. Pre-attention LayerNorm
    Ctx x = bootstrap_to(inf, x_in, cfg.norm1_btp_level);
    x = norm(inf, x, cfg.norm1_target_level, cfg.norm1_cfg);

    // 2. QKV projections (interleaved)
    Ctx q = linear_interleaved(inf, x, "q", hD, hD);
    Ctx k = linear_interleaved(inf, x, "k", hD, hD);
    Ctx v = linear_interleaved(inf, x, "v", hD, hD);

    if (has_bias(inf, "bq")) {
        cc->EvalAddInPlace(q, inf.w["bq"][0]);
        cc->EvalAddInPlace(k, inf.w["bk"][0]);
        cc->EvalAddInPlace(v, inf.w["bv"][0]);
    }

    // 3. KV cache
    if (cfg.cache_btp_level > 0) {
        k = bootstrap_to(inf, k, cfg.cache_btp_level);
        v = bootstrap_to(inf, v, cfg.cache_btp_level);
    }
    cache_kv(inf, k, v);

    // 4. Attention: QK^T → softmax → AttnV → OutProj
    Ctx s = qk_transpose(inf, q);
    s = bootstrap_to(inf, s, cfg.attn_btp_level);
    s = softmax(inf, s, cfg.softmax_cfg, nullptr, cfg.attn_causal_mask);
    if (cfg.attn_v_btp_level > 0)
        s = bootstrap_to(inf, s, cfg.attn_v_btp_level);

    Ctx attn_out = attn_v(inf, s);
    attn_out = linear_interleaved(inf, attn_out, "out", hD, hD);
    if (has_bias(inf, "bo"))
        cc->EvalAddInPlace(attn_out, inf.w["bo"][0]);

    // 5. Residual (attention)
    Ctx xr = x_in;
    reduce_to_level(cc, xr, level_of(attn_out), inf.slots);
    cc->EvalAddInPlace(xr, attn_out);

    // 6. Pre-MLP LayerNorm
    x = bootstrap_to(inf, xr, cfg.norm2_btp_level);
    x = norm(inf, x, cfg.norm2_target_level, cfg.norm2_cfg);

    // 7. Up projection + bias
    Ctx up_ct = linear_interleaved(inf, x, "up", hD, fD);
    if (has_bias(inf, "bu"))
        cc->EvalAddInPlace(up_ct, inf.w["bu"][0]);

    // 8. GELU
    up_ct = bootstrap_to(inf, up_ct, cfg.gelu_btp_level);
    Ctx y = gelu(inf, up_ct, cfg.gelu_cfg);

    // 9. Down projection + bias
    if (cfg.down_btp_level > 0)
        y = bootstrap_to(inf, y, cfg.down_btp_level);
    y = linear_interleaved(inf, y, "down", fD, hD);
    if (has_bias(inf, "bd"))
        cc->EvalAddInPlace(y, inf.w["bd"][0]);

    // 10. Residual (MLP)
    if (level_of(xr) < level_of(y))
        reduce_to_level(cc, xr, level_of(y), inf.slots);
    else if (level_of(y) < level_of(xr))
        reduce_to_level(cc, y, level_of(xr), inf.slots);
    cc->EvalAddInPlace(xr, y);

    std::cout << "Decoder done in " << t.elapsed_s() << " s\n";
    return xr;
}

// ── GPT-2 Model ────────────────────────────────────────────────────────

Ctx gpt2_model(Inference& inf, const Ctx& x_in, const GPT2ModelConfig& cfg) {
    Timer t;
    Ctx x = x_in;
    bool stream = !inf.weight_dir.empty()
               && inf.w.find("layer0_q") == inf.w.end();

    for (int i = 0; i < cfg.num_layers; ++i) {
        std::cout << "--- GPT-2 Layer " << i << " ---\n";
        if (stream) load_layer_weights(inf, i);

        const GPT2LayerConfig& lcfg = cfg.layers.empty()
            ? GPT2LayerConfig{} : cfg.layers.at((size_t)i);
        x = gpt2_decoder(inf, x, lcfg, stream ? i : -1);

        if (stream) free_layer_weights(inf, i);
    }

    x = bootstrap_to(inf, x, cfg.final_norm_btp_level);
    x = norm(inf, x, cfg.final_norm_target_level, cfg.final_norm_cfg);
    std::cout << "GPT-2 model complete in " << t.elapsed_s() << " s\n";
    return x;
}

// ── GPT-2 Diagnostic Model ─────────────────────────────────────────────

void gpt2_diag_model(Inference& inf, const Ctx& x_in, int num_layers) {
    auto cfg = gpt2_optimized_config();
    cfg.num_layers = num_layers;

    const int hD = inf.size.hidDim;
    Ctx x = x_in;
    bool stream = !inf.weight_dir.empty()
               && inf.w.find("layer0_q") == inf.w.end();

    std::cout << "\n  layer | in_lvl | out_lvl | max_abs_val | decrypt_ok\n";
    std::cout << "  ------|--------|---------|-------------|----------\n";

    Timer total_t;
    for (int i = 0; i < num_layers; ++i) {
        if (stream) load_layer_weights(inf, i);

        const GPT2LayerConfig& lcfg = cfg.layers.empty()
            ? GPT2LayerConfig{} : cfg.layers.at((size_t)i);

        int in_lvl = (int)level_of(x);
        x = gpt2_decoder(inf, x, lcfg, stream ? i : -1);
        int out_lvl = (int)level_of(x);

        if (stream) free_layer_weights(inf, i);

        Ctx probe = bootstrap_to(inf, x, inf.total_depth);
        double max_abs = 0.0;
        bool ok = true;
        try {
            auto dec = decrypt(inf.cc(), probe, inf.fhe->sk());
            for (int j = 0; j < hD && j < (int)dec.size(); ++j)
                max_abs = std::max(max_abs, std::abs(dec[j]));
        } catch (const std::exception& e) {
            ok = false;
            std::printf("  %5d | %6d | %7d | DECRYPT FAIL | %s\n",
                        i, in_lvl, out_lvl, e.what());
            continue;
        }
        std::printf("  %5d | %6d | %7d | %11.4f | %s\n",
                    i, in_lvl, out_lvl, max_abs, ok ? "YES" : "NO");
    }

    std::cout << "\n  --- Final LayerNorm ---\n";
    x = bootstrap_to(inf, x, cfg.final_norm_btp_level);
    x = norm(inf, x, cfg.final_norm_target_level, cfg.final_norm_cfg);

    Ctx h = bootstrap_to(inf, x, inf.total_depth);
    try {
        auto dec = decrypt(inf.cc(), h, inf.fhe->sk());
        double max_abs = 0.0;
        for (int j = 0; j < hD && j < (int)dec.size(); ++j)
            max_abs = std::max(max_abs, std::abs(dec[j]));
        std::printf("  final | %6d | %7d | %11.4f | YES\n",
                    (int)level_of(x), (int)level_of(h), max_abs);
    } catch (const std::exception& e) {
        std::printf("  final | DECRYPT FAIL | %s\n", e.what());
    }
    std::printf("\nDiagnostic done in %.1f s\n", total_t.elapsed_s());
}

// ── Zero-shot Classification ───────────────────────────────────────────

static Ctx eval_sum_all(Inference& inf, Ctx x) {
    const int S = inf.slots;
    for (int step = 1; step < S; step *= 2)
        rotate_add_inplace(inf, x, step);
    return x;
}

std::vector<Ctx> gpt2_classify(
    Inference&              inf,
    const Ctx&              x_in,
    const std::vector<int>& candidate_label_indices,
    int                     classify_pos,
    const GPT2ModelConfig&  cfg)
{
    const CC& cc = inf.cc();
    Timer t;

    Ctx x = gpt2_model(inf, x_in, cfg);
    Ctx h = bootstrap_to(inf, x, inf.total_depth);

    std::vector<Ctx> logits;
    logits.reserve(candidate_label_indices.size());

    for (int tok : candidate_label_indices) {
        Ptx lm_row;
        auto raw_lm_it = inf.raw_w.find("lm_head");
        if (raw_lm_it != inf.raw_w.end() && tok < (int)raw_lm_it->second.size()) {
            lm_row = cc->MakeCKKSPackedPlaintext(raw_lm_it->second[tok]);
        } else {
            const std::string wname = "lm_head_" + std::to_string(tok);
            if (inf.w.find(wname) == inf.w.end()) {
                inf.w[wname].clear();
                inf.w[wname].push_back(gpt2_rand_plaintext(inf));
            }
            lm_row = inf.w.at(wname)[0];
        }
        Ctx logit = eval_sum_all(inf, cc->EvalMult(h, lm_row));
        logits.push_back(logit);
    }

    std::cout << "GPT-2 classify done in " << t.elapsed_s() << " s\n";
    return logits;
}

// ── Text Generation ────────────────────────────────────────────────────

std::vector<int> gpt2_generate(
    Inference&              inf,
    const std::vector<int>& prompt_token_ids,
    int                     max_new_tokens,
    const GPT2ModelConfig&  cfg)
{
    const CC& cc = inf.cc();
    const int S  = inf.slots;
    Timer total_t;

    bool has_wte = (inf.raw_w.find("wte") != inf.raw_w.end());
    bool has_wpe = (inf.raw_w.find("wpe") != inf.raw_w.end());
    bool has_lm  = (inf.raw_w.find("lm_head") != inf.raw_w.end());
    int  vocab_size = has_lm ? (int)inf.raw_w["lm_head"].size() : 0;

    // Encode last prompt position
    std::vector<double> h_plain(S, 0.0);
    if (has_wte && has_wpe) {
        int last_pos = (int)prompt_token_ids.size() - 1;
        int tok = prompt_token_ids[last_pos];
        const auto& wte_vals = inf.raw_w["wte"][tok];
        const auto& wpe_vals = inf.raw_w["wpe"][last_pos];
        for (int j = 0; j < S && j < (int)wte_vals.size(); ++j)
            h_plain[j] = wte_vals[j] + wpe_vals[j];
    } else {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (auto& v : h_plain) v = dist(gpt2_rng_);
    }

    Ctx x = encrypt(cc, cc->MakeCKKSPackedPlaintext(h_plain), inf.fhe->pk());
    std::vector<int> generated;

    for (int step = 0; step < max_new_tokens; ++step) {
        int pos = (int)prompt_token_ids.size() + step;
        std::cout << "--- Generation step " << step << " (pos=" << pos << ") ---\n";

        Ctx h = gpt2_model(inf, x, cfg);
        h = bootstrap_to(inf, h, inf.total_depth);
        auto h_dec = decrypt(cc, h, inf.fhe->sk());

        int best_tok = 0;
        double best_logit = -1e30;
        if (has_lm) {
            for (int tok = 0; tok < vocab_size; ++tok) {
                const auto& lm_vals = inf.raw_w["lm_head"][tok];
                double logit = 0.0;
                int hD = inf.size.hidDim;
                for (int j = 0; j < hD && j < (int)lm_vals.size(); ++j)
                    logit += h_dec[j] * lm_vals[j];
                if (logit > best_logit) { best_logit = logit; best_tok = tok; }
            }
        } else {
            best_tok = std::abs((int)(h_dec[0] * 1000)) % 50257;
        }

        generated.push_back(best_tok);
        std::cout << "  Generated token: " << best_tok << "\n";

        if (has_wte && has_wpe && pos < (int)inf.raw_w["wpe"].size()) {
            const auto& wte_vals = inf.raw_w["wte"][best_tok];
            const auto& wpe_vals = inf.raw_w["wpe"][pos];
            std::vector<double> next_h(S, 0.0);
            for (int j = 0; j < S && j < (int)wte_vals.size(); ++j)
                next_h[j] = wte_vals[j] + wpe_vals[j];
            x = encrypt(cc, cc->MakeCKKSPackedPlaintext(next_h), inf.fhe->pk());
        } else {
            x = encrypt(cc, cc->MakeCKKSPackedPlaintext(h_plain), inf.fhe->pk());
        }
    }

    std::cout << "GPT-2 generation complete in " << total_t.elapsed_s() << " s\n";
    return generated;
}
