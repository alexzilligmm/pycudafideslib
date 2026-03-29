# pycudafideslib

GPU-accelerated FHE inference for transformer models (GPT-2, LLaMA) using CKKS
homomorphic encryption via [FIDESlib](third_party/FIDESlib/) + OpenFHE on NVIDIA GPUs.

Implements the [CacheMIR](https://arxiv.org/abs/2602.11470) interleaved packing scheme
with automated bootstrap placement via a DP optimizer.

## Full Pipeline

The pipeline is a 10-step sequence that takes you from source code to a fully
encrypted GPT-2 next-token prediction. Each step produces artifacts consumed
by later steps.

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │   scripts/05_full_inference.sh — runs all 10 steps in one SLURM job    │
 └──┬──────────────────────────────────────────────────────────────────────┘
    │
    │  1. BUILD              cmake + nvcc → build/bin/cuda_cachemir + test binaries
    │
    │  2. BENCH              BenchAll: measure latency of every FHE op (QKV, GELU,
    │                        Softmax, Norm, ...) at each CKKS level 1..13
    │                        → bench_all_output.txt → new_data.csv
    │
    │  3. PLACEMENT          bootstrap.py: DP optimizer reads new_data.csv,
    │                        finds optimal bootstrap insertion points
    │                        → new_data_placement.json
    │
    │  4. CONFIG             gen_config.py: reads placement JSON, generates
    │                        per-layer BTP levels + FHE context params
    │                        → include/gpt2_optimized_config.h
    │
    │  5. REBUILD            cmake + nvcc again (the config header changed)
    │
    │  6. WEIGHTS + VECTORS  prepare_gpt2_weights.py: download HuggingFace GPT-2,
    │                        absorb LayerNorm gamma into W, pad 768→1024 / 3072→4096,
    │                        BSGS-pack diagonals → weights-gpt2/ (standard)
    │                                             → weights-gpt2-interleaved/ (CacheMIR)
    │                        generate_gpt2_test_vectors.py: plaintext reference vectors
    │                        → test_vectors_logN16/
    │
    │  7. PYTHON TESTS       test_gen_config.py: placement→C++ header translation
    │                        test_bsgs_linear.py: numerical BSGS packing verification
    │                          (real_mode, bench_mode, interleaved, rectangular,
    │                           streaming, memory efficiency comparison)
    │
    │  8. C++ TESTS          9 GPU test binaries:
    │                          test_basics        — encrypt/mult/bootstrap smoke
    │                          test_depth_budget  — level tracking and depth guard
    │                          test_linear        — BSGS + interleaved matmul
    │                          test_non_linear    — GELU, Norm, Softmax (with/without BTP)
    │                          test_gpt2_real     — per-layer diagnostic with real weights
    │                          test_primitives    — Newton-Raphson, Goldschmidt
    │                          test_poly_approx   — polynomial approximation accuracy
    │                          test_poly_approx_time — polynomial approximation timing
    │                          test_remez_taylor  — Remez/Taylor approximation
    │
    │  9. SMOKE TEST         DiagModel: 1-layer inference with real weights,
    │                        decrypts after each stage to check noise propagation
    │
    │ 10. FULL INFERENCE     Generate: 12-layer encrypted GPT-2, tokenize prompt,
    │                        embed → encrypt → 12 decoders → decrypt → argmax
    │                        → predicted next token
    └──────────────────────────────────────────────────────────────────────────
```

## Quick Start (Leonardo HPC / SLURM)

All compute runs via `sbatch`. Python runs via `uv run`.

### Option A: Run the entire pipeline

```bash
sbatch scripts/05_full_inference.sh                 # everything (~8-12 hours)
sbatch scripts/05_full_inference.sh --skip-bench     # reuse existing bench CSV
sbatch scripts/05_full_inference.sh --skip-weights   # reuse existing weights
```

### Option B: Run steps individually

```bash
# 1. Install dependencies (FIDESlib + OpenFHE → deps/)
sbatch scripts/01_install_deps.sh

# 2. Build
sbatch scripts/02_build.sh

# 3. Run tests
sbatch scripts/03_run_tests.sh

# 4. Generate weights
sbatch scripts/04a_generate_weights.sh               # standard BSGS
sbatch scripts/04c_generate_weights_interleaved.sh   # CacheMIR interleaved

# 5. Encrypted generation
sbatch scripts/11_generate_text.sh
sbatch scripts/11_generate_text.sh --text "Once upon a time" --layers 12
```

## CKKS Parameters (Paper-Aligned)

All parameters follow the CacheMIR paper (arXiv:2602.11470, §7.1):

| Parameter | Value | Notes |
|-----------|-------|-------|
| logN | 16 | Ring dim N'=65536, slots S=32768 |
| L (max level) | 13 | Working multiplicative depth |
| K (BTP depth) | 15 | Bootstrap circuit depth overhead |
| level_budget | {4, 3} | CtS=4, StC=3 levels |
| h_weight | 192 | Sparse ternary secret (Hamming weight) |
| Secret dist | SPARSE_ENCAPSULATED | Bossuat et al. 2022, special Chebyshev coefficients |
| btp_scale_bits | 50 | Per-level modulus in bootstrap path |
| first_mod_bits | 53 | log2(q0) ≈ 53 |
| scale_bits | 41 | Working modulus for non-bootstrap ops |

**Note on OpenFHE vs Lattigo:** The paper uses Lattigo which has separate P primes
(61-bit) for bootstrap. OpenFHE/FIDESlib bootstrap shares the Q modulus chain — all
K=15 bootstrap levels use the same working moduli. This requires `btp_scale_bits≥50`
(vs the paper's 41-bit which only works in Lattigo). See `project_bootstrap_params.md`
for the full analysis.

## Project Structure

```
pycudafideslib/
  include/
    fideslib_wrapper.h        # Thin CKKS wrapper: make_ckks_context(), encrypt/decrypt
    inference.h               # Inference struct: weights, raw_w, cache, dimensions
    gpt2.h                    # GPT-2 model/layer config, declarations
    gpt2_optimized_config.h   # AUTO-GENERATED by gen_config.py — do not hand-edit
    llama.h                   # LLaMA declarations
    nonlinear.h               # GELU/SiLU/Norm/Softmax configs & declarations
  src/
    main.cu               # CLI: -test Generate/Classify/Decoder/BenchAll/...
    gpt2.cu               # GPT-2 decoder, model, generate, classify, weight loading
    linear.cu             # BSGS + CacheMIR interleaved matmul, QKV, attention
    nonlinear.cu          # GELU, sign, softmax, LayerNorm (Newton-Raphson + Goldschmidt)
    bootstrap.cu          # bootstrap_to() — conditional BTP helper
    primitives.cu         # Newton-Raphson, Goldschmidt inverse square root
    llama.cu              # LLaMA decoder (RoPE, gated SiLU MLP)
  tests/
    test_basics.h/.cu       # encrypt/mult/bootstrap, shared context via SetUpTestSuite
    test_non_linear.h/.cu   # GELU, Norm, Softmax, SoftmaxCausalMask, SoftmaxWithoutOracle
    test_linear.cu          # BSGS + interleaved matmul correctness (LinearTest, RealModeLinearTest)
    test_gpt2_real.cu       # End-to-end layer tests with real weights (GPT2LinearTest, GPT2DecoderTest)
    test_depth_budget.cu    # Level tracking and DepthGuard
    test_primitives.cu      # Newton-Raphson, Goldschmidt
    test_poly_approx.cu     # Polynomial approximation accuracy
    test_poly_approx_time.cu# Polynomial approximation timing
    test_remez_taylor.cu    # Remez/Taylor approximation
    test_gen_config.py      # Python: gen_config.py placement→C++ translation
    test_bsgs_linear.py     # Python: BSGS packing numerical verification
    generate_gpt2_test_vectors.py  # Generate plaintext reference vectors for C++ tests
  scripts/
    00_pipeline.sh        # Legacy full pipeline (use 05_full_inference.sh instead)
    01_install_deps.sh    # Build FIDESlib + OpenFHE → deps/
    02_build.sh           # cmake + make (sbatch only — GPU needed for NCCL detection)
    03_run_tests.sh       # Build + run all 9 C++ test binaries (normal QOS, 2h)
    04a_generate_weights.sh     # Standard BSGS weights (logN=16)
    04c_generate_weights_interleaved.sh  # CacheMIR interleaved weights (logN=16)
    05_full_inference.sh  # **THE** pipeline: build→bench→placement→config→rebuild→
                          #   weights→vectors→py-tests→cpp-tests→smoke→full-generation
    11_generate_text.sh   # End-to-end encrypted text generation
    tokenize_input.py     # Tokenize prompt text → token IDs for C++ binary
  bootstrap.py            # DP placement optimizer (reads latency CSV, writes placement JSON)
  gen_config.py           # Placement JSON → include/gpt2_optimized_config.h
  prepare_gpt2_weights.py # HuggingFace GPT-2 → BSGS-packed .txt per layer
  estimate_ranges.py      # Plaintext inference → per-layer activation ranges → taylor_rescale
```

## Python Tests

### test_gen_config.py

Validates that `gen_config.py` correctly translates bootstrap placement JSON into
C++ config headers. Tests pruned routes (composed decoder layers), flat routes
(individual ops), both cpp and json output formats, and custom layer counts.
Run with: `uv run python tests/test_gen_config.py`

### test_bsgs_linear.py

Comprehensive numerical verification of the BSGS linear kernel for CKKS matrix-vector
multiply. Tests three packing modes:

1. **real_mode** (baby_step=1): correct for any S ≥ d
2. **bench_mode** (baby_step=intRot): only correct when S = d
3. **interleaved_mode** (CacheMIR): correct for any S ≥ d, fewer Galois keys

Covers: square matrices, rectangular Up/Down, streaming block-by-block, shrink
(d_in > d_out) with corrected giant step, and a memory efficiency table comparing
standard vs interleaved plaintext counts across configs.
Run with: `uv run python tests/test_bsgs_linear.py`

## C++ Tests

All test fixtures use `SetUpTestSuite()` with static context — the expensive
logN=16 CKKS context is built once per fixture class, not per test.

| Binary | What it tests |
|--------|--------------|
| test_basics | Basic encrypt/mult/decrypt, bootstrap at various levels |
| test_depth_budget | Level tracking, DepthGuard bootstrap triggers |
| test_linear | BSGS matmul (square, rectangular), interleaved matmul, rotation indices |
| test_non_linear | NormApprox (no padding), NormApproxWithPadding, GeLU, Softmax (oracle/no-oracle), sign() |
| test_gpt2_real | Per-layer tests with real HuggingFace weights vs plaintext reference vectors |
| test_primitives | Newton-Raphson inverse, Goldschmidt inverse square root |
| test_poly_approx | Chebyshev, Taylor polynomial approximation accuracy |
| test_poly_approx_time | Polynomial evaluation timing benchmarks |
| test_remez_taylor | Remez minimax vs Taylor polynomial comparison |

**Known issue:** Some test binaries exit with code 134 (SIGABRT) at teardown due to
heap corruption in OpenFHE/FIDESlib cleanup. All tests report PASSED before the crash.
The test runner script handles this by running each binary independently.

## Key Design Notes

### realHidDim / realFfDim Padding

GPT-2's hidden dimension (768) is padded to the next power of two (1024) for BSGS
efficiency. The original dimension is passed as `-realHidDim 768` so that LayerNorm
normalises over the correct 768 elements, not the padded 1024.

### BSGS vs CacheMIR Interleaved Packing

| | BSGS (standard) | CacheMIR (interleaved) |
|---|---|---|
| Weight plaintexts per matrix | d | d_in × d_out / S |
| Mult depth | 2 | 1 |
| Rotation cost | d + √d | 2 log₂(t) + 2 |
| Memory at logN=16, GPT-2 | 4096 Ptx/layer | 384 Ptx/layer (10.7×) |

### Bootstrap Placement

`bootstrap_to(inf, ct, target)` is **conditional**: fires only when remaining depth
is below target, otherwise no-op. The per-layer BTP levels are computed by the DP
optimizer (`bootstrap.py`) and baked into `gpt2_optimized_config.h`.

### Memory: raw_w vs w

`inf.w` holds FIDESlib `Ptx` objects (NTT-expanded, ~10 MB each at logN=16).
`inf.raw_w` holds raw `vector<double>` rows (8 bytes × S per row).

Embedding tables (wte 50k rows, wpe 1k rows, lm_head 50k rows) live in `raw_w` —
only the row needed at runtime is encoded to Ptx on-demand.

## GPT-2 Decoder Architecture (Encrypted)

Each of the 12 decoder layers runs this pipeline under encryption:

```
x_in ─┬─→ [BTP] → LayerNorm1 → QKV projections → KV cache
      │                         → QK^T → [BTP] → Softmax → AttnV → OutProj + bias
      │                                                              │
      └─→ reduce_to_level ──────────────────────────── + (residual) ─┘
                                                          │
      ┌─→ reduce_to_level ──────────────────────── + (residual) ─→ output
      │                                              │
      └── [BTP] → LayerNorm2 → Up + bias → [BTP] → GELU → Down + bias
```

After 12 layers: final LayerNorm → decrypt → lm_head argmax (plaintext) → next token.
