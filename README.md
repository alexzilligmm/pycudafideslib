# pycudafideslib

GPU-accelerated FHE inference for transformer models (GPT-2, LLaMA) using CKKS homomorphic encryption via [FIDESlib](third_party/FIDESlib/) + OpenFHE.

## Architecture

```
 prepare_gpt2_weights.py
 (HuggingFace -> .txt)
         |
         v
 estimate_ranges.py  --->  ranges-<model>.json
 (plaintext inference          (per-op min/max
  to profile activations)       value ranges)
         |
         v
 remez_cli  ------------->  poly_coeffs-<model>.json
 (minimax fit over             (standard monomial coefficients
  profiled ranges)              + recommended degree, for PS eval)
         |
         | (update src/nonlinear.cu if coefficients changed)
         v
 run_bench.sh  --->  cuda_cachemir  --->  raw_result.csv  --->  new_data.csv
 (sweep all ops       (C++/CUDA          (per-op timings)       (latency table)
  at each level)       benchmark)                                      |
                                                                       v
                                                              bootstrap.py
                                                              (optimal bootstrap
                                                               placement via DP)
                                                                       |
                                                                       v
                                                              gpt2_decoder() / decoder()
                                                              (FHE inference with
                                                               bootstrap_to() calls
                                                               at optimal positions)
```

## Quick Start

### 1. Build

```bash
# Install dependencies (FIDESlib + OpenFHE)
bash scripts/01_install_deps.sh

# Build
mkdir -p build && cd build
cmake .. -DFIDESLIB_ROOT=/path/to/fideslib
make -j$(nproc)
cd ..
```

The binary lands at `build/bin/cuda_cachemir`. Tests land in the same directory.

### 2. Run Tests

```bash
bash scripts/03_run_tests.sh
# or individually:
./build/bin/test_basics
./build/bin/test_linear
./build/bin/test_non_linear
```

### 3. Benchmark: Measure Per-Op Latencies

This is the key step. The bootstrap placement optimizer needs to know how long each FHE operation takes at each CKKS level on your specific GPU.

**LLaMA:**
```bash
./run_bench.sh --model llama --logN 16 --hidDim 4096 --ffDim 16384
```

**GPT-2:**
```bash
./run_bench.sh --model gpt2 --logN 16 --hidDim 768 --ffDim 3072
```

This loops over every `(operation, level)` pair, times each one, and produces:
- `raw_result.csv` -- per-run raw timings
- `new_data.csv` -- tab-separated latency table

You can also run individual benchmarks:
```bash
./build/bin/cuda_cachemir -test QKV -level 5 -logN 12
./build/bin/cuda_cachemir -test GELU -level 8 -logN 12
./build/bin/cuda_cachemir -test Softmax -level 16 -btpLevel 12 -logN 16
./build/bin/cuda_cachemir -test Decoder -model gpt2 -logN 12 -hidDim 768 -ffDim 3072
```

Available `-test` values:

| Test | Description | Model |
|------|-------------|-------|
| `QKV` | Single Q/K/V projection | both |
| `RoPE` | Rotary position embeddings | LLaMA |
| `Cache` | KV cache update | both |
| `QK_T` | QK^T dot product | both |
| `AttnV` | Attention-value multiply | both |
| `Out` | Output projection | both |
| `UpGate` | Up + Gate projection (2x) | LLaMA |
| `Up` | Single up projection | GPT-2 |
| `Down` | Down projection | both |
| `SiLU` | SiLU activation (Chebyshev-127) | LLaMA |
| `GELU` | GELU activation (piecewise poly) | GPT-2 |
| `CtMult` | Ciphertext-ciphertext multiply | both |
| `Softmax` | Softmax (with `-btpLevel` sweep) | both |
| `Norm` | LayerNorm (Newton + Goldschmidt) | both |
| `Decoder` | Full decoder layer | both |
| `Model` | Full model (32 or 12 layers) | both |

### 4. Optimize Bootstrap Placement

Once you have `new_data.csv` with measured latencies:

```bash
# LLaMA
python bootstrap.py --file new_data.csv --model llama --max-level 16

# GPT-2
python bootstrap.py --file new_data.csv --model gpt2 --max-level 16

# Custom CKKS parameters
python bootstrap.py --file new_data.csv --model gpt2 --boot-lat 42.0 --max-level 16
```

Output: the optimal route through the computation showing **where to place each `bootstrap_to()` call** and the expected end-to-end latency.

```
======================================================================
  gpt2 -- Optimal Bootstrap Placement
======================================================================
  End-to-end latency : 2340.00 s
  Route (12 steps):
    [  0] MHA                   level=0
    [  1] FFN                   level=3  <-- BOOTSTRAP (level 5 -> 3)
    [  2] MHA                   level=0  <-- BOOTSTRAP (level 6 -> 0)
    ...
  Bootstraps needed: 8
======================================================================
```

### 5. Estimate Activation Ranges and Fit Polynomial Approximations

Before building with real weights, profile the activation value ranges across the model so that polynomial approximations (GELU, SiLU, Softmax, Norm) are fitted over the correct domain. Wrong intervals waste levels or degrade accuracy.

**Step 5a — collect per-layer activation ranges from plaintext inference:**

```bash
# Run on real or representative inputs; writes ranges-<model>.json
python estimate_ranges.py --model gpt2 --weights weights-gpt2/ --out ranges-gpt2.json
python estimate_ranges.py --model llama --weights weights-llama/ --out ranges-llama.json
```

Output (`ranges-gpt2.json`) contains per-op min/max observed values, e.g.:
```json
{
  "gelu_input":   {"min": -8.2,  "max": 6.1},
  "softmax_input":{"min": -142.0,"max": 3.4},
  "norm_variance":{"min": 1e-4,  "max": 12.3}
}
```

**Step 5b — run the Remez CLI to produce minimax polynomial coefficients (standard basis):**

```bash
# Fits a minimax polynomial over the profiled ranges via the Remez algorithm;
# writes poly_coeffs-<model>.json with standard monomial coefficients + recommended degree.
# These feed directly into eval_polynomial_ps (Paterson-Stockmeyer, standard basis).
remez_cli --ranges ranges-gpt2.json --model gpt2 --out poly_coeffs-gpt2.json

# Use polyeval.py to compare approximation strategies (Remez, Taylor, Chebyshev)
# and inspect error vs. degree trade-offs — it is an analysis tool, not the fitter:
python polyeval.py --ranges ranges-gpt2.json --coeffs poly_coeffs-gpt2.json --plot
```

The Remez tests can also be run directly to validate a specific approximation:
```bash
./build/bin/test_remez_taylor   # unit tests for Remez & Taylor approximations
```

After updating coefficients in `src/nonlinear.cu`, rebuild before benchmarking:
```bash
make -C build -j
```

### 6. Extract GPT-2 Weights from HuggingFace

```bash
pip install transformers torch
python prepare_gpt2_weights.py --model gpt2 --out_dir weights-gpt2 --slots 2048
```

Produces `weights-gpt2/` with per-layer weight files (`.txt`, one plaintext vector per line):
```
weights-gpt2/
  wte.txt                  # token embeddings (50257 x 768)
  wpe.txt                  # position embeddings (1024 x 768)
  layer0_Wq.txt            # Q weight (768 x 768), flattened into slots-wide vectors
  layer0_bq.txt            # Q bias (768), zero-padded to slots
  layer0_Wu.txt            # MLP up weight (768 x 3072)
  ...
  ln_f_g.txt               # final LayerNorm gamma
  ln_f_b.txt               # final LayerNorm beta
```

### 6. Run FHE Inference

```bash
# GPT-2 decoder (single layer)
./build/bin/cuda_cachemir -test Decoder -model gpt2 \
    -logN 12 -hidDim 768 -ffDim 3072 -numHeads 12

# GPT-2 full model (12 layers)
./build/bin/cuda_cachemir -test Model -model gpt2 \
    -logN 16 -hidDim 768 -ffDim 3072 -seqLen 1024 -numHeads 12

# LLaMA full model (32 layers)
./build/bin/cuda_cachemir -test Model -model llama \
    -logN 16 -hidDim 4096 -ffDim 11008 -seqLen 512 -numHeads 32
```

## Full Pipeline (End to End)

```bash
# 1. Extract weights
pip install transformers torch
python prepare_gpt2_weights.py --model gpt2 --out_dir weights-gpt2 --slots 32768

# 2. Profile activation ranges (plaintext, no FHE)
python estimate_ranges.py --model gpt2 --weights weights-gpt2/ --out ranges-gpt2.json

# 3. Fit minimax polynomials (standard basis, for PS evaluation)
remez_cli --ranges ranges-gpt2.json --model gpt2 --out poly_coeffs-gpt2.json
#    → update coefficients in src/nonlinear.cu if needed

# 4. Build
mkdir -p build && cd build && cmake .. && make -j && cd ..

# 5. Benchmark all ops to get latency table
./run_bench.sh --model gpt2 --logN 16 --hidDim 768 --ffDim 3072

# 6. run_bench.sh automatically calls bootstrap.py at the end,
#    but you can also re-run with different parameters:
python bootstrap.py --file new_data.csv --model gpt2

# 7. Read the placement output, update bootstrap_to() targets
#    in src/gpt2.cu if needed, rebuild, and run inference:
make -C build -j
./build/bin/cuda_cachemir -test Model -model gpt2 \
    -logN 16 -hidDim 768 -ffDim 3072 -numHeads 12
```

## Project Structure

```
pycudafideslib/
  include/
    fideslib_wrapper.h    # CKKS context, encode, encrypt, decrypt
    inference.h           # Inference struct (weights, cache, dimensions)
    llama.h               # LLaMA declarations
    gpt2.h                # GPT-2 declarations
    nonlinear.h           # GELU/SiLU/Norm/Softmax configs & declarations
    ckks_primitives.h     # Newton-Raphson, Goldschmidt, Chebyshev
  src/
    main.cu               # CLI benchmark harness
    linear.cu             # BSGS matrix-vector multiply, QKV, attention
    nonlinear.cu          # GELU, SiLU, sign, softmax, norm, argmax
    llama.cu              # LLaMA decoder (32 layers, RoPE, gated SiLU MLP)
    gpt2.cu               # GPT-2 decoder (12 layers, no RoPE, GELU MLP)
    primitives.cu         # Newton-Raphson, Goldschmidt inv-sqrt
    poly_eval.cu          # Chebyshev, Horner, Paterson-Stockmeyer, rational
    bootstrap.cu          # bootstrap_to() helper
  tests/
    test_basics.cu        # FHE context smoke test
    test_linear.cu        # BSGS matmul correctness
    test_non_linear.cu    # GELU, softmax, sign accuracy
    test_primitives.cu    # Goldschmidt, Newton-Raphson
    test_poly_approx.cu   # Polynomial evaluation methods
    test_remez_taylor.cu  # Remez & Taylor approximation
  bootstrap.py            # Bootstrap placement optimizer (DP solver)
  test_bootstrap.py       # Tests for bootstrap.py
  run_bench.sh            # Automation: benchmark all ops -> data.csv -> bootstrap.py
  prepare_gpt2_weights.py # Extract GPT-2 weights from HuggingFace
  polyeval.py             # Polynomial evaluation comparison (offline)
  CMakeLists.txt          # Build system
  third_party/FIDESlib/   # GPU-accelerated FHE library (OpenFHE 1.4.2)
  cachemir/               # Original Go/Lattigo reference implementation
```

## Approximation Variants

The bootstrap optimizer supports exploring multiple approximation strategies per nonlinear op. To add a new variant (e.g. GELU via Chebyshev instead of piecewise):

1. Implement it in `nonlinear.cu`
2. Benchmark it: add a row `GELU_cheb` to your data CSV
3. Register it in `bootstrap.py`:
   ```python
   def solve_gelu_chebyshev(cfg, data, prune):
       p = cfg.placer("GELU_cheb")
       p.add_layer("GELU_cheb", data["GELU_cheb"])
       p.add_layer("last_layer", cfg.zero_layer())
       return p.solve_latency_only(prune)

   ACTIVATION_VARIANTS["GELU"].append(("chebyshev", solve_gelu_chebyshev))
   ```

The optimizer will automatically try all registered variants and pick the element-wise best across the level-to-level latency table.

## CKKS Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `logN` | 12-16 | Ring dimension N = 2^logN. Larger = more slots but slower |
| `depth` | 24 | Multiplicative depth budget |
| `scale_bits` | 40 | Per-level modulus size |
| `bootstrap` | enabled | +9 extra levels for bootstrap circuit |
| `slots` | N/2 | SIMD packing width |

## GPT-2 vs LLaMA

| | GPT-2 small | LLaMA 7B |
|---|---|---|
| Layers | 12 | 32 |
| hidDim | 768 | 4096 |
| ffDim | 3072 | 11008 |
| Heads | 12 | 32 |
| Position encoding | Learned (added at input) | RoPE (per-layer) |
| MLP | Up -> GELU -> Down | Up + Gate -> SiLU -> elem_mult -> Down |
| Biases | Yes (all layers) | No |
| Activation | GELU (piecewise, ~20 levels) | SiLU (Chebyshev-127, ~7 levels) |


