"""
Bootstrap placement optimizer.

Given per-operation latency profiles at each CKKS level, finds the optimal
placement of bootstrapping operations to minimise end-to-end latency through
a transformer model.

Supports:
  - Multiple approximation variants per nonlinear op  (Cachemir-style exploration)
  - Data-driven pipeline definitions                   (no copy-paste per model)
  - Sub-solution caching                               (solve each sub-circuit once)
  - Parameterised CKKS settings                        (boot_lat, max_level)

Usage:
    python bootstrap.py --file data.csv --model llama --max-level 16
    python bootstrap.py --file data_gpt2.csv --model gpt2 --boot-lat 42.0
"""

import numpy as np
import pandas as pd
import argparse
import copy
import json
import time


# ── Core solver ──────────────────────────────────────────────────────────

class Layer:
    def __init__(self, name, lat, max_level):
        self.name = name
        self.lat = lat
        self.depth = 0
        if not isinstance(lat[0], list):
            for l in lat:
                if l == np.inf:
                    self.depth += 1
            self.compose = False
        else:
            self.compose = True
        self.min_lat = [np.inf] * (max_level + 1)
        self.route = [[] for _ in range(max_level + 1)]


class Bootplacer:
    def __init__(self, name, boot_lat, max_level, choose_min=False):
        self.layers = []
        self.name = name
        self.boot_lat = boot_lat
        self.max_level = max_level
        self.choose_min = choose_min

    def add_layer(self, name, lat):
        self.layers.append(Layer(name, lat, self.max_level))

    def solve(self, prune):
        ml = self.max_level
        bl = self.boot_lat
        output = [[np.inf for _ in range(ml + 1)] for _ in range(ml + 1)]
        # routes_out[i][j] = route for input level i, output level j
        routes_out = [[None for _ in range(ml + 1)] for _ in range(ml + 1)]

        for i in range(ml + 1):
            self.layers[0].min_lat = [0] * (ml + 1)
            for j in range(len(self.layers) - 1):
                lj = self.layers[j]
                lj1 = self.layers[j + 1]
                lj1.min_lat = [np.inf] * (ml + 1)
                lj.route = [[] for _ in range(ml + 1)]

                if not lj.compose:
                    if j == 0:
                        for k in range(ml - lj.depth + 1):
                            lj1.min_lat[k] = lj.lat[k + lj.depth]
                            if i < k + lj.depth:
                                lj1.min_lat[k] += bl
                            lj.route[k] = [(lj.name, i)]
                    else:
                        ljm1 = self.layers[j - 1]
                        for k in range(ml - lj.depth + 1):
                            if prune:
                                from_higher = (
                                    lj.min_lat[k + lj.depth]
                                    + lj.lat[k + lj.depth]
                                )
                                from_lower = (
                                    lj.min_lat[0]
                                    + lj.lat[k + lj.depth]
                                    + bl
                                )
                                if from_higher < from_lower:
                                    lj1.min_lat[k] = from_higher
                                    lj.route[k] = copy.deepcopy(
                                        ljm1.route[k + lj.depth]
                                    )
                                    lj.route[k].append(
                                        (lj.name, k + lj.depth)
                                    )
                                else:
                                    lj1.min_lat[k] = from_lower
                                    lj.route[k] = copy.deepcopy(ljm1.route[0])
                                    lj.route[k].append((lj.name, 0))
                            else:
                                for l in range(ml + 1):
                                    lat = (
                                        lj.min_lat[l]
                                        + lj.lat[k + lj.depth]
                                    )
                                    if l < k + lj.depth:
                                        lat += bl
                                    if lat < lj1.min_lat[k]:
                                        lj1.min_lat[k] = lat
                                        lj.route[k] = copy.deepcopy(
                                            ljm1.route[l]
                                        )
                                        lj.route[k].append((lj.name, l))
                else:
                    if j == 0:
                        for k in range(ml + 1):
                            lj1.min_lat[k] = lj.lat[i][k]
                            lj.route[k] = [(lj.name, i)]
                    else:
                        ljm1 = self.layers[j - 1]
                        for k in range(ml + 1):
                            for l in range(ml + 1):
                                cost = lj.min_lat[k] + lj.lat[k][l]
                                if lj1.min_lat[l] > cost:
                                    lj1.min_lat[l] = cost
                                    lj.route[l] = copy.deepcopy(ljm1.route[k])
                                    lj.route[l].append((lj.name, k))

            for j in range(ml + 1):
                output[i][j] = self.layers[-1].min_lat[j]
                if np.isfinite(output[i][j]):
                    routes_out[i][j] = copy.deepcopy(self.layers[-2].route[j])

        if self.choose_min:
            global_min = min(min(row) for row in output)
            print(f"[{self.name}] minimal end-to-end latency: {global_min:.2f}")

        return output, routes_out

    def solve_latency_only(self, prune):
        """Return only the latency table (for sub-circuits that feed into
        compose layers where routes are not needed at the outer level)."""
        output, _ = self.solve(prune)
        return output

    def solve_with_routes(self, prune):
        """Return (latency_table, routes_table)."""
        return self.solve(prune)

    def get_route(self, prune, input_level, output_level):
        """Solve and return the route for a specific input→output pair."""
        _, routes = self.solve(prune)
        r = routes[input_level][output_level]
        return r if r is not None else []


# ── CKKS config ──────────────────────────────────────────────────────────

class CKKSConfig:
    """CKKS parameters that affect bootstrap placement."""
    def __init__(self, boot_lat=56.08, max_level=16):
        self.boot_lat = boot_lat
        self.max_level = max_level

    def placer(self, name, choose_min=False):
        return Bootplacer(name, self.boot_lat, self.max_level, choose_min)

    def zero_layer(self):
        return [0] * (self.max_level + 1)


# ── Data loading ─────────────────────────────────────────────────────────

def add_shortcut(data, boot_lat):
    for i in range(len(data)):
        for j in range(i):
            data[j][i] += boot_lat
    return data


def read_data(file_name):
    df = pd.read_csv(file_name, sep="\t")
    result_dict = {}
    for _, row in df.iterrows():
        values = row.values.tolist()
        for i, value in enumerate(values):
            if value == 0:
                values[i] = np.inf
        name = values[0]
        values[0] = np.inf
        result_dict[name] = values

    if "RoPE" in result_dict:
        result_dict["Cache"] = [
            result_dict["RoPE"][i] + result_dict["Cache"][i]
            for i in range(len(result_dict["RoPE"]))
        ]
        del result_dict["RoPE"]

    return result_dict


# ── Sub-circuit definitions ──────────────────────────────────────────────
#
# Each sub-circuit is a function(cfg, data, prune) -> level-to-level latency table.
# For nonlinear ops with multiple approximation strategies, we define one
# function per variant and let `best_variant()` pick the winner.


class SolverCache:
    """Memoises sub-circuit solutions so shared blocks (norm, softmax, ...)
    are solved once even when referenced from multiple places."""
    def __init__(self):
        self._cache = {}

    def get(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return copy.deepcopy(self._cache[key])

    def clear(self):
        self._cache.clear()


def best_variant(variants, cfg, data, prune):
    """Given a list of (name, solver_fn) pairs, solve each and return the
    element-wise minimum latency table.  This is the key extension point
    for Cachemir-style exploration across approximation strategies."""
    best = None
    best_name = None
    for name, fn in variants:
        table = fn(cfg, data, prune)
        if best is None:
            best = table
            best_name = name
        else:
            ml = cfg.max_level
            improved = False
            for i in range(ml + 1):
                for j in range(ml + 1):
                    if table[i][j] < best[i][j]:
                        best[i][j] = table[i][j]
                        improved = True
            if improved:
                best_name = name + "+" + best_name
    print(f"  best variant(s): {best_name}")
    return best


# ── Iteration-level sub-circuit configs ──────────────────────────────────
#
# These mirror the C++ config structs and let the optimizer decide at which
# iteration of an iterative method to insert a bootstrap.
#
# Depth costs per iteration (from src/primitives.cu):
#   NR iteration (inv_sqrt_newton): EvalMult(ct,1.5) + Square + Mult×2 + Sub = 3 mults
#   GS iteration (goldschmidt_inv): Add + Mult(x0,factor) + Square(E) = 2 mults
#   GS inv_sqrt (goldschmidt_inv_sqrt): Mult(sqrt,ans) + Mult×(-0.5) + Add + Mult×2 = 3 mults
#
# GELU depth (from src/nonlinear.cu):
#   sign(): F4(F4(G4(G4(x)))) — 4 degree-9 poly_ps, each consuming ceil(log2(9))=4
#           CKKS levels → 16 actual CKKS levels total.
#   lt_function(): sign + 2 mults → 18 actual CKKS levels.
#   C++ GELU requires remaining ≥ 19 at input (no internal bootstraps in sign()).
#
# NOTE: The optimizer's piecewise model uses sign_depth=4 *model-units* (1 per
# poly eval, not 4 per poly eval), treating each polynomial as one CtMult latency
# unit for timing purposes. This is correct for latency accounting but means the
# model underestimates sign's actual CKKS depth by 4×. The piecewise solver
# compensates by scheduling 22 internal bootstraps, but C++ sign() has none.
# extract_decoder_config() therefore clamps gelu_btp_level ≥ max_level so the
# C++ implementation always receives remaining=max_level ≥ 19 at GELU input.

class EncLLMConfig:
    """EncLLM-specific iteration counts and depth profiles for sub-circuit modeling."""
    # Norm: inv_sqrt via Newton-Raphson, then multiply by Goldschmidt inv_sqrt
    nr_iters = 16       # default from NORM_ENCLLM_GPT2
    gs_iters = 14       # default from NORM_ENCLLM_GPT2
    # Softmax: exp squarings + Goldschmidt inverse
    exp_r = 7           # squarings in exp_approx
    gs_inv_iters = 14   # Goldschmidt inverse iterations
    # GELU: sign depth for lt_function
    # Each value is in optimizer model-units (1 per poly eval = 1 CtMult latency unit).
    # Actual CKKS depth of sign() = 4 poly evals × 4 levels = 16 (see note above).
    sign_depth = 4      # 4 poly evals in sign (F4∘F4∘G4∘G4), 1 model-unit each
    lt_extra = 2        # subtract + mult + sign + mult in lt_function beyond sign
    gelu_poly_depth = 3 # poly_f0/f1 degree 3-6 via PS = ~3 levels
    gelu_combine = 3    # 3× indicator × polynomial multiplications

# Default EncLLM config matching NORM_ENCLLM_GPT2 / SOFTMAX_ENCLLM_GPT2 / GELU_ENCLLM_GPT2
ENCLLM_CFG = EncLLMConfig()


# ── Shared sub-circuits (model-agnostic) ─────────────────────────────────

# --- Softmax sub-circuit (per-iteration Goldschmidt) ---

def solve_shortcut_softmax(cfg, data, prune):
    """Goldschmidt inverse iterations inside softmax.
    Each GS iteration costs 2 levels (mult + square)."""
    p = cfg.placer("Shortcut_softmax")
    if "Softmax" in data:
        # Legacy: treat as a single block if benchmarked as one
        p.add_layer("GoldIter", data["Softmax"])
    else:
        # Per-iteration model: gs_inv_iters × 2 mults
        for i in range(ENCLLM_CFG.gs_inv_iters):
            p.add_layer(f"gs_inv_{i}", data["CtMult"])
            p.add_layer(f"gs_inv_{i}_sq", data["CtMult"])
    p.add_layer("mult", data["CtMult"])
    p.add_layer("last_layer", cfg.zero_layer())
    return p.solve_latency_only(prune)


def solve_softmax(cfg, data, prune):
    """Full softmax: max phase (8 sign-based comparisons) + exp + GS inverse.
    The 8 mults model the sign-based max computation (log2(S/stride) steps,
    each consuming ~1 mult level for the comparison arithmetic)."""
    p = cfg.placer("Softmax")
    lat_sc = add_shortcut(solve_shortcut_softmax(cfg, data, prune), cfg.boot_lat)
    # Max phase: each step = rotation + sub + sign + mult + add + mult
    # sign itself is very deep (~16 levels for G4∘G4∘F4∘F4) but benchmarked
    # as part of the Softmax block. Model as 8 comparison steps.
    for i in range(8):
        p.add_layer(f"max_cmp_{i}", data["CtMult"])
    # exp_approx: 1 mult + exp_r squarings
    p.add_layer("exp_mult", data["CtMult"])
    for i in range(ENCLLM_CFG.exp_r):
        p.add_layer(f"exp_sq_{i}", data["CtMult"])
    # GS inverse as composed sub-circuit
    p.add_layer("gs_inverse", lat_sc)
    p.add_layer("last_layer", cfg.zero_layer())
    return p.solve_latency_only(prune)


# --- Norm sub-circuit (per-iteration NR + GS) ---

def solve_shortcut_norm(cfg, data, prune):
    """NR iterations followed by GS iterations inside norm.
    NR: 3 levels/iter, GS: 2 levels/iter (from goldschmidt_inv_sqrt)."""
    p = cfg.placer("Shortcut_norm")
    if "SqrtNt" in data and "SqrtGold" in data:
        # Hybrid: use benchmarked blocks but model them as multiple iterations
        # so the optimizer can place bootstraps between NR and GS phases.
        # NR phase: nr_iters iterations × 3 mults each
        for i in range(ENCLLM_CFG.nr_iters):
            p.add_layer(f"nr_{i}_a", data["CtMult"])  # EvalMult(ct, 1.5)
            p.add_layer(f"nr_{i}_b", data["CtMult"])  # EvalSquare + EvalMult
            p.add_layer(f"nr_{i}_c", data["CtMult"])  # EvalMult(c, y3)
        # GS phase: gs_iters iterations × 3 mults each
        for i in range(ENCLLM_CFG.gs_iters):
            p.add_layer(f"gs_{i}_a", data["CtMult"])  # EvalMult(sqrt, ans)
            p.add_layer(f"gs_{i}_b", data["CtMult"])  # Mult(-0.5) + Add
            p.add_layer(f"gs_{i}_c", data["CtMult"])  # Mult(sqrt, res) or Mult(ans, res)
    p.add_layer("final_mult", data["CtMult"])
    p.add_layer("last_layer", cfg.zero_layer())
    return p.solve_latency_only(prune)


def solve_norm(cfg, data, prune):
    """Full norm: mean/var computation + NR + GS + final multiply."""
    p = cfg.placer("Norm")
    lat_sc = add_shortcut(solve_shortcut_norm(cfg, data, prune), cfg.boot_lat)
    # Mean/variance computation: ~1 mult
    p.add_layer("mean_var", data["CtMult"])
    # NR + GS as composed sub-circuit
    p.add_layer("nr_gs", lat_sc)
    p.add_layer("last_layer", cfg.zero_layer())
    return p.solve_latency_only(prune)


# ── Activation variants ─────────────────────────────────────────────────
#
# Each variant is a function(cfg, data, prune) -> latency table.
# Add new approximation strategies here and register them in the
# ACTIVATION_VARIANTS dict below.

def solve_silu_chebyshev(cfg, data, prune):
    """SiLU via single Chebyshev series (e.g. degree 127)."""
    p = cfg.placer("SiLU_cheb")
    p.add_layer("SiLU", data["SiLU"])
    p.add_layer("last_layer", cfg.zero_layer())
    return p.solve_latency_only(prune)


def solve_gelu_piecewise(cfg, data, prune):
    """GELU via piecewise polynomial: 3× lt_function + 3× bootstrap + combine.

    Each lt_function = sign(x - threshold) + 2 mults.
    sign = G4(G4(F4(F4(x)))) = 4 degree-9 poly_ps evals ≈ 4 levels each.
    After all 3 lt_functions: 3 bootstraps refresh the indicators.
    Then: 2 poly_eval (degree 3,6 ≈ 3 levels) + 3 indicator×poly mults.

    The optimizer can place bootstraps between the 3 lt_function evaluations
    and between the indicator-combine and polynomial phases."""
    p = cfg.placer("GELU_pw")
    if "GELU" in data:
        # Check if we have per-step data; if not, decompose using CtMult
        sign_d = ENCLLM_CFG.sign_depth + ENCLLM_CFG.lt_extra
        # 3× lt_function phases (each ≈ sign_depth + lt_extra levels)
        for lt_idx in range(3):
            for s in range(sign_d):
                p.add_layer(f"lt{lt_idx}_sign_{s}", data["CtMult"])
        # 3× bootstrap (modeled as a shortcut to level 0 — free via bootstrap)
        # The optimizer sees the level reset via the shortcut mechanism.
        # We model each bootstrap as consuming boot_lat but resetting level.
        # For simplicity, add a composed layer that includes bootstrap cost.
        # After bootstraps: indicator combine (3 subs/adds ≈ 0 levels)
        # Then: 2 poly evals + 3 final mults
        for i in range(ENCLLM_CFG.gelu_poly_depth):
            p.add_layer(f"poly_{i}", data["CtMult"])
        for i in range(ENCLLM_CFG.gelu_combine):
            p.add_layer(f"combine_{i}", data["CtMult"])
    p.add_layer("last_layer", cfg.zero_layer())
    return p.solve_latency_only(prune)


def solve_gelu_atomic(cfg, data, prune):
    """GELU as a single benchmarked block (legacy, no internal bootstrap control)."""
    p = cfg.placer("GELU_atomic")
    p.add_layer("GELU", data["GELU"])
    p.add_layer("last_layer", cfg.zero_layer())
    return p.solve_latency_only(prune)


# Registry: maps activation name -> list of (variant_name, solver_fn).
# best_variant() will try all and take the element-wise minimum.
ACTIVATION_VARIANTS = {
    "SiLU": [
        ("chebyshev", solve_silu_chebyshev),
        # ("rational",  solve_silu_rational),   # add when profiled
    ],
    "GELU": [
        ("piecewise_iter", solve_gelu_piecewise),
        ("atomic", solve_gelu_atomic),
        # ("chebyshev", solve_gelu_chebyshev),  # add when profiled
    ],
}


def solve_activation(name, cfg, data, prune, cache):
    """Solve the best activation variant, with caching."""
    variants = ACTIVATION_VARIANTS.get(name, [])
    # Filter to variants whose data keys actually exist
    available = []
    for vname, fn in variants:
        try:
            fn(cfg, data, prune)  # test if data keys exist
            available.append((vname, fn))
        except KeyError:
            pass
    if not available:
        raise KeyError(f"No data for any {name} variant. "
                       f"Expected one of: {[v[0] for v in variants]}")
    return cache.get(
        f"activation_{name}",
        lambda: best_variant(available, cfg, data, prune)
    )


# ── Norm variant support ────────────────────────────────────────────────
#
# The C++ side already supports LINEAR / REMEZ / TAYLOR init for the
# Newton-Raphson seed.  To explore these in the bootstrap placer, add
# rows like "SqrtNt_remez", "SqrtNt_taylor" to your data.csv and
# register them:
#
# NORM_VARIANTS = [
#     ("newton",  solve_shortcut_norm),            # current
#     ("remez",   solve_shortcut_norm_remez),       # with data["SqrtNt_remez"]
#     ("taylor",  solve_shortcut_norm_taylor),      # with data["SqrtNt_taylor"]
# ]


# ── Model-specific pipelines ────────────────────────────────────────────
#
# Each model defines its pipeline as a function that composes the shared
# sub-circuits.  The only model-specific knowledge is the layer sequence
# and count.

# ── LLaMA ──

def solve_qk_llama(cfg, data, prune, cache):
    def _solve():
        p = cfg.placer("QK")
        lat_sm = cache.get("softmax", lambda: solve_softmax(cfg, data, prune))
        p.add_layer("Cache", data["Cache"])
        p.add_layer("QK_T", data["QK_T"])
        p.add_layer("Softmax", lat_sm)
        p.add_layer("last_layer", cfg.zero_layer())
        return p.solve_latency_only(prune)
    return cache.get("qk_llama", _solve)


def solve_MHA_llama(cfg, data, prune, cache):
    def _solve():
        p = cfg.placer("MHA")
        lat_qk = add_shortcut(solve_qk_llama(cfg, data, prune, cache), cfg.boot_lat)
        lat_norm = cache.get("norm", lambda: solve_norm(cfg, data, prune))
        p.add_layer("Norm", lat_norm)
        p.add_layer("QKV", [3 * t for t in data["QKV"]])
        p.add_layer("RoPE_to_AttnV", lat_qk)
        p.add_layer("AttnV", data["AttnV"])
        p.add_layer("O", data["QKV"])
        p.add_layer("last_layer", cfg.zero_layer())
        return p.solve_latency_only(prune)
    return cache.get("mha_llama", _solve)


def solve_FFN_llama(cfg, data, prune, cache):
    def _solve():
        p = cfg.placer("FFN")
        lat_norm = cache.get("norm", lambda: solve_norm(cfg, data, prune))
        lat_act = add_shortcut(
            solve_activation("SiLU", cfg, data, prune, cache), cfg.boot_lat
        )
        p.add_layer("Norm", lat_norm)
        p.add_layer("UpGate", [2 * t for t in data["UpGate"]])
        p.add_layer("SiLU", lat_act)
        p.add_layer("elem_mult", data["CtMult"])
        p.add_layer("Down", data["Down"])
        p.add_layer("last_layer", cfg.zero_layer())
        return p.solve_latency_only(prune)
    return cache.get("ffn_llama", _solve)


def solve_decoder_llama(cfg, data, prune, cache):
    def _solve():
        p = cfg.placer("Decoder")
        lat_MHA = add_shortcut(
            solve_MHA_llama(cfg, data, prune, cache), cfg.boot_lat
        )
        lat_FFN = add_shortcut(
            solve_FFN_llama(cfg, data, prune, cache), cfg.boot_lat
        )
        if prune:
            p.add_layer("MHA", lat_MHA)
            p.add_layer("FFL", lat_FFN)
        else:
            for i in range(32):
                p.add_layer("MHA_i", lat_MHA)
                p.add_layer("FFL_i", lat_FFN)
        p.add_layer("last_layer", cfg.zero_layer())
        return p.solve_latency_only(prune)
    return cache.get("decoder_llama", _solve)


def solve_model_llama(cfg, data, prune):
    print("Solving LLaMA Model...\n")
    cache = SolverCache()
    p = cfg.placer("Model", choose_min=True)
    lat_dec = solve_decoder_llama(cfg, data, prune, cache)
    if prune:
        for i in range(32):
            p.add_layer(f"decoder_{i}", lat_dec)
    else:
        p.add_layer("decoder", lat_dec)
    p.add_layer("last_layer", cfg.zero_layer())
    return p.solve(prune)


# ── GPT-2 ──

GPT2_N_LAYERS = 12


def solve_qk_gpt2(cfg, data, prune, cache):
    def _solve():
        p = cfg.placer("QK_GPT2")
        lat_sm = cache.get("softmax", lambda: solve_softmax(cfg, data, prune))
        # No RoPE, no cache rotation
        p.add_layer("QK_T", data["QK_T"])
        p.add_layer("Softmax", lat_sm)
        p.add_layer("last_layer", cfg.zero_layer())
        return p.solve_latency_only(prune)
    return cache.get("qk_gpt2", _solve)


def solve_MHA_gpt2(cfg, data, prune, cache):
    def _solve():
        p = cfg.placer("MHA_GPT2")
        lat_qk = add_shortcut(
            solve_qk_gpt2(cfg, data, prune, cache), cfg.boot_lat
        )
        lat_norm = cache.get("norm", lambda: solve_norm(cfg, data, prune))
        p.add_layer("Norm", lat_norm)
        p.add_layer("QKV", [3 * t for t in data["QKV"]])
        p.add_layer("QK_to_AttnV", lat_qk)
        p.add_layer("AttnV", data["AttnV"])
        p.add_layer("O", data["QKV"])
        p.add_layer("last_layer", cfg.zero_layer())
        return p.solve_latency_only(prune)
    return cache.get("mha_gpt2", _solve)


def solve_FFN_gpt2(cfg, data, prune, cache):
    def _solve():
        p = cfg.placer("FFN_GPT2")
        lat_norm = cache.get("norm", lambda: solve_norm(cfg, data, prune))
        lat_act = add_shortcut(
            solve_activation("GELU", cfg, data, prune, cache), cfg.boot_lat
        )
        # No gate, no elem_mult
        p.add_layer("Norm", lat_norm)
        p.add_layer("Up", data["Up"])
        p.add_layer("GELU", lat_act)
        p.add_layer("Down", data["Down"])
        p.add_layer("last_layer", cfg.zero_layer())
        return p.solve_latency_only(prune)
    return cache.get("ffn_gpt2", _solve)


def solve_decoder_gpt2(cfg, data, prune, cache):
    def _solve():
        p = cfg.placer("Decoder_GPT2")
        lat_MHA = add_shortcut(
            solve_MHA_gpt2(cfg, data, prune, cache), cfg.boot_lat
        )
        lat_FFN = add_shortcut(
            solve_FFN_gpt2(cfg, data, prune, cache), cfg.boot_lat
        )
        if prune:
            p.add_layer("MHA", lat_MHA)
            p.add_layer("FFN", lat_FFN)
        else:
            for i in range(GPT2_N_LAYERS):
                p.add_layer("MHA_i", lat_MHA)
                p.add_layer("FFN_i", lat_FFN)
        p.add_layer("last_layer", cfg.zero_layer())
        return p.solve_latency_only(prune)
    return cache.get("decoder_gpt2", _solve)


def solve_model_gpt2(cfg, data, prune):
    print("Solving GPT-2 Model...\n")
    cache = SolverCache()
    p = cfg.placer("GPT2_Model", choose_min=True)
    lat_dec = solve_decoder_gpt2(cfg, data, prune, cache)
    if prune:
        for i in range(GPT2_N_LAYERS):
            p.add_layer(f"decoder_{i}", lat_dec)
    else:
        p.add_layer("decoder", lat_dec)
    p.add_layer("last_layer", cfg.zero_layer())
    return p.solve(prune)


# ── Detailed route explanation ────────────────────────────────────────────

def _route_bootstraps_summary(route):
    """Return a compact string showing bootstrap positions in a route."""
    btps = []
    for t in range(1, len(route)):
        _, prev_lvl = route[t - 1]
        cur_name, cur_lvl = route[t]
        if cur_lvl < prev_lvl:
            btps.append((t, cur_name, prev_lvl, cur_lvl))
    return btps


def _solve_subcircuit_route(placer_fn, cfg, data, prune, input_lvl, output_lvl):
    """Build a sub-circuit placer, solve it, return route for given levels."""
    p = placer_fn(cfg, data, prune)
    return p.get_route(prune, input_lvl, output_lvl)


def _print_route_detail(route, indent="    "):
    """Print a route with bootstrap markers."""
    btps = _route_bootstraps_summary(route)
    btp_positions = {b[0] for b in btps}
    for t, (name, lvl) in enumerate(route):
        marker = ""
        if t in btp_positions:
            b = next(b for b in btps if b[0] == t)
            marker = f"  *** BTP ({b[2]}→{b[3]})"
        print(f"{indent}[{t:3d}] {name:25s} lvl={lvl}{marker}")
    n_btps = len(btps)
    print(f"{indent}  ({n_btps} bootstrap{'s' if n_btps != 1 else ''} in this sub-circuit)")


def explain_gpt2_decoder(cfg, data, prune, input_lvl, logN=16):
    """Print detailed internal bootstrap placement for one GPT-2 decoder.

    Solves the decoder sub-circuits using the actual composed solver to get
    the jointly-optimal route, then expands each composed layer into its
    internal route showing bootstrap positions at each iteration."""
    cache = SolverCache()

    # ── Build each sub-circuit ──
    def _build_shortcut_norm():
        p = cfg.placer("Shortcut_norm")
        if "SqrtNt" in data and "SqrtGold" in data:
            for i in range(ENCLLM_CFG.nr_iters):
                p.add_layer(f"NR_{i}_mult1.5", data["CtMult"])
                p.add_layer(f"NR_{i}_square", data["CtMult"])
                p.add_layer(f"NR_{i}_mult", data["CtMult"])
            for i in range(ENCLLM_CFG.gs_iters):
                p.add_layer(f"GS_{i}_mult_sqrt", data["CtMult"])
                p.add_layer(f"GS_{i}_mult_half", data["CtMult"])
                p.add_layer(f"GS_{i}_mult_res", data["CtMult"])
        p.add_layer("final_mult", data["CtMult"])
        p.add_layer("last_layer", cfg.zero_layer())
        return p

    def _build_softmax_full():
        p = cfg.placer("Softmax_full")
        lat_gs = add_shortcut(
            _build_shortcut_softmax().solve_latency_only(prune), cfg.boot_lat)
        for i in range(8):
            p.add_layer(f"max_cmp_{i}", data["CtMult"])
        p.add_layer("exp_mult", data["CtMult"])
        for i in range(ENCLLM_CFG.exp_r):
            p.add_layer(f"exp_sq_{i}", data["CtMult"])
        p.add_layer("GS_inverse", lat_gs)
        p.add_layer("last_layer", cfg.zero_layer())
        return p

    def _build_shortcut_softmax():
        p = cfg.placer("Shortcut_softmax")
        if "Softmax" in data:
            p.add_layer("GoldIter", data["Softmax"])
        else:
            for i in range(ENCLLM_CFG.gs_inv_iters):
                p.add_layer(f"GS_inv_{i}", data["CtMult"])
                p.add_layer(f"GS_inv_{i}_sq", data["CtMult"])
        p.add_layer("mult", data["CtMult"])
        p.add_layer("last_layer", cfg.zero_layer())
        return p

    def _build_gelu_pw():
        p = cfg.placer("GELU_pw")
        sign_d = ENCLLM_CFG.sign_depth + ENCLLM_CFG.lt_extra
        for lt_idx in range(3):
            for s in range(sign_d):
                p.add_layer(f"lt{lt_idx}_sign_{s}", data["CtMult"])
        for i in range(ENCLLM_CFG.gelu_poly_depth):
            p.add_layer(f"poly_{i}", data["CtMult"])
        for i in range(ENCLLM_CFG.gelu_combine):
            p.add_layer(f"combine_{i}", data["CtMult"])
        p.add_layer("last_layer", cfg.zero_layer())
        return p

    # ── Build and solve the full MHA + FFN composed chain ──
    lat_norm_sc = add_shortcut(
        _build_shortcut_norm().solve_latency_only(prune), cfg.boot_lat)
    norm_p = cfg.placer("Norm")
    norm_p.add_layer("mean_var", data["CtMult"])
    norm_p.add_layer("NR_GS", lat_norm_sc)
    norm_p.add_layer("last_layer", cfg.zero_layer())
    lat_norm = norm_p.solve_latency_only(prune)

    lat_sm = _build_softmax_full().solve_latency_only(prune)
    lat_qk = add_shortcut(
        (lambda: (
            p := cfg.placer("QK"),
            p.add_layer("QK_T", data["QK_T"]),
            p.add_layer("Softmax", lat_sm),
            p.add_layer("last_layer", cfg.zero_layer()),
            p.solve_latency_only(prune)
        )[-1])(), cfg.boot_lat)

    lat_gelu = add_shortcut(
        _build_gelu_pw().solve_latency_only(prune), cfg.boot_lat)

    # Build full MHA placer with routes
    mha_p = cfg.placer("MHA")
    mha_p.add_layer("Norm1", lat_norm)
    mha_p.add_layer("QKV_x3", [3 * t for t in data["QKV"]])
    mha_p.add_layer("QK_Softmax", lat_qk)
    mha_p.add_layer("AttnV", data["AttnV"])
    mha_p.add_layer("O_proj", data["QKV"])
    mha_p.add_layer("last_layer", cfg.zero_layer())
    _, mha_routes = mha_p.solve_with_routes(prune)

    # Build full FFN placer with routes
    ffn_p = cfg.placer("FFN")
    ffn_p.add_layer("Norm2", lat_norm)
    ffn_p.add_layer("Up", data["Up"])
    ffn_p.add_layer("GELU", lat_gelu)
    ffn_p.add_layer("Down", data["Down"])
    ffn_p.add_layer("last_layer", cfg.zero_layer())
    _, ffn_routes = ffn_p.solve_with_routes(prune)

    # Build full decoder placer with routes
    lat_mha = add_shortcut(mha_p.solve_latency_only(prune), cfg.boot_lat)
    lat_ffn = add_shortcut(ffn_p.solve_latency_only(prune), cfg.boot_lat)
    dec_p = cfg.placer("Decoder")
    dec_p.add_layer("MHA", lat_mha)
    dec_p.add_layer("FFN", lat_ffn)
    dec_p.add_layer("last_layer", cfg.zero_layer())
    _, dec_routes = dec_p.solve_with_routes(prune)

    # ── Find optimal decoder route at given input level ──
    ml = cfg.max_level
    best_j, best_c = 0, np.inf
    for j in range(ml + 1):
        lat_table, _ = dec_p.solve_with_routes(prune)
        if lat_table[input_lvl][j] < best_c:
            best_c = lat_table[input_lvl][j]
            best_j = j
    dec_route = dec_routes[input_lvl][best_j]

    if not dec_route:
        print(f"  No valid decoder route from level {input_lvl}")
        return 0

    # Extract MHA input level and FFN input level from decoder route
    mha_in = dec_route[0][1]   # (MHA, input_level)
    ffn_in = dec_route[1][1] if len(dec_route) > 1 else best_j  # (FFN, input_level)

    # ── Print ──
    total_btps = 0
    boot_lat = cfg.boot_lat

    def _print_subcircuit(label, placer, routes_table, in_lvl, prefix="  │ "):
        """Print a composed sub-circuit route with bootstrap details."""
        nonlocal total_btps
        # Find best output
        bc, bj = np.inf, 0
        for j in range(ml + 1):
            _, lat_t = placer.solve_with_routes(prune)
            if lat_t[in_lvl] is not None:
                # Use the latency table directly
                pass
        lat_t = placer.solve_latency_only(prune)
        for j in range(ml + 1):
            if lat_t[in_lvl][j] < bc:
                bc = lat_t[in_lvl][j]
                bj = j
        route = routes_table[in_lvl][bj] if routes_table[in_lvl][bj] else []
        btps = _route_bootstraps_summary(route)
        total_btps += len(btps)
        print(f"{prefix}{label}: lvl {in_lvl} → {bj}  ({bc:.4f}s, {len(btps)} btps)")
        for t, name, fl, tl in btps:
            print(f"{prefix}  BTP after {name}: {fl}→{tl}")
        return bj

    def _print_simple(label, depth, lvl, prefix="  │ "):
        new_lvl = lvl - depth
        print(f"{prefix}{label}: lvl {lvl} → {new_lvl}  (depth {depth})")
        return new_lvl

    print(f"\n{'─'*70}")
    print(f"  Decoder internal breakdown (input level {input_lvl})")
    print(f"  Decoder route: MHA@{mha_in} → FFN@{ffn_in} → output@{best_j}")
    print(f"  Decoder latency: {best_c:.4f}s")
    print(f"{'─'*70}")

    # === MHA ===
    print(f"\n  ┌─ MHA (input level {mha_in})")
    mha_route = mha_routes[mha_in]
    # Find best MHA output
    mha_lat = mha_p.solve_latency_only(prune)
    mha_best_j, mha_best_c = 0, np.inf
    for j in range(ml + 1):
        if mha_lat[mha_in][j] < mha_best_c:
            mha_best_c = mha_lat[mha_in][j]
            mha_best_j = j
    mha_route_opt = mha_routes[mha_in][mha_best_j] or []

    print(f"  │ MHA route ({len(mha_route_opt)} steps, {mha_best_c:.4f}s):")
    btps = _route_bootstraps_summary(mha_route_opt)
    total_btps += len(btps)
    for t, (name, lvl) in enumerate(mha_route_opt):
        btp_mark = ""
        if t in {b[0] for b in btps}:
            b = next(b for b in btps if b[0] == t)
            btp_mark = f"  *** BTP {b[2]}→{b[3]}"
        print(f"  │   [{t:2d}] {name:20s} lvl={lvl}{btp_mark}")
    print(f"  │ ({len(btps)} bootstraps in MHA)")
    print(f"  └─ MHA output: level {mha_best_j}")

    # Now expand composed layers within MHA (Norm1, QK_Softmax)
    # Find Norm1 input/output from route
    for name, lvl in mha_route_opt:
        if name == "Norm1":
            norm1_in = lvl
            break
    else:
        norm1_in = mha_in

    # Get NR+GS internal route for Norm1
    nr_gs_p = _build_shortcut_norm()
    nr_gs_lat = nr_gs_p.solve_latency_only(prune)
    _, nr_gs_routes = nr_gs_p.solve_with_routes(prune)

    # Find best NR+GS route from norm1_in
    nr_gs_best_j, nr_gs_best_c = 0, np.inf
    for j in range(ml + 1):
        if nr_gs_lat[norm1_in][j] < nr_gs_best_c:
            nr_gs_best_c = nr_gs_lat[norm1_in][j]
            nr_gs_best_j = j
    nr_gs_route = nr_gs_routes[norm1_in][nr_gs_best_j] or []
    nr_gs_btps = _route_bootstraps_summary(nr_gs_route)

    print(f"\n  Norm1 NR+GS detail (lvl {norm1_in}→{nr_gs_best_j}, {len(nr_gs_btps)} btps):")
    # Group by phase (NR vs GS) for compact output
    nr_btp_iters = set()
    gs_btp_iters = set()
    for _, name, fl, tl in nr_gs_btps:
        if name.startswith("NR_"):
            parts = name.split("_")
            nr_btp_iters.add(int(parts[1]))
        elif name.startswith("GS_"):
            parts = name.split("_")
            gs_btp_iters.add(int(parts[1]))

    nr_btps_per_iter = {}
    gs_btps_per_iter = {}
    for _, name, fl, tl in nr_gs_btps:
        if name.startswith("NR_"):
            it = int(name.split("_")[1])
            nr_btps_per_iter.setdefault(it, []).append(f"{fl}→{tl}")
        elif name.startswith("GS_"):
            it = int(name.split("_")[1])
            gs_btps_per_iter.setdefault(it, []).append(f"{fl}→{tl}")

    print(f"    NR phase ({ENCLLM_CFG.nr_iters} iters, 3 mults/iter):")
    for it in range(ENCLLM_CFG.nr_iters):
        if it in nr_btps_per_iter:
            transitions = ", ".join(nr_btps_per_iter[it])
            print(f"      iter {it:2d}: BTP [{transitions}]")
    print(f"    GS phase ({ENCLLM_CFG.gs_iters} iters, 3 mults/iter):")
    for it in range(ENCLLM_CFG.gs_iters):
        if it in gs_btps_per_iter:
            transitions = ", ".join(gs_btps_per_iter[it])
            print(f"      iter {it:2d}: BTP [{transitions}]")
    print(f"    Total NR+GS bootstraps: {len(nr_gs_btps)}")
    total_btps += len(nr_gs_btps)

    # === FFN ===
    print(f"\n  ┌─ FFN (input level {ffn_in})")
    ffn_lat = ffn_p.solve_latency_only(prune)
    ffn_best_j, ffn_best_c = 0, np.inf
    for j in range(ml + 1):
        if ffn_lat[ffn_in][j] < ffn_best_c:
            ffn_best_c = ffn_lat[ffn_in][j]
            ffn_best_j = j
    ffn_route_opt = ffn_routes[ffn_in][ffn_best_j] or []
    btps = _route_bootstraps_summary(ffn_route_opt)
    total_btps += len(btps)
    for t, (name, lvl) in enumerate(ffn_route_opt):
        btp_mark = ""
        if t in {b[0] for b in btps}:
            b = next(b for b in btps if b[0] == t)
            btp_mark = f"  *** BTP {b[2]}→{b[3]}"
        print(f"  │   [{t:2d}] {name:20s} lvl={lvl}{btp_mark}")
    print(f"  │ ({len(btps)} bootstraps in FFN)")
    print(f"  └─ FFN output: level {ffn_best_j}")

    # GELU internal detail
    gelu_pw_p = _build_gelu_pw()
    _, gelu_routes = gelu_pw_p.solve_with_routes(prune)
    # Find GELU input from FFN route
    gelu_in = None
    for name, lvl in ffn_route_opt:
        if name == "GELU":
            gelu_in = lvl
            break
    if gelu_in is not None:
        gelu_lat = gelu_pw_p.solve_latency_only(prune)
        gelu_best_j, gelu_best_c = 0, np.inf
        for j in range(ml + 1):
            if gelu_lat[gelu_in][j] < gelu_best_c:
                gelu_best_c = gelu_lat[gelu_in][j]
                gelu_best_j = j
        gelu_route = gelu_routes[gelu_in][gelu_best_j] or []
        gelu_btps = _route_bootstraps_summary(gelu_route)
        total_btps += len(gelu_btps)

        print(f"\n  GELU piecewise detail (lvl {gelu_in}→{gelu_best_j}, {len(gelu_btps)} btps):")
        lt_btps = {0: [], 1: [], 2: []}
        poly_btps = []
        combine_btps = []
        for _, name, fl, tl in gelu_btps:
            if name.startswith("lt"):
                idx = int(name[2])
                lt_btps[idx].append(f"{fl}→{tl}")
            elif name.startswith("poly"):
                poly_btps.append(f"{fl}→{tl}")
            elif name.startswith("combine"):
                combine_btps.append(f"{fl}→{tl}")
        for idx in range(3):
            if lt_btps[idx]:
                print(f"    lt_function[{idx}]: BTP [{', '.join(lt_btps[idx])}]")
            else:
                print(f"    lt_function[{idx}]: no BTP")
        if poly_btps:
            print(f"    poly_eval: BTP [{', '.join(poly_btps)}]")
        if combine_btps:
            print(f"    combine: BTP [{', '.join(combine_btps)}]")

    # Norm2 detail (same solver as Norm1 but different input level)
    for name, lvl in ffn_route_opt:
        if name == "Norm2":
            norm2_in = lvl
            break
    else:
        norm2_in = ffn_in
    nr_gs_best_j2, nr_gs_best_c2 = 0, np.inf
    for j in range(ml + 1):
        if nr_gs_lat[norm2_in][j] < nr_gs_best_c2:
            nr_gs_best_c2 = nr_gs_lat[norm2_in][j]
            nr_gs_best_j2 = j
    nr_gs_route2 = nr_gs_routes[norm2_in][nr_gs_best_j2] or []
    nr_gs_btps2 = _route_bootstraps_summary(nr_gs_route2)

    print(f"\n  Norm2 NR+GS detail (lvl {norm2_in}→{nr_gs_best_j2}, {len(nr_gs_btps2)} btps):")
    nr_btps2 = {}
    gs_btps2 = {}
    for _, name, fl, tl in nr_gs_btps2:
        if name.startswith("NR_"):
            it = int(name.split("_")[1])
            nr_btps2.setdefault(it, []).append(f"{fl}→{tl}")
        elif name.startswith("GS_"):
            it = int(name.split("_")[1])
            gs_btps2.setdefault(it, []).append(f"{fl}→{tl}")
    print(f"    NR phase ({ENCLLM_CFG.nr_iters} iters):")
    for it in range(ENCLLM_CFG.nr_iters):
        if it in nr_btps2:
            print(f"      iter {it:2d}: BTP [{', '.join(nr_btps2[it])}]")
    print(f"    GS phase ({ENCLLM_CFG.gs_iters} iters):")
    for it in range(ENCLLM_CFG.gs_iters):
        if it in gs_btps2:
            print(f"      iter {it:2d}: BTP [{', '.join(gs_btps2[it])}]")
    print(f"    Total NR+GS bootstraps: {len(nr_gs_btps2)}")
    total_btps += len(nr_gs_btps2)

    # ── VRAM estimation ──
    # At logN=N, ring dimension = 2^N, each coefficient ≈ 8 bytes (64-bit)
    # Ciphertext at level L has (L+1) limbs in the modulus chain
    # Size ≈ 2 * ring_dim * (level + btp_extra + 1) * 8 bytes (2 polynomials per ct)
    ring_dim = 2 ** logN
    total_depth = cfg.max_level + 9  # btp_extra ≈ 9
    ct_size_at_max = 2 * ring_dim * (total_depth + 1) * 8  # bytes at max level
    key_size = ct_size_at_max * 2  # Galois key ≈ 2× ciphertext size

    # Number of distinct rotation keys needed in bench_mode
    bench_keys = 12  # ~12 keys with placeholder index 5
    # Real rotation keys: estimate from compute_gpt2_rot_indices
    real_keys_est = 200  # ~200 distinct rotation indices

    print(f"\n{'─'*70}")
    print(f"  Summary")
    print(f"{'─'*70}")
    print(f"  Internal bootstraps per decoder: {total_btps}")
    print(f"  Total for {GPT2_N_LAYERS} layers: ~{total_btps * GPT2_N_LAYERS}")
    print(f"  Bootstrap time per decoder (est): ~{total_btps * boot_lat:.2f}s")
    print(f"  Decoder latency (incl. bootstraps): ~{best_c:.2f}s")
    print(f"")
    print(f"  VRAM estimate (ring_dim={ring_dim}, depth={total_depth}):")
    print(f"    Ciphertext at max level: ~{ct_size_at_max / 1e6:.1f} MB")
    print(f"    Galois key (per index): ~{key_size / 1e6:.1f} MB")
    print(f"    Bench mode ({bench_keys} keys): ~{bench_keys * key_size / 1e9:.2f} GB")
    print(f"    Real mode (~{real_keys_est} keys): ~{real_keys_est * key_size / 1e9:.1f} GB")
    print(f"")
    print(f"  NOTE: Ciphertexts at the same level share the same modulus chain,")
    print(f"  so linear ops at a fixed level can reuse Galois keys. The optimizer")
    print(f"  currently minimizes latency only — VRAM-aware scheduling is a TODO.")
    print(f"{'─'*70}")

    return total_btps


# ── Bootstrap extraction ─────────────────────────────────────────────────

def extract_bootstraps(route):
    """Given a route [(layer_name, level), ...], return positions where
    a bootstrap was inserted.  A bootstrap happens when the level at
    step t is lower (= fresher) than the level consumed by step t-1."""
    bootstraps = []
    for t in range(1, len(route)):
        prev_name, prev_lvl = route[t - 1]
        cur_name, cur_lvl = route[t]
        if cur_lvl < prev_lvl:
            bootstraps.append({
                "after": prev_name,
                "before": cur_name,
                "from_level": prev_lvl,
                "to_level": cur_lvl,
                "position": t,
            })
    return bootstraps


def find_optimal_route(output, routes, max_level):
    """Find the (input, output) level pair with minimum latency.
    Returns (min_latency, best_i, best_j, route, bootstraps)."""
    best_lat = np.inf
    best_i, best_j = 0, 0
    for i in range(max_level + 1):
        for j in range(max_level + 1):
            if output[i][j] < best_lat:
                best_lat = output[i][j]
                best_i, best_j = i, j
    route = routes[best_i][best_j]
    if route is None:
        return best_lat, best_i, best_j, [], []
    return best_lat, best_i, best_j, route, extract_bootstraps(route)


def print_placement(output, routes, max_level, model_name):
    """Pretty-print the optimal bootstrap placement."""
    best_lat, bi, bj, route, bootstraps = find_optimal_route(
        output, routes, max_level
    )
    print(f"\n{'='*70}")
    print(f"  {model_name} — Optimal Bootstrap Placement")
    print(f"{'='*70}")
    print(f"  End-to-end latency : {best_lat:.2f} s")
    print(f"  Input level        : {bi}")
    print(f"  Output level       : {bj}")
    print(f"  Route ({len(route)} steps):")
    btp_positions = {b["position"] for b in bootstraps}
    for t, (name, lvl) in enumerate(route):
        marker = ""
        if t in btp_positions:
            b = next(b for b in bootstraps if b["position"] == t)
            marker = (f"  <-- BOOTSTRAP "
                      f"(level {b['from_level']} -> {b['to_level']})")
        print(f"    [{t:3d}] {name:20s}  level={lvl}{marker}")
    print(f"\n  Bootstraps needed: {len(bootstraps)}")
    for b in bootstraps:
        print(f"    after {b['after']:20s}  before {b['before']:20s}"
              f"  (level {b['from_level']} -> {b['to_level']})")
    print(f"{'='*70}\n")
    return best_lat, route, bootstraps


def extract_decoder_config(cfg, data, prune, input_lvl):
    """Extract per-decoder C++ config values from the solver routes.

    Returns a dict mapping GPT2LayerConfig field names to optimal values,
    plus annotations showing WHERE each BTP fires in the computation pipeline.

    The C++ decoder has 4 bootstrap_to() calls (conditional — only fires if
    current depth < target):
      1. before Norm1       → norm1_btp_level
      2. after QK^T         → attn_btp_level
      3. before Norm2       → norm2_btp_level
      4. after Up, before GELU → gelu_btp_level

    Plus internal DepthGuard BTPs within Norm/Softmax/GELU sub-circuits:
      - norm1_target_level / norm2_target_level control the DepthGuard
        refresh level inside Newton-Raphson & Goldschmidt iterations.
    """
    ml = cfg.max_level

    # ── Build sub-circuit latency tables (same as explain_gpt2_decoder) ──
    def _build_shortcut_norm():
        p = cfg.placer("Shortcut_norm")
        if "SqrtNt" in data and "SqrtGold" in data:
            for i in range(ENCLLM_CFG.nr_iters):
                p.add_layer(f"NR_{i}_mult1.5", data["CtMult"])
                p.add_layer(f"NR_{i}_square", data["CtMult"])
                p.add_layer(f"NR_{i}_mult", data["CtMult"])
            for i in range(ENCLLM_CFG.gs_iters):
                p.add_layer(f"GS_{i}_mult_sqrt", data["CtMult"])
                p.add_layer(f"GS_{i}_mult_half", data["CtMult"])
                p.add_layer(f"GS_{i}_mult_res", data["CtMult"])
        p.add_layer("final_mult", data["CtMult"])
        p.add_layer("last_layer", cfg.zero_layer())
        return p

    def _build_softmax_full():
        p = cfg.placer("Softmax_full")
        lat_gs = add_shortcut(
            _build_shortcut_softmax().solve_latency_only(prune), cfg.boot_lat)
        for i in range(8):
            p.add_layer(f"max_cmp_{i}", data["CtMult"])
        p.add_layer("exp_mult", data["CtMult"])
        for i in range(ENCLLM_CFG.exp_r):
            p.add_layer(f"exp_sq_{i}", data["CtMult"])
        p.add_layer("GS_inverse", lat_gs)
        p.add_layer("last_layer", cfg.zero_layer())
        return p

    def _build_shortcut_softmax():
        p = cfg.placer("Shortcut_softmax")
        if "Softmax" in data:
            p.add_layer("GoldIter", data["Softmax"])
        else:
            for i in range(ENCLLM_CFG.gs_inv_iters):
                p.add_layer(f"GS_inv_{i}", data["CtMult"])
                p.add_layer(f"GS_inv_{i}_sq", data["CtMult"])
            p.add_layer("mult", data["CtMult"])
        p.add_layer("last_layer", cfg.zero_layer())
        return p

    def _build_gelu_pw():
        p = cfg.placer("GELU_pw")
        sign_d = ENCLLM_CFG.sign_depth + ENCLLM_CFG.lt_extra
        for lt_idx in range(3):
            for s in range(sign_d):
                p.add_layer(f"lt{lt_idx}_sign_{s}", data["CtMult"])
        for i in range(ENCLLM_CFG.gelu_poly_depth):
            p.add_layer(f"poly_{i}", data["CtMult"])
        for i in range(ENCLLM_CFG.gelu_combine):
            p.add_layer(f"combine_{i}", data["CtMult"])
        p.add_layer("last_layer", cfg.zero_layer())
        return p

    lat_norm_sc = add_shortcut(
        _build_shortcut_norm().solve_latency_only(prune), cfg.boot_lat)
    norm_p = cfg.placer("Norm")
    norm_p.add_layer("mean_var", data["CtMult"])
    norm_p.add_layer("NR_GS", lat_norm_sc)
    norm_p.add_layer("last_layer", cfg.zero_layer())
    lat_norm = norm_p.solve_latency_only(prune)

    lat_sm = _build_softmax_full().solve_latency_only(prune)
    lat_qk = add_shortcut(
        (lambda: (
            p := cfg.placer("QK"),
            p.add_layer("QK_T", data["QK_T"]),
            p.add_layer("Softmax", lat_sm),
            p.add_layer("last_layer", cfg.zero_layer()),
            p.solve_latency_only(prune)
        )[-1])(), cfg.boot_lat)

    lat_gelu = add_shortcut(
        _build_gelu_pw().solve_latency_only(prune), cfg.boot_lat)

    # ── Solve MHA and FFN with routes ──
    mha_p = cfg.placer("MHA")
    mha_p.add_layer("Norm1", lat_norm)
    mha_p.add_layer("QKV_x3", [3 * t for t in data["QKV"]])
    mha_p.add_layer("QK_Softmax", lat_qk)
    mha_p.add_layer("AttnV", data["AttnV"])
    mha_p.add_layer("O_proj", data["QKV"])
    mha_p.add_layer("last_layer", cfg.zero_layer())
    mha_lat = mha_p.solve_latency_only(prune)
    _, mha_routes = mha_p.solve_with_routes(prune)

    ffn_p = cfg.placer("FFN")
    ffn_p.add_layer("Norm2", lat_norm)
    ffn_p.add_layer("Up", data["Up"])
    ffn_p.add_layer("GELU", lat_gelu)
    ffn_p.add_layer("Down", data["Down"])
    ffn_p.add_layer("last_layer", cfg.zero_layer())
    ffn_lat = ffn_p.solve_latency_only(prune)
    _, ffn_routes = ffn_p.solve_with_routes(prune)

    # ── Compose decoder and find optimal route ──
    lat_mha = add_shortcut(mha_lat, cfg.boot_lat)
    lat_ffn = add_shortcut(ffn_lat, cfg.boot_lat)
    dec_p = cfg.placer("Decoder")
    dec_p.add_layer("MHA", lat_mha)
    dec_p.add_layer("FFN", lat_ffn)
    dec_p.add_layer("last_layer", cfg.zero_layer())
    _, dec_routes = dec_p.solve_with_routes(prune)

    best_j, best_c = 0, np.inf
    dec_lat = dec_p.solve_latency_only(prune)
    for j in range(ml + 1):
        if dec_lat[input_lvl][j] < best_c:
            best_c = dec_lat[input_lvl][j]
            best_j = j
    dec_route = dec_routes[input_lvl][best_j] or []

    mha_in = dec_route[0][1] if dec_route else input_lvl
    ffn_in = dec_route[1][1] if len(dec_route) > 1 else best_j

    # ── Extract MHA route levels ──
    mha_best_j, mha_best_c = 0, np.inf
    for j in range(ml + 1):
        if mha_lat[mha_in][j] < mha_best_c:
            mha_best_c = mha_lat[mha_in][j]
            mha_best_j = j
    mha_route = mha_routes[mha_in][mha_best_j] or []

    # ── Extract FFN route levels ──
    ffn_best_j, ffn_best_c = 0, np.inf
    for j in range(ml + 1):
        if ffn_lat[ffn_in][j] < ffn_best_c:
            ffn_best_c = ffn_lat[ffn_in][j]
            ffn_best_j = j
    ffn_route = ffn_routes[ffn_in][ffn_best_j] or []

    # ── Map route levels to C++ config fields ──
    # Route format: [(name, input_level), ...]
    # The input_level is the remaining depth when entering that sub-circuit.
    mha_levels = {name: lvl for name, lvl in mha_route}
    ffn_levels = {name: lvl for name, lvl in ffn_route}

    # norm1_btp_level: bootstrap target before Norm1 = max level
    #   (bootstrap_to is now conditional — only fires if needed)
    norm1_btp = ml
    # attn_btp_level: bootstrap target before Softmax.
    #   In optimizer: QK_Softmax is composed (QK_T + Softmax).
    #   In C++: BTP fires AFTER QK^T, targeting this level for Softmax.
    #
    # Softmax minimum depth: exp_approx(r=7) consumes exp_r+1 = 8 CKKS levels,
    # then the final EvalMult(exp_ct, inv_sum) needs 1 more → min remaining = 9.
    # The optimizer may underestimate this because its Softmax model accounts for
    # internal BTPs that C++ SoftmaxConfig::btp_min_remaining handles autonomously.
    # Clamp attn_btp ≥ kSoftmaxMinRemaining = exp_r + 2 = 9 so the C++ code
    # always enters Softmax with at least that many remaining levels.
    kSoftmaxMinRemaining = 9   # exp_r(7) + 1 (EvalMult) + 1 (safety)
    attn_btp = max(mha_levels.get("QK_Softmax", ml), kSoftmaxMinRemaining)
    # attn_v_btp_level: bootstrap target before AttnV (0 = skip)
    #   Set to max_level if the optimizer route shows a BTP before AttnV.
    #   bootstrap_to() is conditional so this only fires when depth is low.
    attn_v_btp = 0
    for t in range(1, len(mha_route)):
        if mha_route[t][0] == "AttnV" and mha_route[t][1] < mha_route[t-1][1]:
            attn_v_btp = ml
            break
    # norm2_btp_level: bootstrap target before Norm2
    norm2_btp = ml
    # gelu_btp_level: bootstrap target before GELU.
    # The optimizer's piecewise_iter GELU model assumes sign() can be bootstrapped
    # mid-evaluation (sign_depth=4 model-units per composition; 22 internal BTPs).
    # The C++ sign() has NO internal bootstraps: it runs F4∘F4∘G4∘G4 straight through,
    # consuming 4 compositions × ceil(log2(9))=4 CKKS levels = 16 levels.
    # lt_function adds 2 more; GELU needs 1 final EvalMult → min remaining = 19.
    #
    # With max_level=ml and BTP to target=T: remaining_after_btp = min(ml, T).
    # For target=ml: remaining = ml (no level drop, fresh bootstrap output).
    # Constraint: ml ≥ 19.  We enforce gelu_btp ≥ ml so C++ always gets remaining=ml.
    gelu_btp = max(ffn_levels.get("GELU", ml), ml)
    # down_btp_level: bootstrap target before Down (0 = skip)
    down_btp = 0
    for t in range(1, len(ffn_route)):
        if ffn_route[t][0] == "Down" and ffn_route[t][1] < ffn_route[t-1][1]:
            down_btp = ml
            break
    # norm target levels for internal DepthGuard BTPs: use max for fewest BTPs
    norm_target = ml

    # Count internal BTPs from NR+GS routes
    nr_gs_p = _build_shortcut_norm()
    nr_gs_lat = nr_gs_p.solve_latency_only(prune)
    _, nr_gs_routes = nr_gs_p.solve_with_routes(prune)

    def _count_btps(route):
        n = 0
        for t in range(1, len(route)):
            if route[t][1] < route[t-1][1]:
                n += 1
        return n

    # Internal BTPs within one Norm (NR+GS)
    norm_in = mha_levels.get("Norm1", mha_in)
    nr_best_j, nr_best_c = 0, np.inf
    for j in range(ml + 1):
        if nr_gs_lat[norm_in][j] < nr_best_c:
            nr_best_c = nr_gs_lat[norm_in][j]
            nr_best_j = j
    nr_gs_route = nr_gs_routes[norm_in][nr_best_j] or []
    nr_gs_btps = _count_btps(nr_gs_route)

    # Internal BTPs within GELU
    gelu_pw_p = _build_gelu_pw()
    gelu_pw_lat = gelu_pw_p.solve_latency_only(prune)
    _, gelu_routes = gelu_pw_p.solve_with_routes(prune)
    gelu_in = ffn_levels.get("GELU", ml)
    gelu_best_j = 0
    gelu_best_c = np.inf
    for j in range(ml + 1):
        if gelu_pw_lat[gelu_in][j] < gelu_best_c:
            gelu_best_c = gelu_pw_lat[gelu_in][j]
            gelu_best_j = j
    gelu_route = gelu_routes[gelu_in][gelu_best_j] or []
    gelu_btps = _count_btps(gelu_route)

    # MHA inter-block BTPs (between Norm1→QKV→QK_Softmax→AttnV→O_proj)
    mha_inter_btps = _count_btps(mha_route)
    # FFN inter-block BTPs
    ffn_inter_btps = _count_btps(ffn_route)

    result = {
        "norm1_btp_level": int(norm1_btp),
        "norm1_target_level": int(norm_target),
        # cache_btp_level: always BTP k,v before cache update so vCache stays at
        # a low consumed level. Without this, FLEXIBLEAUTO auto-aligns the BTP
        # output (level ~19) to the cache (at QKV level ~30), consuming 11+ extra
        # levels in attn_v and exhausting the modulus chain.
        "cache_btp_level": int(ml),
        "attn_btp_level": int(attn_btp),
        "attn_v_btp_level": int(attn_v_btp),
        "norm2_btp_level": int(norm2_btp),
        "norm2_target_level": int(norm_target),
        "gelu_btp_level": int(gelu_btp),
        "down_btp_level": int(down_btp),
        "decoder_latency": float(best_c),
        "annotations": {
            "mha_route": [
                {"op": name, "input_level": int(lvl),
                 "note": "BTP before this op" if t > 0 and lvl < mha_route[t-1][1] else ""}
                for t, (name, lvl) in enumerate(mha_route)
            ],
            "ffn_route": [
                {"op": name, "input_level": int(lvl),
                 "note": "BTP before this op" if t > 0 and lvl < ffn_route[t-1][1] else ""}
                for t, (name, lvl) in enumerate(ffn_route)
            ],
            "norm_internal_btps": int(nr_gs_btps),
            "gelu_internal_btps": int(gelu_btps),
            "mha_inter_btps": int(mha_inter_btps),
            "ffn_inter_btps": int(ffn_inter_btps),
            "total_btps_per_decoder": int(
                mha_inter_btps + ffn_inter_btps + 2 * nr_gs_btps + gelu_btps
            ),
        },
    }
    return result


def export_placement_json(output, routes, max_level, model_name, out_path,
                          decoder_config=None, logN=16):
    """Export optimal bootstrap placement as JSON for config generation."""
    best_lat, bi, bj, route, bootstraps = find_optimal_route(
        output, routes, max_level
    )
    result = {
        "model": model_name,
        "logN": logN,
        "max_level": max_level,
        "end_to_end_latency": float(best_lat),
        "input_level": int(bi),
        "output_level": int(bj),
        "route": [{"name": name, "level": int(lvl)} for name, lvl in route],
        "bootstraps": [
            {
                "after": b["after"],
                "before": b["before"],
                "from_level": int(b["from_level"]),
                "to_level": int(b["to_level"]),
                "position": int(b["position"]),
            }
            for b in bootstraps
        ],
    }
    if decoder_config:
        result["decoder_config"] = decoder_config
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  JSON placement written to {out_path}")
    return result


# ── Model registry ──────────────────────────────────────────────────────

MODELS = {
    "llama": solve_model_llama,
    "gpt2":  solve_model_gpt2,
}


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap placement optimizer for FHE transformer inference"
    )
    parser.add_argument("--file", type=str, default="./data.csv")
    parser.add_argument("--prune", type=int, default=1)
    parser.add_argument("--model", type=str, default="llama",
                        choices=list(MODELS.keys()))
    parser.add_argument("--boot-lat", type=float, default=56.08,
                        help="Bootstrap latency in seconds")
    parser.add_argument("--max-level", type=int, default=16,
                        help="Maximum CKKS level budget")
    parser.add_argument("--logN", type=int, default=16,
                        help="Ring dimension log2 (e.g. 12 or 16)")
    parser.add_argument("--json", type=str, default="",
                        help="Path to write JSON placement (default: <file>_placement.json)")
    parser.add_argument("--uniform", action="store_true",
                        help=(
                            "Force uniform BTP at the start of every decoder block. "
                            "All layers use the same config (max_level input, no "
                            "inter-decoder optimization). Predictable depth pattern: "
                            "BTP → Norm1 → QKV → cache → Softmax → AttnV → Norm2 → "
                            "Up → GELU → Down, repeated for each layer."
                        ))
    args = parser.parse_args()

    cfg = CKKSConfig(boot_lat=args.boot_lat, max_level=args.max_level)
    data = read_data(args.file)
    json_path = args.json if args.json else args.file.replace(".csv", "_placement.json")

    if args.uniform and args.model == "gpt2":
        # ── Uniform mode: every decoder starts fresh at max_level ──────────
        # Skip inter-decoder optimization. Force BTP at every layer boundary.
        # Decoder config is computed once assuming input_level = max_level.
        print("Solving GPT-2 Model (UNIFORM mode — BTP at every block)...\n")
        dec_cfg = extract_decoder_config(cfg, data, args.prune, cfg.max_level)
        dec_lat  = dec_cfg["decoder_latency"]
        n_layers = GPT2_N_LAYERS
        total_lat = n_layers * (cfg.boot_lat + dec_lat)

        print(f"\n{'='*70}")
        print(f"  gpt2 — Uniform Bootstrap Placement (BTP every block)")
        print(f"{'='*70}")
        print(f"  End-to-end latency : {total_lat:.2f} s")
        print(f"  Input level        : {cfg.max_level}")
        print(f"  Strategy           : BTP before EVERY decoder (same config all layers)")
        print(f"  Bootstraps needed  : {n_layers}  (one per layer boundary)")
        print(f"{'='*70}\n")

        explain_gpt2_decoder(cfg, data, args.prune, cfg.max_level, args.logN)

        # Build synthetic route / bootstraps for JSON export
        route = [(f"decoder_{i}", cfg.max_level) for i in range(n_layers)]
        bootstraps_list = [
            {
                "after":      f"decoder_{i}",
                "before":     f"decoder_{i+1}",
                "from_level": cfg.max_level,
                "to_level":   cfg.max_level,
                "position":   i + 1,
            }
            for i in range(n_layers - 1)
        ]
        result = {
            "model": args.model,
            "mode": "uniform",
            "logN": args.logN,
            "max_level": cfg.max_level,
            "end_to_end_latency": total_lat,
            "input_level": cfg.max_level,
            "output_level": cfg.max_level,
            "route": [{"name": n, "level": l} for n, l in route],
            "bootstraps": bootstraps_list,
            "decoder_config": dec_cfg,
        }
        import json as _json
        with open(json_path, "w") as f:
            _json.dump(result, f, indent=2)
        print(f"  JSON placement written to {json_path}")
        return

    output, routes = MODELS[args.model](cfg, data, args.prune)
    best_lat, route, bootstraps = print_placement(output, routes, cfg.max_level, args.model)

    # Detailed per-decoder breakdown (GPT-2 only)
    dec_cfg = None
    if args.model == "gpt2" and route:
        # Use the input level of the most common decoder (skip first if different)
        decoder_input_lvl = route[1][1] if len(route) > 1 else route[0][1]
        explain_gpt2_decoder(cfg, data, args.prune, decoder_input_lvl, args.logN)
        dec_cfg = extract_decoder_config(cfg, data, args.prune, decoder_input_lvl)

    # Always emit JSON
    export_placement_json(output, routes, cfg.max_level, args.model, json_path,
                          decoder_config=dec_cfg, logN=args.logN)


if __name__ == "__main__":
    begin = time.time()
    main()
    end = time.time()
    print("Solver time:", (end - begin) * 1000, "ms")
