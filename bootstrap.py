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


# ── Shared sub-circuits (model-agnostic) ─────────────────────────────────

def solve_shortcut_softmax(cfg, data, prune):
    p = cfg.placer("Shortcut_softmax")
    p.add_layer("GoldIter", data["Softmax"])
    p.add_layer("mult", data["CtMult"])
    p.add_layer("last_layer", cfg.zero_layer())
    return p.solve_latency_only(prune)


def solve_softmax(cfg, data, prune):
    p = cfg.placer("Softmax")
    lat_sc = add_shortcut(solve_shortcut_softmax(cfg, data, prune), cfg.boot_lat)
    for i in range(8):
        p.add_layer(f"mult_{i}", data["CtMult"])
    p.add_layer("shortcut", lat_sc)
    p.add_layer("last_layer", cfg.zero_layer())
    return p.solve_latency_only(prune)


def solve_shortcut_norm(cfg, data, prune):
    p = cfg.placer("Shortcut_norm")
    p.add_layer("Newton", data["SqrtNt"])
    p.add_layer("GoldIter", data["SqrtGold"])
    p.add_layer("mult", data["CtMult"])
    p.add_layer("last_layer", cfg.zero_layer())
    return p.solve_latency_only(prune)


def solve_norm(cfg, data, prune):
    p = cfg.placer("Norm")
    lat_sc = add_shortcut(solve_shortcut_norm(cfg, data, prune), cfg.boot_lat)
    p.add_layer("mult", data["CtMult"])
    p.add_layer("shortcut", lat_sc)
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
    """GELU via piecewise polynomial (3 sign evals + 2 poly evals)."""
    p = cfg.placer("GELU_pw")
    p.add_layer("GELU", data["GELU"])
    p.add_layer("last_layer", cfg.zero_layer())
    return p.solve_latency_only(prune)


# When you profile a new approximation (e.g. GELU via single Chebyshev-255),
# add a row "GELU_cheb" to your data.csv and register the solver here:
#
#   def solve_gelu_chebyshev(cfg, data, prune):
#       p = cfg.placer("GELU_cheb")
#       p.add_layer("GELU_cheb", data["GELU_cheb"])
#       p.add_layer("last_layer", cfg.zero_layer())
#       return p.solve_latency_only(prune)

# Registry: maps activation name -> list of (variant_name, solver_fn).
# best_variant() will try all and take the element-wise minimum.
ACTIVATION_VARIANTS = {
    "SiLU": [
        ("chebyshev", solve_silu_chebyshev),
        # ("rational",  solve_silu_rational),   # add when profiled
    ],
    "GELU": [
        ("piecewise", solve_gelu_piecewise),
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
    args = parser.parse_args()

    cfg = CKKSConfig(boot_lat=args.boot_lat, max_level=args.max_level)
    data = read_data(args.file)
    output, routes = MODELS[args.model](cfg, data, args.prune)
    print_placement(output, routes, cfg.max_level, args.model)


if __name__ == "__main__":
    begin = time.time()
    main()
    end = time.time()
    print("Solver time:", (end - begin) * 1000, "ms")
