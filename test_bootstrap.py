"""
Test the refactored bootstrap placement optimizer.

Creates synthetic latency data and runs both LLaMA and GPT-2 solvers,
verifying they produce finite solutions and the caching / variant
selection works correctly.
"""

import numpy as np
import tempfile
import os
import sys

# ── Synthetic data generation ────────────────────────────────────────────

def make_row(name, max_level, base_lat, depth=1):
    """Create a latency row: levels below `depth` are 0 (= inf), rest = base_lat.
    In practice latency decreases at higher levels (less work), so we add
    a small gradient."""
    row = [0.0] * (max_level + 1)
    for i in range(depth, max_level + 1):
        row[i] = base_lat + (max_level - i) * 0.5  # slightly cheaper at higher levels
    return [name] + row


def write_tsv(path, rows, max_level):
    header = ["Op"] + [str(i) for i in range(max_level + 1)]
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(str(x) for x in row) + "\n")


def make_llama_data(path, max_level=16):
    rows = [
        make_row("CtMult",   max_level, 1.0, depth=1),
        make_row("Softmax",  max_level, 3.0, depth=2),
        make_row("SqrtNt",   max_level, 2.0, depth=2),
        make_row("SqrtGold", max_level, 1.5, depth=1),
        make_row("RoPE",     max_level, 0.5, depth=1),
        make_row("Cache",    max_level, 0.8, depth=1),
        make_row("QK_T",     max_level, 2.0, depth=1),
        make_row("QKV",      max_level, 4.0, depth=1),
        make_row("AttnV",    max_level, 3.0, depth=1),
        make_row("SiLU",     max_level, 5.0, depth=3),
        make_row("UpGate",   max_level, 4.0, depth=1),
        make_row("Down",     max_level, 4.0, depth=1),
    ]
    write_tsv(path, rows, max_level)


def make_gpt2_data(path, max_level=16):
    rows = [
        make_row("CtMult",   max_level, 1.0, depth=1),
        make_row("Softmax",  max_level, 3.0, depth=2),
        make_row("SqrtNt",   max_level, 2.0, depth=2),
        make_row("SqrtGold", max_level, 1.5, depth=1),
        make_row("QK_T",     max_level, 2.0, depth=1),
        make_row("QKV",      max_level, 4.0, depth=1),
        make_row("AttnV",    max_level, 3.0, depth=1),
        make_row("GELU",     max_level, 8.0, depth=4),  # more expensive than SiLU
        make_row("Up",       max_level, 4.0, depth=1),
        make_row("Down",     max_level, 4.0, depth=1),
    ]
    write_tsv(path, rows, max_level)


# ── Tests ────────────────────────────────────────────────────────────────

def test_read_data():
    """read_data parses TSV and merges RoPE into Cache."""
    from bootstrap import read_data

    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
        make_llama_data(f.name)
        tmp = f.name
    try:
        data = read_data(tmp)
        assert "Cache" in data, "Cache should exist (merged from RoPE + Cache)"
        assert "RoPE" not in data, "RoPE should be deleted after merge"
        assert "CtMult" in data
        # Values at level 0 should be inf (was 0 in TSV)
        assert data["CtMult"][0] == np.inf
        # Values at valid levels should be finite
        assert np.isfinite(data["CtMult"][2])
        print("  PASS: read_data")
    finally:
        os.unlink(tmp)


def test_llama_solver():
    """LLaMA solver runs and produces finite latency."""
    from bootstrap import CKKSConfig, read_data, solve_model_llama

    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
        make_llama_data(f.name, max_level=8)
        tmp = f.name
    try:
        cfg = CKKSConfig(boot_lat=10.0, max_level=8)
        data = read_data(tmp)
        result, routes = solve_model_llama(cfg, data, prune=1)
        assert len(result) == 9  # max_level+1
        assert len(result[0]) == 9
        finite_count = sum(
            1 for row in result for v in row if np.isfinite(v)
        )
        assert finite_count > 0, "Should have at least one finite path"
        min_lat = min(min(row) for row in result)
        assert np.isfinite(min_lat), f"Global min should be finite, got {min_lat}"
        # Check routes are populated
        has_route = any(
            routes[i][j] is not None
            for i in range(9) for j in range(9)
            if np.isfinite(result[i][j])
        )
        assert has_route, "Should have at least one route"
        print(f"  PASS: llama solver (min_lat={min_lat:.2f})")
    finally:
        os.unlink(tmp)


def test_gpt2_solver():
    """GPT-2 solver runs and produces finite latency."""
    from bootstrap import CKKSConfig, read_data, solve_model_gpt2

    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
        make_gpt2_data(f.name, max_level=8)
        tmp = f.name
    try:
        cfg = CKKSConfig(boot_lat=10.0, max_level=8)
        data = read_data(tmp)
        result, routes = solve_model_gpt2(cfg, data, prune=1)
        assert len(result) == 9
        finite_count = sum(
            1 for row in result for v in row if np.isfinite(v)
        )
        assert finite_count > 0
        min_lat = min(min(row) for row in result)
        assert np.isfinite(min_lat), f"Global min should be finite, got {min_lat}"
        print(f"  PASS: gpt2 solver (min_lat={min_lat:.2f})")
    finally:
        os.unlink(tmp)


def test_gpt2_cheaper_than_llama():
    """GPT-2 (12 layers, no gate/RoPE) should be cheaper than LLaMA (32 layers)
    with comparable per-op latencies."""
    from bootstrap import CKKSConfig, read_data, solve_model_llama, solve_model_gpt2

    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f1, \
         tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f2:
        make_llama_data(f1.name, max_level=8)
        make_gpt2_data(f2.name, max_level=8)
        tmp1, tmp2 = f1.name, f2.name
    try:
        cfg = CKKSConfig(boot_lat=10.0, max_level=8)
        llama_result, _ = solve_model_llama(cfg, read_data(tmp1), prune=1)
        gpt2_result, _ = solve_model_gpt2(cfg, read_data(tmp2), prune=1)

        llama_min = min(min(row) for row in llama_result)
        gpt2_min = min(min(row) for row in gpt2_result)
        assert gpt2_min < llama_min, (
            f"GPT-2 ({gpt2_min:.2f}) should be cheaper than LLaMA ({llama_min:.2f})"
        )
        print(f"  PASS: gpt2 ({gpt2_min:.2f}) < llama ({llama_min:.2f})")
    finally:
        os.unlink(tmp1)
        os.unlink(tmp2)


def test_solver_cache():
    """SolverCache returns same result and doesn't re-solve."""
    from bootstrap import SolverCache

    call_count = 0
    def expensive():
        nonlocal call_count
        call_count += 1
        return [[1.0, 2.0], [3.0, 4.0]]

    cache = SolverCache()
    r1 = cache.get("test", expensive)
    r2 = cache.get("test", expensive)
    assert call_count == 1, f"Should only call once, called {call_count}"
    assert r1 == r2
    # Verify deepcopy (mutations shouldn't leak)
    r1[0][0] = 999.0
    r3 = cache.get("test", expensive)
    assert r3[0][0] == 1.0, "Cache should return fresh deepcopy"
    print("  PASS: SolverCache")


def test_best_variant():
    """best_variant takes element-wise minimum across variant tables."""
    from bootstrap import best_variant, CKKSConfig

    cfg = CKKSConfig(max_level=2)

    def variant_a(cfg, data, prune):
        return [[1.0, 5.0, 9.0],
                [5.0, 1.0, 9.0],
                [9.0, 9.0, 1.0]]

    def variant_b(cfg, data, prune):
        return [[9.0, 9.0, 2.0],
                [9.0, 2.0, 9.0],
                [2.0, 9.0, 9.0]]

    result = best_variant(
        [("A", variant_a), ("B", variant_b)],
        cfg, {}, True
    )
    expected = [[1.0, 5.0, 2.0],
                [5.0, 1.0, 9.0],
                [2.0, 9.0, 1.0]]
    assert result == expected, f"Expected {expected}, got {result}"
    print("  PASS: best_variant")


def test_different_ckks_configs():
    """Different boot_lat / max_level produce different results."""
    from bootstrap import CKKSConfig, read_data, solve_model_gpt2

    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
        make_gpt2_data(f.name, max_level=8)
        tmp = f.name
    try:
        data = read_data(tmp)

        cfg_fast = CKKSConfig(boot_lat=5.0, max_level=8)
        cfg_slow = CKKSConfig(boot_lat=50.0, max_level=8)

        r_fast, _ = solve_model_gpt2(cfg_fast, data, prune=1)
        r_slow, _ = solve_model_gpt2(cfg_slow, data, prune=1)

        min_fast = min(min(row) for row in r_fast)
        min_slow = min(min(row) for row in r_slow)
        assert min_fast < min_slow, (
            f"Faster bootstrap ({min_fast:.2f}) should yield lower latency "
            f"than slower ({min_slow:.2f})"
        )
        print(f"  PASS: boot_lat=5 ({min_fast:.2f}) < boot_lat=50 ({min_slow:.2f})")
    finally:
        os.unlink(tmp)


def test_no_prune():
    """prune=0 (exhaustive search) should also work."""
    from bootstrap import CKKSConfig, read_data, solve_model_gpt2

    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
        # Use small max_level but large enough for GELU depth=4
        make_gpt2_data(f.name, max_level=8)
        tmp = f.name
    try:
        cfg = CKKSConfig(boot_lat=10.0, max_level=8)
        data = read_data(tmp)
        result, _ = solve_model_gpt2(cfg, data, prune=0)
        min_lat = min(min(row) for row in result)
        assert np.isfinite(min_lat)
        print(f"  PASS: no-prune gpt2 (min_lat={min_lat:.2f})")
    finally:
        os.unlink(tmp)


def test_bootstrap_extraction():
    """Routes are returned and bootstrap positions are correctly extracted."""
    from bootstrap import (CKKSConfig, read_data, solve_model_gpt2,
                           find_optimal_route, extract_bootstraps)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
        make_gpt2_data(f.name, max_level=8)
        tmp = f.name
    try:
        cfg = CKKSConfig(boot_lat=10.0, max_level=8)
        data = read_data(tmp)
        output, routes = solve_model_gpt2(cfg, data, prune=1)
        best_lat, bi, bj, route, bootstraps = find_optimal_route(
            output, routes, cfg.max_level
        )
        assert np.isfinite(best_lat)
        assert len(route) > 0, "Route should not be empty"
        # Each route entry is (name, level)
        for name, lvl in route:
            assert isinstance(name, str)
            assert isinstance(lvl, (int, float))
        # Bootstraps should have correct structure
        for b in bootstraps:
            assert b["to_level"] < b["from_level"], (
                f"Bootstrap should go to a fresher level: {b}")
            assert "after" in b and "before" in b
        print(f"  PASS: bootstrap extraction "
              f"({len(bootstraps)} bootstraps in {len(route)} steps)")
    finally:
        os.unlink(tmp)


def test_extract_bootstraps_unit():
    """Unit test for extract_bootstraps on a hand-crafted route."""
    from bootstrap import extract_bootstraps

    route = [
        ("Norm", 0), ("QKV", 1), ("QK_T", 2),
        ("Softmax", 0),  # bootstrap here: level 2 -> 0
        ("AttnV", 1), ("O", 2),
        ("Up", 0),       # bootstrap here: level 2 -> 0
        ("GELU", 4), ("Down", 5),
    ]
    btps = extract_bootstraps(route)
    assert len(btps) == 2
    assert btps[0]["after"] == "QK_T"
    assert btps[0]["before"] == "Softmax"
    assert btps[0]["from_level"] == 2
    assert btps[0]["to_level"] == 0
    assert btps[1]["after"] == "O"
    assert btps[1]["before"] == "Up"
    print("  PASS: extract_bootstraps unit")


if __name__ == "__main__":
    tests = [
        test_read_data,
        test_solver_cache,
        test_best_variant,
        test_extract_bootstraps_unit,
        test_llama_solver,
        test_gpt2_solver,
        test_gpt2_cheaper_than_llama,
        test_bootstrap_extraction,
        test_different_ckks_configs,
        test_no_prune,
    ]
    failed = 0
    for t in tests:
        print(f"\n{t.__name__}:")
        try:
            t()
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"{len(tests) - failed}/{len(tests)} passed, {failed} failed")
    sys.exit(failed)
