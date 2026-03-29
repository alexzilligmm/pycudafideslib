"""Tests for gen_config.py — bootstrap placement → C++ config translation."""

import json
import subprocess
import sys
import tempfile
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN_CONFIG = os.path.join(REPO, "gen_config.py")


def _run(placement, fmt="cpp", num_layers=12):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(placement, f)
        f.flush()
        result = subprocess.run(
            [sys.executable, GEN_CONFIG, f.name, "--format", fmt,
             "--num-layers", str(num_layers)],
            capture_output=True, text=True,
        )
    os.unlink(f.name)
    assert result.returncode == 0, f"gen_config.py failed:\n{result.stderr}"
    return result.stdout


def make_pruned_placement():
    """Pruned route: composed decoder layers with inter-layer bootstraps."""
    return {
        "model": "gpt2",
        "max_level": 16,
        "end_to_end_latency": 1200.0,
        "input_level": 0,
        "output_level": 3,
        "route": [
            {"name": f"decoder_{i}", "level": 0 if i % 2 == 0 else 3}
            for i in range(12)
        ],
        "bootstraps": [
            {
                "after": f"decoder_{i}",
                "before": f"decoder_{i+1}",
                "from_level": 0,
                "to_level": 3,
                "position": i + 1,
            }
            for i in range(0, 11, 2)
        ],
    }


def make_flat_placement():
    """Flat route: individual ops visible (prune=0 style)."""
    return {
        "model": "gpt2",
        "max_level": 16,
        "end_to_end_latency": 100.0,
        "input_level": 0,
        "output_level": 5,
        "decoder_config": {
            "norm1_btp_level": 16,
            "norm1_target_level": 16,
            "cache_btp_level": 16,
            "attn_btp_level": 9,
            "attn_v_btp_level": 0,
            "norm2_btp_level": 16,
            "norm2_target_level": 16,
            "gelu_btp_level": 16,
            "down_btp_level": 0,
        },
        "route": [
            {"name": "Norm", "level": 0},
            {"name": "QKV", "level": 2},
            {"name": "QK_T", "level": 3},
            {"name": "Softmax", "level": 0},  # bootstrap before softmax
            {"name": "AttnV", "level": 5},
            {"name": "O", "level": 6},
            {"name": "Norm", "level": 0},      # bootstrap before norm2
            {"name": "Up", "level": 2},
            {"name": "GELU", "level": 0},      # bootstrap before GELU
            {"name": "Down", "level": 5},
        ],
        "bootstraps": [
            {"after": "QK_T", "before": "Softmax", "from_level": 3, "to_level": 0, "position": 3},
            {"after": "O", "before": "Norm", "from_level": 6, "to_level": 0, "position": 6},
            {"after": "Up", "before": "GELU", "from_level": 2, "to_level": 0, "position": 8},
        ],
    }


def test_pruned_cpp_output():
    out = _run(make_pruned_placement(), "cpp")
    assert "make_gpt2_optimized_config" in out
    assert "GPT2ModelConfig" in out
    assert "num_layers = 12" in out
    assert "Auto-generated" in out
    print("PASS: pruned cpp output")


def test_pruned_json_output():
    out = _run(make_pruned_placement(), "json")
    data = json.loads(out)
    assert data["model"] == "gpt2"
    assert data["num_layers"] == 12
    assert len(data["route"]) == 12
    # Odd route entries should have level=3 (from bootstrap)
    assert data["route"][1].get("level") == 3
    print("PASS: pruned json output")


def test_flat_cpp_output():
    out = _run(make_flat_placement(), "cpp")
    assert "make_gpt2_optimized_config" in out
    # Should detect bootstrap before softmax → attn_btp_level
    assert "attn_btp_level" in out
    # Should detect bootstrap before GELU → gelu_btp_level
    assert "gelu_btp_level" in out
    # Should detect bootstrap before second Norm → norm2_btp_level
    assert "norm2_btp_level" in out
    print("PASS: flat cpp output")


def test_flat_json_output():
    out = _run(make_flat_placement(), "json")
    data = json.loads(out)
    dc = data.get("decoder_config", {})
    assert "attn_btp_level" in dc
    assert "gelu_btp_level" in dc
    assert "norm2_btp_level" in dc
    print("PASS: flat json output")


def test_custom_num_layers():
    placement = make_pruned_placement()
    out = _run(placement, "cpp", num_layers=6)
    assert "num_layers = 6" in out
    print("PASS: custom num_layers")


if __name__ == "__main__":
    test_pruned_cpp_output()
    test_pruned_json_output()
    test_flat_cpp_output()
    test_flat_json_output()
    test_custom_num_layers()
    print("\nAll gen_config tests passed!")
