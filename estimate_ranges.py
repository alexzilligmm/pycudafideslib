"""
Profile activation ranges for GPT-2 by running plaintext inference and
compute optimal FHE configuration values from observed ranges.

Outputs:
  - ranges-gpt2.json: per-layer and global activation statistics
  - Printed recommendations for NormConfig, SoftmaxConfig, GeluConfig

Usage:
    uv run python estimate_ranges.py [--model gpt2] [--n_samples 256] \
                                     [--seq_len 128] [--out ranges-gpt2.json]
"""

import argparse
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def compute_optimal_linear_init(d_min, d_max):
    """
    Optimal tangent-line initialisation for Newton-Raphson 1/sqrt(x).
    From the Cachemir / EncLLM paper.
    Returns (alpha, beta) such that  y0 = alpha * x + beta  ≈ 1/sqrt(x)
    on [d_min, d_max].
    """
    alpha = 4.0 / (d_min + d_max)
    beta = alpha ** 2 / 4.0
    return alpha, beta


def compute_optimal_linear_init_minimax(d_min, d_max):
    """
    Minimax-optimal linear initialisation for Goldschmidt 1/x.
    Returns (alpha, beta) such that  x0 = alpha  is a good starting
    guess for goldschmidt_inv(a) when a in [d_min, d_max].
    """
    alpha = (d_min ** 0.5 + d_max ** 0.5) ** 2 / (2 * d_min * d_max)
    beta = 1.0 / (d_min * d_max)
    return alpha, beta

def profile_gpt2(model, tokenizer, n_samples: int, seq_len: int, device: str):
    model.eval()
    model.to(device)
    n_layers = model.config.n_layer
    hid = model.config.n_embd
    n_heads = model.config.n_head
    head_dim = hid // n_heads

    stats = {
        "gelu_input":          [[] for _ in range(n_layers)],
        "softmax_input":       [[] for _ in range(n_layers)],
        "softmax_exp_sum":     [[] for _ in range(n_layers)],
        "norm1_input":         [[] for _ in range(n_layers)],
        "norm1_variance":      [[] for _ in range(n_layers)],
        "norm2_input":         [[] for _ in range(n_layers)],
        "norm2_variance":      [[] for _ in range(n_layers)],
        "residual_post_attn":  [[] for _ in range(n_layers)],
        "residual_post_mlp":   [[] for _ in range(n_layers)],
    }

    vocab_size = model.config.vocab_size
    rng = np.random.default_rng(42)

    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "In a shocking finding, scientists discovered a herd of unicorns",
        "The largest city in the world by population is",
        "def fibonacci(n):\n    if n <= 1:\n        return n",
        "Breaking news: researchers have found that",
        "Once upon a time in a land far far away there lived",
        "The meaning of life is",
        "import torch\nimport numpy as np\n\nclass Model(",
    ]

    input_ids_list = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, return_tensors="pt")
        if tokens.shape[1] < seq_len:
            pad = torch.randint(0, vocab_size, (1, seq_len - tokens.shape[1]))
            tokens = torch.cat([tokens, pad], dim=1)
        input_ids_list.append(tokens[:, :seq_len])

    for _ in range(max(0, n_samples - len(prompts))):
        ids = torch.tensor(rng.integers(0, vocab_size, size=(1, seq_len)),
                           dtype=torch.long)
        input_ids_list.append(ids)
    input_ids_list = input_ids_list[:n_samples]

    def record(name, layer_idx, tensor):
        t = tensor.detach().float()
        stats[name][layer_idx].append({
            "min": t.min().item(),
            "max": t.max().item(),
            "mean": t.mean().item(),
            "std": t.std().item(),
            "absmax": t.abs().max().item(),
        })

    with torch.no_grad():
        for sample_idx, input_ids in enumerate(input_ids_list):
            input_ids = input_ids.to(device)

            wte = model.transformer.wte(input_ids)
            wpe = model.transformer.wpe(
                torch.arange(seq_len, device=device).unsqueeze(0))
            hidden = wte + wpe

            for i, block in enumerate(model.transformer.h):
                # Pre-attention LayerNorm
                record("norm1_input", i, hidden)
                ln1_out = block.ln_1(hidden)
                mean = hidden.mean(dim=-1, keepdim=True)
                var = ((hidden - mean) ** 2).mean(dim=-1)
                record("norm1_variance", i, var)

                # QKV + Attention
                qkv = block.attn.c_attn(ln1_out)
                q, k, v = qkv.split(hid, dim=-1)
                B, T, C = q.shape
                q = q.view(B, T, n_heads, head_dim).transpose(1, 2)
                k = k.view(B, T, n_heads, head_dim).transpose(1, 2)
                v = v.view(B, T, n_heads, head_dim).transpose(1, 2)

                attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
                record("softmax_input", i, attn_scores)

                # Causal mask then softmax — track exp sum for Goldschmidt init
                causal_mask = torch.tril(torch.ones(T, T, device=device)).view(1, 1, T, T)
                scores_masked = attn_scores.masked_fill(causal_mask == 0, float("-inf"))
                # Numerically stable: exp(x - max)
                score_max = scores_masked.max(dim=-1, keepdim=True).values
                score_max = score_max.clamp(min=-1e9)  # avoid -inf
                exp_scores = torch.exp(scores_masked - score_max)
                exp_sum = exp_scores.sum(dim=-1)
                record("softmax_exp_sum", i, exp_sum)

                attn_weights = exp_scores / exp_sum.unsqueeze(-1)
                attn_out = torch.matmul(attn_weights, v)
                attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
                attn_out = block.attn.c_proj(attn_out)

                hidden = hidden + attn_out
                record("residual_post_attn", i, hidden)

                # Pre-MLP LayerNorm
                record("norm2_input", i, hidden)
                ln2_out = block.ln_2(hidden)
                mean2 = hidden.mean(dim=-1, keepdim=True)
                var2 = ((hidden - mean2) ** 2).mean(dim=-1)
                record("norm2_variance", i, var2)

                # MLP
                up = block.mlp.c_fc(ln2_out)
                record("gelu_input", i, up)
                act = F.gelu(up)
                down = block.mlp.c_proj(act)
                hidden = hidden + down
                record("residual_post_mlp", i, hidden)

            if (sample_idx + 1) % 50 == 0 or sample_idx == 0:
                print(f"  Processed {sample_idx + 1}/{len(input_ids_list)} samples")

    return stats


# ── Aggregation ─────────────────────────────────────────────────────────

def aggregate(stats):
    result = {}
    for name, layer_list in stats.items():
        per_layer = []
        global_min, global_max, global_absmax = float("inf"), float("-inf"), 0.0
        all_means, all_stds = [], []

        for samples in layer_list:
            if not samples:
                per_layer.append(None)
                continue
            lmin = min(s["min"] for s in samples)
            lmax = max(s["max"] for s in samples)
            labsmax = max(s["absmax"] for s in samples)
            lmean = np.mean([s["mean"] for s in samples])
            lstd = np.mean([s["std"] for s in samples])
            per_layer.append({
                "min": round(lmin, 6), "max": round(lmax, 6),
                "absmax": round(labsmax, 6),
                "mean": round(float(lmean), 6), "std": round(float(lstd), 6),
            })
            global_min = min(global_min, lmin)
            global_max = max(global_max, lmax)
            global_absmax = max(global_absmax, labsmax)
            all_means.append(lmean)
            all_stds.append(lstd)

        result[name] = {
            "global": {
                "min": round(global_min, 6), "max": round(global_max, 6),
                "absmax": round(global_absmax, 6),
                "mean": round(float(np.mean(all_means)), 6),
                "std": round(float(np.mean(all_stds)), 6),
            },
            "per_layer": per_layer,
        }
    return result


# ── Summary + recommendations ───────────────────────────────────────────

def print_summary(result):
    print("\n" + "=" * 70)
    print("  ACTIVATION RANGE SUMMARY")
    print("=" * 70)

    for name, data in result.items():
        g = data["global"]
        print(f"\n  {name}:")
        print(f"    global  min={g['min']:>12.4f}  max={g['max']:>12.4f}"
              f"  absmax={g['absmax']:>10.4f}  std={g['std']:>10.4f}")
        for i, layer in enumerate(data["per_layer"]):
            if layer is None:
                continue
            print(f"    L{i:02d}     min={layer['min']:>12.4f}"
                  f"  max={layer['max']:>12.4f}"
                  f"  absmax={layer['absmax']:>10.4f}"
                  f"  std={layer['std']:>10.4f}")

    # ── Compute optimal configs ──
    print("\n" + "=" * 70)
    print("  FHE CONFIG RECOMMENDATIONS")
    print("=" * 70)

    # GELU rescale factor
    gelu_g = result["gelu_input"]["global"]
    print(f"\n  GELU:")
    print(f"    Input range: [{gelu_g['min']:.2f}, {gelu_g['max']:.2f}]")
    print(f"    absmax: {gelu_g['absmax']:.2f}")
    # rescale_factor maps (x - threshold) into [-1, 1] for sign()
    # sign() needs input in [-1, 1], so rescale = 1 / max_distance_from_threshold
    # The thresholds are -4, -1.95, 3; worst case distance is absmax + 4
    sign_input_range = gelu_g["absmax"] + 4.0
    gelu_rescale = 1.0 / sign_input_range
    print(f"    Suggested rescale_factor: {gelu_rescale:.6f}")
    print(f"    Current rescale_factor:   0.100000")

    # Softmax given_max_val
    sm_g = result["softmax_input"]["global"]
    print(f"\n  Softmax:")
    print(f"    QK^T range: [{sm_g['min']:.4f}, {sm_g['max']:.4f}]")
    suggested_max = math.ceil(sm_g["absmax"] * 1.2)
    print(f"    Suggested given_max_val: {suggested_max}.0 (absmax={sm_g['absmax']:.2f} + 20%)")

    # Softmax Goldschmidt 1/x init from exp_sum range
    es_g = result["softmax_exp_sum"]["global"]
    print(f"\n  Softmax Goldschmidt 1/x init (minimax):")
    print(f"    exp_sum range: [{es_g['min']:.6f}, {es_g['max']:.6f}]")
    d_min_sm = max(es_g["min"], 1e-6)
    d_max_sm = es_g["max"]
    alpha_sm, beta_sm = compute_optimal_linear_init_minimax(d_min_sm, d_max_sm)
    print(f"    Suggested gs_inv_init (alpha): {alpha_sm:.6f}")
    print(f"    Current gs_inv_init: 3.8")

    # LayerNorm — current setup uses TAYLOR init
    n1v_g = result["norm1_variance"]["global"]
    n2v_g = result["norm2_variance"]["global"]
    var_min = min(n1v_g["min"], n2v_g["min"])
    var_max = max(n1v_g["max"], n2v_g["max"])
    var_min = max(var_min, 1e-10)  # clamp
    print(f"\n  LayerNorm (TAYLOR init):")
    print(f"    Norm1 var: [{n1v_g['min']:.6f}, {n1v_g['max']:.6f}]")
    print(f"    Norm2 var: [{n2v_g['min']:.6f}, {n2v_g['max']:.6f}]")
    print(f"    Combined var range: [{var_min:.6f}, {var_max:.6f}]")
    print(f"    Current taylor_z0: 0.5")
    print(f"    gs_d_min: {var_min:.6f}")
    print(f"    gs_d_max: {var_max:.6f}")

    # For future REMEZ init (uses Goldschmidt internally, needs linear init)
    alpha_nr, beta_nr = compute_optimal_linear_init(var_min, var_max)
    print(f"\n  LayerNorm (REMEZ init — for future use):")
    print(f"    Optimal GS init: alpha={alpha_nr:.6f}  beta={beta_nr:.6f}")
    print(f"    (nr_init_coeffs = {{ {beta_nr:.6f}, {alpha_nr:.6f} }})")

    # Per-layer recommendations
    print(f"\n  Per-layer variance ranges and taylor_z0:")
    for i in range(len(result["norm1_variance"]["per_layer"])):
        n1 = result["norm1_variance"]["per_layer"][i]
        n2 = result["norm2_variance"]["per_layer"][i]
        if n1 is None or n2 is None:
            continue
        vmin = max(min(n1["min"], n2["min"]), 1e-10)
        vmax = max(n1["max"], n2["max"])
        print(f"    L{i:02d}  var=[{vmin:.4f}, {vmax:.4f}]"
              f"  taylor_z0={1.0/math.sqrt((vmin+vmax)/2):.4f}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Profile GPT-2 activation ranges for FHE configuration")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--n_samples", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--out", default="ranges-gpt2.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    model = GPT2LMHeadModel.from_pretrained(args.model)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)

    print(f"Profiling {args.n_samples} samples, seq_len={args.seq_len}...")
    raw_stats = profile_gpt2(model, tokenizer, args.n_samples,
                             args.seq_len, args.device)

    result = aggregate(raw_stats)
    print_summary(result)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved ranges to {args.out}")


if __name__ == "__main__":
    main()
