"""
Generate test vectors for validating our C++ GPT-2 layer against PyTorch.

For each of the first N layers:
  1. Generate a random hidden-state input
  2. Run it through the real PyTorch GPT-2 decoder layer
  3. Save input and intermediate/final outputs as .txt files

The companion CUDA test (test_gpt2_real.cu) loads these files, runs the same
input through our C++ ops (linear with BSGS-packed absorbed weights), and
compares the outputs.

Usage: uv run python tests/generate_gpt2_test_vectors.py [--out_dir test_vectors] [--layers 1]
"""

import argparse
import math
import numpy as np
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prepare_gpt2_weights import (
    find_bsgs_factors, bsgs_pack_square, bsgs_pack_rect,
    absorb_ln_into_linear, pad_matrix, pad_vector,
)
from test_bsgs_linear import (
    pack_weights_interleaved, make_interleaved,
    pack_weights_paper, pack_weights_paper_rect, encode_sparse, decode_sparse,
    linear_paper, compute_perm_f, _find_bsgs_params,
)


def encode_expand_output(y, S, d_in, d_out):
    """Encode expand output y in interleaved slot format.

    For expand (d_in < d_out): the C++ algorithm outputs at slot m*t_out
    the value y[perm(m)] where perm(m) = m//α + (m%α)*d_in, α = d_out/d_in.
    This matches what linear_interleaved produces after the fix (baby=min).
    """
    alpha = d_out // d_in
    t_out = S // d_out
    cy = np.zeros(S)
    for m in range(d_out):
        cy[m * t_out] = y[m // alpha + (m % alpha) * d_in]
    return cy


def save_vec(path, v):
    """Save 1-D vector as single line of space-separated floats."""
    np.savetxt(str(path), v.reshape(1, -1), fmt="%.15e")


def save_mat(path, M):
    """Save 2-D matrix: one row per line, space-separated."""
    np.savetxt(str(path), M, fmt="%.15e")


def replicate_to_slots(v, S):
    """Replicate vector of length d into S slots with period d."""
    d = len(v)
    out = np.zeros(S, dtype=np.float64)
    for start in range(0, S, d):
        end = min(start + d, S)
        out[start:end] = v[:end - start]
    return out


def our_norm(x):
    """Bare LayerNorm matching our CUDA norm(): (x - mean) / sqrt(var + eps)."""
    mean = np.mean(x)
    var = np.var(x)
    return (x - mean) / np.sqrt(var + 1e-5)


def gelu_new(x):
    """GPT-2 GELU matching PyTorch gelu_new / our CUDA gelu()."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="test_vectors")
    parser.add_argument("--layers", type=int, default=1, help="Number of layers to generate")
    parser.add_argument("--hid", type=int, default=1024,
                        help="Padded hidden dim (power of 2). GPT-2 small: 768→1024")
    parser.add_argument("--ff", type=int, default=4096,
                        help="Padded FFN dim (power of 2). GPT-2 small: 3072→4096")
    parser.add_argument("--slots", type=int, default=32768,
                        help="CKKS slots = 2^(logN-1). Default 32768 (logN=16)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    from transformers import GPT2LMHeadModel

    print(f"Loading GPT-2 small from HuggingFace...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    sd = model.state_dict()
    g = lambda key: sd[key].detach().float().numpy()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hD, fD, S = args.hid, args.ff, args.slots
    d_ff = max(hD, fD)

    np.random.seed(args.seed)

    for layer_idx in range(args.layers):
        print(f"\n=== Layer {layer_idx} ===")
        ldir = out / f"layer{layer_idx}"
        ldir.mkdir(exist_ok=True)

        prefix = f"transformer.h.{layer_idx}"
        hid_real = 768

        # ── Load raw weights ──────────────────────────────────────────
        c_attn_w = g(f"{prefix}.attn.c_attn.weight")  # (768, 2304)
        c_attn_b = g(f"{prefix}.attn.c_attn.bias")
        Wq_raw = c_attn_w[:, :hid_real].T
        Wk_raw = c_attn_w[:, hid_real:2*hid_real].T
        Wv_raw = c_attn_w[:, 2*hid_real:].T
        bq_raw = c_attn_b[:hid_real]
        bk_raw = c_attn_b[hid_real:2*hid_real]
        bv_raw = c_attn_b[2*hid_real:]

        Wo_raw = g(f"{prefix}.attn.c_proj.weight").T
        bo_raw = g(f"{prefix}.attn.c_proj.bias")
        Wu_raw = g(f"{prefix}.mlp.c_fc.weight").T
        bu_raw = g(f"{prefix}.mlp.c_fc.bias")
        Wd_raw = g(f"{prefix}.mlp.c_proj.weight").T
        bd_raw = g(f"{prefix}.mlp.c_proj.bias")

        ln1_g = g(f"{prefix}.ln_1.weight")
        ln1_b = g(f"{prefix}.ln_1.bias")
        ln2_g = g(f"{prefix}.ln_2.weight")
        ln2_b = g(f"{prefix}.ln_2.bias")

        # ── Pad to target dims ────────────────────────────────────────
        Wq_pad = pad_matrix(Wq_raw, hD, hD)
        Wk_pad = pad_matrix(Wk_raw, hD, hD)
        Wv_pad = pad_matrix(Wv_raw, hD, hD)
        bq_pad = pad_vector(bq_raw, hD)
        bk_pad = pad_vector(bk_raw, hD)
        bv_pad = pad_vector(bv_raw, hD)
        Wo_pad = pad_matrix(Wo_raw, hD, hD)
        bo_pad = pad_vector(bo_raw, hD)
        Wu_pad = pad_matrix(Wu_raw, fD, hD)
        bu_pad = pad_vector(bu_raw, fD)
        Wd_pad = pad_matrix(Wd_raw, hD, fD)
        bd_pad = pad_vector(bd_raw, hD)
        ln1_g_pad = pad_vector(ln1_g, hD)
        ln1_b_pad = pad_vector(ln1_b, hD)
        ln2_g_pad = pad_vector(ln2_g, hD)
        ln2_b_pad = pad_vector(ln2_b, hD)

        # ── Absorb LN1 into QKV, LN2 into Up ─────────────────────────
        Wq_abs, bq_abs = absorb_ln_into_linear(Wq_pad, bq_pad, ln1_g_pad, ln1_b_pad)
        Wk_abs, bk_abs = absorb_ln_into_linear(Wk_pad, bk_pad, ln1_g_pad, ln1_b_pad)
        Wv_abs, bv_abs = absorb_ln_into_linear(Wv_pad, bv_pad, ln1_g_pad, ln1_b_pad)
        Wu_abs, bu_abs = absorb_ln_into_linear(Wu_pad, bu_pad, ln2_g_pad, ln2_b_pad)

        # ── BSGS pack weights ─────────────────────────────────────────
        # BSGS computes packed_M^T @ x.  Pack W.T so BSGS gives W @ x.
        # In real mode, linear() skips pre/post-broadcast — no intRot scaling.
        for name, W in [("Wq", Wq_abs), ("Wk", Wk_abs), ("Wv", Wv_abs), ("Wo", Wo_pad)]:
            packed = bsgs_pack_square(W.T, S, hD)
            save_mat(ldir / f"{name}.txt", packed)

        packed_Wu = bsgs_pack_rect(Wu_abs.T, S, hD, fD)
        save_mat(ldir / "Wu.txt", packed_Wu)
        packed_Wd = bsgs_pack_rect(Wd_pad.T, S, fD, hD)
        save_mat(ldir / "Wd.txt", packed_Wd)

        # Save absorbed biases (replicated to S slots)
        for name, b, d in [("bq", bq_abs, hD), ("bk", bk_abs, hD), ("bv", bv_abs, hD),
                           ("bo", bo_pad, hD), ("bu", bu_abs, fD), ("bd", bd_pad, hD)]:
            save_vec(ldir / f"{name}.txt", replicate_to_slots(b, S))

        # ── Generate random input ─────────────────────────────────────
        # Scale to realistic GPT-2 hidden state range
        x_np = (np.random.randn(hD) * 0.5).astype(np.float64)

        # ── Run through PyTorch ───────────────────────────────────────
        # We need a float32 (768,) vector for PyTorch
        x_pt_np = x_np[:hid_real].astype(np.float32)
        x_torch = torch.tensor(x_pt_np).unsqueeze(0).unsqueeze(0)  # (1,1,768)

        block = model.transformer.h[layer_idx]
        with torch.no_grad():
            # Pre-attention LN → QKV
            ln1_out = block.ln_1(x_torch)
            qkv = block.attn.c_attn(ln1_out)
            q_pt, k_pt, v_pt = qkv.split(hid_real, dim=-1)

            # Out projection (for seqLen=1, attn_output = V)
            n_heads = 12
            head_dim = hid_real // n_heads
            q_mh = q_pt.view(1, 1, n_heads, head_dim).transpose(1, 2)
            k_mh = k_pt.view(1, 1, n_heads, head_dim).transpose(1, 2)
            v_mh = v_pt.view(1, 1, n_heads, head_dim).transpose(1, 2)
            scores = torch.matmul(q_mh, k_mh.transpose(-2, -1)) / math.sqrt(head_dim)
            probs = torch.softmax(scores, dim=-1)  # 1.0 for seqLen=1
            attn_out_mh = torch.matmul(probs, v_mh)
            attn_out = attn_out_mh.transpose(1, 2).contiguous().view(1, 1, hid_real)
            out_proj = block.attn.c_proj(attn_out)

            # Residual 1
            res1 = x_torch + out_proj

            # Pre-MLP LN → MLP
            ln2_out = block.ln_2(res1)
            up_out = block.mlp.c_fc(ln2_out)
            gelu_out = block.mlp.act(up_out)
            down_out = block.mlp.c_proj(gelu_out)

            # Residual 2 (full layer output)
            res2 = res1 + down_out

        # ── Save input (replicated to S slots) ────────────────────────
        x_slots = replicate_to_slots(x_np, S)
        save_vec(ldir / "input.txt", x_slots)

        # ── Compute our pipeline intermediates ────────────────────────
        # norm1
        norm1 = our_norm(x_np)

        # Q, K, V via absorbed weights (padded dims)
        q_ours = Wq_abs @ norm1 + bq_abs
        k_ours = Wk_abs @ norm1 + bk_abs
        v_ours = Wv_abs @ norm1 + bv_abs

        # Verify against PyTorch Q (first 768 elements should match)
        q_pt_vec = q_pt.squeeze().numpy().astype(np.float64)
        q_err = np.max(np.abs(q_ours[:hid_real] - q_pt_vec))
        print(f"  Q verify: max_err={q_err:.2e} (should be < 1e-3)")

        # For seqLen=1: attention output = V
        attn_ours = v_ours.copy()

        # Output projection
        out_ours = Wo_pad.astype(np.float64) @ attn_ours + bo_pad.astype(np.float64)

        # Residual 1
        res1_ours = x_np + out_ours

        # norm2 → Up (LN2 absorbed) → GELU → Down
        norm2 = our_norm(res1_ours)
        up_ours = Wu_abs @ norm2 + bu_abs
        gelu_ours = gelu_new(up_ours)
        down_ours = Wd_pad.astype(np.float64) @ gelu_ours + bd_pad.astype(np.float64)

        # Residual 2
        res2_ours = res1_ours + down_ours

        # Verify final output against PyTorch
        res2_pt_vec = res2.squeeze().numpy().astype(np.float64)
        res2_err = np.max(np.abs(res2_ours[:hid_real] - res2_pt_vec))
        print(f"  Layer output verify: max_err={res2_err:.2e} (should be < 1e-2)")

        # ── Save expected outputs (replicated to S slots) ─────────────
        # These are what the CUDA test will compare against.
        # Each is the output of one sub-op in our pipeline.
        save_vec(ldir / "expected_norm1.txt", replicate_to_slots(norm1, S))
        save_vec(ldir / "expected_q.txt", replicate_to_slots(q_ours, S))
        save_vec(ldir / "expected_k.txt", replicate_to_slots(k_ours, S))
        save_vec(ldir / "expected_v.txt", replicate_to_slots(v_ours, S))
        save_vec(ldir / "expected_out_proj.txt", replicate_to_slots(out_ours, S))
        save_vec(ldir / "expected_res1.txt", replicate_to_slots(res1_ours, S))
        save_vec(ldir / "expected_norm2.txt", replicate_to_slots(norm2, S))
        save_vec(ldir / "expected_up.txt", replicate_to_slots(up_ours, S))
        save_vec(ldir / "expected_gelu.txt", replicate_to_slots(gelu_ours, S))
        save_vec(ldir / "expected_down.txt", replicate_to_slots(down_ours, S))
        save_vec(ldir / "expected_output.txt", replicate_to_slots(res2_ours, S))

        # Also save the PyTorch reference (first 768 slots only, for debugging)
        save_vec(ldir / "pytorch_q.txt", pad_vector(q_pt_vec, S))
        save_vec(ldir / "pytorch_output.txt", pad_vector(res2_pt_vec, S))

        # ── Paper's d²/S CacheMIR-packed weights and expected outputs ───
        # Uses sparse encoding (x at every t-th slot) and the paper's
        # d²/S weight plaintexts with pre/post processing rotations.
        # y = x @ W convention, so we pass W.T to get W @ x.
        idir = ldir / "interleaved"
        idir.mkdir(exist_ok=True)

        # Pack square QKV/Out weights: d²/S plaintexts each
        print(f"  Packing paper d²/S weights (hD={hD}, S={S})...")
        n_pt_sq = hD * hD // S
        print(f"    Square: {n_pt_sq} plaintexts each (was {hD} with old interleaved)")
        for name, W in [("Wq", Wq_abs), ("Wk", Wk_abs), ("Wv", Wv_abs), ("Wo", Wo_pad)]:
            # Paper computes y = x @ W_input.  We want y = W @ x, so pass W.T.
            packed, r_i, r_o = pack_weights_paper(W.T, S, hD)
            save_mat(idir / f"{name}.txt", np.array(packed))

        # Wu (Up expand hD→fD): rectangular CacheMIR, hD*fD/S plaintexts.
        n_pt_up = hD * fD // S
        packed_wu, r_i_up, r_o_up = pack_weights_paper_rect(Wu_abs.T, S, hD, fD)
        print(f"    Up rect:   {n_pt_up} plaintexts (r_i={r_i_up}, r_o={r_o_up})")
        save_mat(idir / "Wu.txt", np.array(packed_wu))

        # Wd (Down shrink fD→hD): rectangular CacheMIR, fD*hD/S plaintexts.
        # At logN=13 fD=S → t_in=1 (dense input, preprocess is no-op).
        n_pt_down = fD * hD // S
        packed_wd, r_i_down, r_o_down = pack_weights_paper_rect(Wd_pad.T, S, fD, hD)
        print(f"    Down rect: {n_pt_down} plaintexts (r_i={r_i_down}, r_o={r_o_down})")
        save_mat(idir / "Wd.txt", np.array(packed_wd))

        # Save BSGS params metadata
        import json
        _, r_i_sq, r_o_sq = pack_weights_paper(np.eye(hD), S, hD)
        t_out_up   = S // fD   # = 1 when fD=S (logN=13 with fD=4096)
        t_in_down  = S // fD   # = 1 when fD=S
        t_out_down = S // hD   # = 4 at logN=13
        bsgs_meta = {
            "n_pt_sq": n_pt_sq, "r_i_sq": r_i_sq, "r_o_sq": r_o_sq,
            "n_pt_up": n_pt_up, "r_i_up": r_i_up, "r_o_up": r_o_up,
            "n_pt_down": n_pt_down, "r_i_down": r_i_down, "r_o_down": r_o_down,
            "t_in": S // hD, "t_out_sq": S // hD, "t_out_up": t_out_up,
            "t_in_down": t_in_down, "t_out_down": t_out_down,
        }
        with open(idir / "bsgs_meta.json", "w") as f:
            json.dump(bsgs_meta, f, indent=2)

        # Input and expected outputs in sparse encoding (x[k] at slot k*t)
        # Square QKV/Out: t = S/hD
        save_vec(idir / "input.txt", encode_sparse(norm1, S, hD))
        save_vec(idir / "expected_q.txt", encode_sparse(q_ours, S, hD))
        save_vec(idir / "expected_k.txt", encode_sparse(k_ours, S, hD))
        save_vec(idir / "expected_v.txt", encode_sparse(v_ours, S, hD))
        save_vec(idir / "expected_out_proj.txt", encode_sparse(out_ours, S, hD))

        # Up: sparse input t_in=S/hD; output INTERLEAVED (expand with baby=min).
        # Slot m*t_out holds up_ours[perm(m)] where perm(m)=m//α+(m%α)*hD.
        save_vec(idir / "input_up.txt", encode_sparse(norm2[:hD], S, hD))
        save_vec(idir / "expected_up.txt", encode_expand_output(up_ours[:fD], S, hD, fD))

        # Down: input is GELU output; t_in=S/fD (=1 dense when fD=S); output t_out=S/hD
        save_vec(idir / "input_down.txt", encode_sparse(gelu_ours[:fD], S, fD))
        save_vec(idir / "expected_down.txt", encode_sparse(down_ours[:hD], S, hD))

        # Biases in sparse encoding (match the output slot layout)
        for name, b, dim in [("bq", bq_abs, hD), ("bk", bk_abs, hD), ("bv", bv_abs, hD),
                              ("bo", bo_pad, hD)]:
            save_vec(idir / f"{name}.txt", encode_sparse(b, S, hD))
        # Up bias: interleaved format (matches expand output layout)
        save_vec(idir / "bu.txt", encode_expand_output(bu_abs[:fD], S, hD, fD))
        # Down bias: t_out=S/hD=4 → sparse
        save_vec(idir / "bd.txt", encode_sparse(bd_pad[:hD], S, hD))

        print(f"  Saved paper d²/S weights to {idir}/")

        print(f"  Saved {len(list(ldir.iterdir()))} files to {ldir}/")

    # ── Config metadata ───────────────────────────────────────────────
    import json
    meta = {
        "model": "gpt2",
        "hidDim": hD, "ffDim": fD, "slots": S,
        "num_layers": args.layers, "seed": args.seed,
        "inRot_sq": find_bsgs_factors(hD)[0],
        "outRot_sq": find_bsgs_factors(hD)[1],
        "inRot_ff": find_bsgs_factors(d_ff)[0],
        "outRot_ff": find_bsgs_factors(d_ff)[1],
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone. Metadata in {out}/meta.json")


if __name__ == "__main__":
    main()
