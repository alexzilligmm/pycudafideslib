"""
Extract GPT-2 (small, 124M) weights from HuggingFace and save them
in the flat-text format expected by the C++ FHE inference engine.

Weight layout
-------------
The C++ linear() kernel packs weight matrices column-major into
CKKS plaintext vectors of length `slots = N/2`.  Each .txt file
holds one such vector per line (one float per element, space-separated).

For a weight matrix of shape (in_dim, out_dim) we store it as
    ceil(in_dim * out_dim / slots)  plaintext vectors.

Biases are stored as a single plaintext vector (zero-padded to slots).

Directory structure written to `--out_dir` (default: weights-gpt2/):

    wte.txt               token embeddings        (vocab_size, 768)
    wpe.txt               position embeddings      (1024, 768)

    layer{i}_Wq.txt       Q projection weight      (768, 768)
    layer{i}_Wk.txt       K projection weight      (768, 768)
    layer{i}_Wv.txt       V projection weight      (768, 768)
    layer{i}_Wo.txt       output projection weight  (768, 768)
    layer{i}_bq.txt       ...biases...
    layer{i}_bk.txt
    layer{i}_bv.txt
    layer{i}_bo.txt

    layer{i}_Wu.txt       MLP fc (up) weight       (768, 3072)
    layer{i}_Wd.txt       MLP proj (down) weight   (3072, 768)
    layer{i}_bu.txt
    layer{i}_bd.txt

    layer{i}_ln1_g.txt    LayerNorm 1 gamma        (768,)
    layer{i}_ln1_b.txt    LayerNorm 1 beta         (768,)
    layer{i}_ln2_g.txt    LayerNorm 2 gamma        (768,)
    layer{i}_ln2_b.txt    LayerNorm 2 beta         (768,)

    ln_f_g.txt            final LayerNorm gamma
    ln_f_b.txt            final LayerNorm beta

Usage:
    python prepare_gpt2_weights.py [--model gpt2] [--out_dir weights-gpt2] [--slots 2048]
"""

import argparse
import os
import numpy as np
from pathlib import Path


def save_matrix(path: str, mat: np.ndarray, slots: int):
    """Flatten matrix row-major and chunk into plaintext vectors of length `slots`."""
    flat = mat.flatten()
    n_vecs = int(np.ceil(len(flat) / slots))
    padded = np.zeros(n_vecs * slots, dtype=np.float64)
    padded[: len(flat)] = flat
    vecs = padded.reshape(n_vecs, slots)
    np.savetxt(path, vecs, fmt="%.12f")


def save_vector(path: str, vec: np.ndarray, slots: int):
    """Save a 1-D vector as a single plaintext (zero-padded to `slots`)."""
    padded = np.zeros(slots, dtype=np.float64)
    padded[: len(vec)] = vec
    np.savetxt(path, padded.reshape(1, -1), fmt="%.12f")


def main():
    parser = argparse.ArgumentParser(description="Extract GPT-2 weights for FHE inference")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="HuggingFace model name (default: gpt2 = GPT-2 small 124M)")
    parser.add_argument("--out_dir", type=str, default="weights-gpt2",
                        help="Output directory")
    parser.add_argument("--slots", type=int, default=2048,
                        help="CKKS slot count = 2^(logN-1). Default 2048 for logN=12")
    args = parser.parse_args()

    # Lazy import so the script fails fast on arg errors
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print(f"Loading {args.model} from HuggingFace...")
    model = GPT2LMHeadModel.from_pretrained(args.model)
    model.eval()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    S = args.slots

    sd = model.state_dict()

    def g(key):
        return sd[key].detach().float().numpy()

    # ---- Embeddings --------------------------------------------------------
    wte = g("transformer.wte.weight")          # (vocab_size, 768)
    wpe = g("transformer.wpe.weight")          # (1024, 768)
    save_matrix(str(out / "wte.txt"), wte, S)
    save_matrix(str(out / "wpe.txt"), wpe, S)
    print(f"  wte {wte.shape}  wpe {wpe.shape}")

    # ---- Per-layer weights --------------------------------------------------
    n_layers = model.config.n_layer    # 12 for gpt2-small
    n_heads = model.config.n_head      # 12
    hid = model.config.n_embd          # 768
    head_dim = hid // n_heads          # 64

    for i in range(n_layers):
        prefix = f"transformer.h.{i}"

        # GPT-2 stores Q/K/V as a single fused (768, 2304) matrix.
        # Split into (768,768) each.
        c_attn_w = g(f"{prefix}.attn.c_attn.weight")   # (768, 2304)
        c_attn_b = g(f"{prefix}.attn.c_attn.bias")     # (2304,)

        Wq = c_attn_w[:, :hid]                          # (768, 768)
        Wk = c_attn_w[:, hid:2*hid]
        Wv = c_attn_w[:, 2*hid:]
        bq = c_attn_b[:hid]
        bk = c_attn_b[hid:2*hid]
        bv = c_attn_b[2*hid:]

        # Output projection
        Wo = g(f"{prefix}.attn.c_proj.weight")           # (768, 768)
        bo = g(f"{prefix}.attn.c_proj.bias")             # (768,)

        # MLP: GPT-2 has c_fc (up) and c_proj (down), no gate
        Wu = g(f"{prefix}.mlp.c_fc.weight")              # (768, 3072)
        bu = g(f"{prefix}.mlp.c_fc.bias")                # (3072,)
        Wd = g(f"{prefix}.mlp.c_proj.weight")            # (3072, 768)
        bd = g(f"{prefix}.mlp.c_proj.bias")              # (768,)

        # LayerNorm 1 (before attention)
        ln1_g = g(f"{prefix}.ln_1.weight")               # (768,)
        ln1_b = g(f"{prefix}.ln_1.bias")                 # (768,)

        # LayerNorm 2 (before MLP)
        ln2_g = g(f"{prefix}.ln_2.weight")
        ln2_b = g(f"{prefix}.ln_2.bias")

        lp = f"layer{i}_"

        # Transpose weights so they are (in_features, out_features) for the
        # column-major packing expected by linear().
        # NOTE: GPT-2 Conv1D stores weights as (in, out) already, unlike
        # nn.Linear which stores (out, in). Verify:
        #   c_attn.weight shape is (768, 2304) -> already (in, out).
        save_matrix(str(out / f"{lp}Wq.txt"), Wq, S)
        save_matrix(str(out / f"{lp}Wk.txt"), Wk, S)
        save_matrix(str(out / f"{lp}Wv.txt"), Wv, S)
        save_matrix(str(out / f"{lp}Wo.txt"), Wo, S)
        save_vector(str(out / f"{lp}bq.txt"), bq, S)
        save_vector(str(out / f"{lp}bk.txt"), bk, S)
        save_vector(str(out / f"{lp}bv.txt"), bv, S)
        save_vector(str(out / f"{lp}bo.txt"), bo, S)

        save_matrix(str(out / f"{lp}Wu.txt"), Wu, S)
        save_matrix(str(out / f"{lp}Wd.txt"), Wd, S)
        save_vector(str(out / f"{lp}bu.txt"), bu, S)
        save_vector(str(out / f"{lp}bd.txt"), bd, S)

        save_vector(str(out / f"{lp}ln1_g.txt"), ln1_g, S)
        save_vector(str(out / f"{lp}ln1_b.txt"), ln1_b, S)
        save_vector(str(out / f"{lp}ln2_g.txt"), ln2_g, S)
        save_vector(str(out / f"{lp}ln2_b.txt"), ln2_b, S)

        print(f"  layer {i}: Wq{Wq.shape} Wu{Wu.shape} Wd{Wd.shape}")

    # ---- Final LayerNorm ---------------------------------------------------
    ln_f_g = g("transformer.ln_f.weight")
    ln_f_b = g("transformer.ln_f.bias")
    save_vector(str(out / "ln_f_g.txt"), ln_f_g, S)
    save_vector(str(out / "ln_f_b.txt"), ln_f_b, S)

    # ---- Summary -----------------------------------------------------------
    total_params = sum(p.numel() for p in model.parameters())
    total_files = 2 + n_layers * 16 + 2  # embeddings + per-layer + final LN
    print(f"\nDone. {total_params:,} parameters across {total_files} files in {out}/")
    print(f"  Model: {args.model}")
    print(f"  Layers: {n_layers}, hidDim: {hid}, ffDim: {4*hid}, heads: {n_heads}")
    print(f"  Slots: {S}")


if __name__ == "__main__":
    main()
