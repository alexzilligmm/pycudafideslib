import argparse
import math
import sys
import time as _t
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pathlib import Path


def find_bsgs_factors(d):
    """Find inRot, outRot such that inRot*outRot == d and inRot ~ sqrt(d)."""
    inRot = int(math.sqrt(d))
    while inRot > 1 and d % inRot != 0:
        inRot -= 1
    outRot = d // inRot
    return inRot, outRot


def _modinv(a, m):
    """Modular inverse of a mod m (extended Euclidean)."""
    if m == 1:
        return 0
    g, x = _extended_gcd(a % m, m)[:2]
    assert g == 1, f"No inverse: gcd({a},{m})={g}"
    return x % m


def _extended_gcd(a, b):
    if a == 0:
        return b, 0, 1
    g, x1, y1 = _extended_gcd(b % a, a)
    return g, y1 - (b // a) * x1, x1


def compute_perm_f(S, d):
    """Compute preprocessing permutation f: after preprocess, slot i holds x[f(i)].

    f(i) = ((i + l_i*(d-1)) mod S) // t
    where l_i is the unique l in {0..t-1} such that t | (i + l*(d-1)).
    """
    t = S // d
    if t == 1:
        return np.arange(d)
    inv_dm1 = _modinv(d - 1, t)
    i_arr = np.arange(S)
    l_arr = (-i_arr * inv_dm1) % t
    return ((i_arr + l_arr * (d - 1)) % S) // t


def _get_device():
    """Return torch device: CUDA if available, else CPU."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def bsgs_pack_square(W, S, d):
    """
    BSGS diagonal packing for a d*d weight matrix — GPU-accelerated.

    w[diag][slot] = W[(slot%d + diag%inRot) % d][(slot%d - (diag//inRot)*inRot) % d]

    Key optimisation: the pattern repeats with period d across S slots,
    so we compute a (d, d) core block and tile to (d, S).

    Returns: numpy array of shape (d, S).
    """
    import torch
    inRot, outRot = find_bsgs_factors(d)
    assert inRot * outRot == d, f"Cannot factor d={d} for BSGS"

    dev = _get_device()
    W_t = torch.from_numpy(np.ascontiguousarray(W)).to(dev)

    diags = torch.arange(d, device=dev)
    i_arr = diags % inRot                        # (d,)  baby-step
    j_arr = diags // inRot                       # (d,)  giant-step
    s_core = torch.arange(d, device=dev)         # (d,)  only one period

    # (d, d) index arrays — 32x smaller than (d, S)
    rows = (s_core.unsqueeze(0) + i_arr.unsqueeze(1)) % d
    cols = (s_core.unsqueeze(0) - j_arr.unsqueeze(1) * inRot) % d

    core = W_t[rows, cols]                       # (d, d) on GPU
    # Tile to S slots
    n_tiles = S // d
    packed = core.repeat(1, n_tiles).cpu().numpy()
    return packed


def bsgs_pack_rect(W, S, d_in, d_out):
    """
    BSGS diagonal packing for rectangular matrix (d_in x d_out).
    Pads to square max(d_in, d_out) then packs.
    """
    d = max(d_in, d_out)
    W_sq = np.zeros((d, d), dtype=np.float64)
    W_sq[:d_in, :d_out] = W
    return bsgs_pack_square(W_sq, S, d)


def bsgs_pack_interleaved(W, S, d_in, d_out=0):
    """CacheMIR interleaved weight packing — d_in*d_out/S plaintexts.

    Unified compute_perm_f encoding for all cases (expand/square/shrink):
      t_baby = max(t_in, t_out)
      t_giant = t_baby * r_i
      p_{j,k}[q] = W[f((q + j*t_baby) % S), ((q - k*t_giant) % S) / t_out % d_out]
      where f = compute_perm_f(S, d_in).

    Returns: numpy array of shape (n_pt, S).
    """
    if d_out <= 0:
        d_out = d_in
    t_in  = S // d_in
    t_out = S // d_out
    n_pt  = d_in * d_out // S

    # BSGS factors: r_i * r_o = n_pt
    r_i = max(1, int(math.floor(math.sqrt(n_pt))))
    while n_pt % r_i != 0 and r_i > 1:
        r_i -= 1
    r_o = n_pt // r_i

    q = np.arange(S)
    packed = np.zeros((n_pt, S), dtype=np.float64)

    t_baby = max(t_in, t_out)
    t_giant = t_baby * r_i
    f = compute_perm_f(S, d_in)

    for j in range(r_i):
        shifted_q = (q + j * t_baby) % S
        rows = f[shifted_q]
        for k in range(r_o):
            cols = ((q - k * t_giant) % S) // t_out % d_out
            packed[j * r_o + k] = W[rows, cols]

    return packed


def save_packed(path, weights_array, full_width=False):
    """Save packed weight as binary .npy.

    Standard packing: pattern repeats with period d, save core (n_diags, d).
    Interleaved (full_width=True): pattern is unique across S, save full (n_pt, S).
    """
    if full_width:
        npy_path = str(path).replace('.txt', '.npy')
        np.save(npy_path, weights_array.astype(np.float64))
    else:
        d = weights_array.shape[0]
        core = weights_array[:, :d]  # (d, d) — the unique part
        npy_path = str(path).replace('.txt', '.npy')
        np.save(npy_path, core.astype(np.float64))


def save_vector(path, vec, S):
    """Save a 1-D vector, zero-padded and replicated to S slots as binary .npy."""
    d = len(vec)
    padded = np.tile(vec, S // d + 1)[:S].astype(np.float64)
    npy_path = str(path).replace('.txt', '.npy')
    np.save(npy_path, padded)


def pad_matrix(W, target_rows, target_cols):
    """Zero-pad matrix to target dimensions."""
    result = np.zeros((target_rows, target_cols), dtype=np.float64)
    result[:W.shape[0], :W.shape[1]] = W
    return result


def pad_vector(v, target_len):
    """Zero-pad vector to target length."""
    result = np.zeros(target_len, dtype=np.float64)
    result[:len(v)] = v
    return result


def absorb_ln_into_linear(W, bias, ln_gamma, ln_beta):
    """
    Absorb LayerNorm affine into the next linear layer's weights and bias.

    Given: y = W @ (gamma * norm_x + beta) + bias
         = (W @ diag(gamma)) @ norm_x + (W @ beta + bias)

    Returns: W_absorbed, b_absorbed
    """
    W_absorbed = W * ln_gamma[np.newaxis, :]
    b_absorbed = W @ ln_beta + bias
    return W_absorbed, b_absorbed


def main():
    parser = argparse.ArgumentParser(
        description="Extract GPT-2 weights in BSGS diagonal-packed format "
                    "with LN gamma/bias absorption")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--out_dir", type=str, default="weights-gpt2")
    parser.add_argument("--slots", type=int, default=32768,
                        help="CKKS slots = 2^(logN-1). Default 32768 for logN=16")
    parser.add_argument("--hidDim", type=int, default=1024,
                        help="Padded hidden dim (must be power of 2). "
                             "Default 1024 (GPT-2 small 768 → next pow2)")
    parser.add_argument("--ffDim", type=int, default=4096,
                        help="Padded FFN dim (must be power of 2). "
                             "Default 4096 (GPT-2 small 3072 → next pow2)")
    parser.add_argument("--interleaved", action="store_true",
                        help="Use CacheMIR interleaved packing (d²/S plaintexts). "
                             "4× fewer weight plaintexts per layer.")
    args = parser.parse_args()

    from transformers import GPT2LMHeadModel

    print(f"Loading {args.model} from HuggingFace...")
    model = GPT2LMHeadModel.from_pretrained(args.model)
    model.eval()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    S = args.slots
    hD = args.hidDim
    fD = args.ffDim

    sd = model.state_dict()
    def g(key):
        return sd[key].detach().float().numpy()

    n_layers = model.config.n_layer    # 12
    n_heads  = model.config.n_head     # 12
    hid_real = model.config.n_embd     # 768
    ff_real  = 4 * hid_real            # 3072

    for name, d in [("hidDim", hD), ("ffDim", max(hD, fD))]:
        ratio = S // d
        assert S % d == 0, f"S={S} not divisible by {name}={d}"
        assert ratio & (ratio - 1) == 0, f"S/{name} = {ratio} must be a power of 2"

    print(f"  Real dims: hid={hid_real}, ff={ff_real}, heads={n_heads}")
    print(f"  Padded dims: hid={hD}, ff={fD}")
    print(f"  Slots: {S}, logN~{int(math.log2(2*S))}")

    inRot_sq, outRot_sq = find_bsgs_factors(hD)
    d_ff = max(hD, fD)
    inRot_ff, outRot_ff = find_bsgs_factors(d_ff)
    print(f"  Square BSGS: inRot={inRot_sq}, outRot={outRot_sq}, n_diags={hD}")
    print(f"  FFN BSGS:    inRot={inRot_ff}, outRot={outRot_ff}, n_diags={d_ff}")

    wte = g("transformer.wte.weight")   # (vocab_size, 768)
    wpe = g("transformer.wpe.weight")   # (1024, 768)
    wte_pad = pad_matrix(wte, wte.shape[0], hD)
    wpe_pad = pad_matrix(wpe, wpe.shape[0], hD)

    def save_embedding_table(path, emb):
        """Save (vocab, hD) embedding: each row replicated to S slots, as .npy."""
        n_rows = emb.shape[0]
        result = np.zeros((n_rows, S), dtype=np.float64)
        for start in range(0, S, hD):
            end = min(start + hD, S)
            result[:, start:end] = emb[:, :end - start]
        npy_path = str(path).replace('.txt', '.npy')
        np.save(npy_path, result)

    t0 = _t.time()
    save_embedding_table(out / "wte.txt", wte_pad)
    print(f"  wte saved ({_t.time()-t0:.1f}s) {wte.shape}->{wte_pad.shape}")
    sys.stdout.flush()
    t0 = _t.time()
    save_embedding_table(out / "wpe.txt", wpe_pad)
    print(f"  wpe saved ({_t.time()-t0:.1f}s) {wpe.shape}->{wpe_pad.shape}")
    sys.stdout.flush()

    for i in range(n_layers):
        prefix = f"transformer.h.{i}"
        lp = f"layer{i}_"
        print(f"Processing layer {prefix}")

        c_attn_w = g(f"{prefix}.attn.c_attn.weight")   # (768, 2304) = (in, 3*out)
        c_attn_b = g(f"{prefix}.attn.c_attn.bias")     # (2304,)

        Wq_raw = c_attn_w[:, :hid_real].T               # (768, 768) = (out, in)
        Wk_raw = c_attn_w[:, hid_real:2*hid_real].T
        Wv_raw = c_attn_w[:, 2*hid_real:].T
        bq_raw = c_attn_b[:hid_real]
        bk_raw = c_attn_b[hid_real:2*hid_real]
        bv_raw = c_attn_b[2*hid_real:]

        Wo_raw = g(f"{prefix}.attn.c_proj.weight").T     # (out=768, in=768)
        bo_raw = g(f"{prefix}.attn.c_proj.bias")

        Wu_raw = g(f"{prefix}.mlp.c_fc.weight").T        # (out=3072, in=768)
        bu_raw = g(f"{prefix}.mlp.c_fc.bias")
        Wd_raw = g(f"{prefix}.mlp.c_proj.weight").T      # (out=768, in=3072)
        bd_raw = g(f"{prefix}.mlp.c_proj.bias")

        ln1_g = g(f"{prefix}.ln_1.weight")    # (768,)
        ln1_b = g(f"{prefix}.ln_1.bias")      # (768,)
        ln2_g = g(f"{prefix}.ln_2.weight")    # (768,)
        ln2_b = g(f"{prefix}.ln_2.bias")      # (768,)

        ln1_g_pad = pad_vector(ln1_g, hD)
        ln1_b_pad = pad_vector(ln1_b, hD)
        ln2_g_pad = pad_vector(ln2_g, hD)
        ln2_b_pad = pad_vector(ln2_b, hD)

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

        Wq_abs, bq_abs = absorb_ln_into_linear(Wq_pad, bq_pad, ln1_g_pad, ln1_b_pad)
        Wk_abs, bk_abs = absorb_ln_into_linear(Wk_pad, bk_pad, ln1_g_pad, ln1_b_pad)
        Wv_abs, bv_abs = absorb_ln_into_linear(Wv_pad, bv_pad, ln1_g_pad, ln1_b_pad)

        Wu_abs, bu_abs = absorb_ln_into_linear(Wu_pad, bu_pad, ln2_g_pad, ln2_b_pad)

        d_ff = max(hD, fD)
        layer_t0 = _t.time()

        save_pool = ThreadPoolExecutor(max_workers=2)
        save_futures = []

        if args.interleaved:
            for name, W, d_in, d_out in [
                ("Wq", Wq_abs, hD, hD), ("Wk", Wk_abs, hD, hD),
                ("Wv", Wv_abs, hD, hD), ("Wo", Wo_pad, hD, hD),
                ("Wu", Wu_abs, hD, fD), ("Wd", Wd_pad, fD, hD),
            ]:
                t0 = _t.time()
                packed = bsgs_pack_interleaved(W.T, S, d_in, d_out)
                dt = _t.time() - t0
                n_pt = d_in * d_out // S
                print(f"    {name}: interleaved pack {dt:.2f}s  "
                      f"shape={packed.shape} (n_pt={n_pt})")
                sys.stdout.flush()
                save_futures.append(
                    save_pool.submit(save_packed, str(out / f"{lp}{name}.txt"),
                                     packed, full_width=True))
        else:
            for name, W, d_pack in [
                ("Wq", Wq_abs, hD), ("Wk", Wk_abs, hD),
                ("Wv", Wv_abs, hD), ("Wo", Wo_pad, hD),
            ]:
                t0 = _t.time()
                packed = bsgs_pack_square(W.T, S, d_pack)
                dt = _t.time() - t0
                print(f"    {name}: pack {dt:.2f}s  shape={packed.shape}")
                sys.stdout.flush()
                save_futures.append(
                    save_pool.submit(save_packed, str(out / f"{lp}{name}.txt"),
                                     packed))

            for tag, W, d_in, d_out in [("Wu", Wu_abs, hD, fD),
                                         ("Wd", Wd_pad, fD, hD)]:
                t0 = _t.time()
                packed = bsgs_pack_rect(W.T, S, d_in, d_out)
                dt = _t.time() - t0
                print(f"    {tag}: pack {dt:.2f}s  shape={packed.shape}")
                sys.stdout.flush()
                save_futures.append(
                    save_pool.submit(save_packed, str(out / f"{lp}{tag}.txt"),
                                     packed))

        for name, b in [("bq", bq_abs), ("bk", bk_abs), ("bv", bv_abs),
                        ("bo", bo_pad), ("bu", bu_abs), ("bd", bd_pad)]:
            save_vector(str(out / f"{lp}{name}.txt"), b, S)

        for f in save_futures:
            f.result()
        save_pool.shutdown(wait=False)

        print(f"  layer {i} done in {_t.time()-layer_t0:.1f}s")
        sys.stdout.flush()

    ln_f_g = pad_vector(g("transformer.ln_f.weight"), hD)
    ln_f_b = pad_vector(g("transformer.ln_f.bias"), hD)  # noqa: F841 — used by lm_head bias

    lm_head_absorbed = wte_pad * ln_f_g[np.newaxis, :hD]
    save_embedding_table(out / "lm_head.txt", lm_head_absorbed)
    print(f"  lm_head: {lm_head_absorbed.shape} (final LN gamma absorbed)")

    total_params = sum(p.numel() for p in model.parameters())
    if args.interleaved:
        n_sq_weights = 4 * (hD * hD // S)   # Q,K,V,Out: hD²/S each
        n_up_weights = hD * fD // S          # Up: hD*fD/S (expand hD→fD)
        n_down_weights = fD * hD // S        # Down: fD*hD/S (shrink fD→hD)
        n_weights_per_layer = n_sq_weights + n_up_weights + n_down_weights
        mode_str = "full-interleaved (CacheMIR)"
    else:
        n_sq_weights = 4 * hD
        n_up_weights = max(hD, fD)
        n_down_weights = max(hD, fD)
        n_weights_per_layer = n_sq_weights + n_up_weights + n_down_weights
        mode_str = "standard (replicated)"
    total_diags = n_layers * n_weights_per_layer
    print(f"\nDone. {total_params:,} params in {out}/  [{mode_str}]")
    print(f"  Weights per layer: {n_weights_per_layer} plaintexts")
    if args.interleaved:
        print(f"    attn (Q/K/V/Out): {n_sq_weights} interleaved Ptx")
        print(f"    Up:               {n_up_weights} interleaved Ptx (expand)")
        print(f"    Down:             {n_down_weights} interleaved Ptx (shrink)")
    print(f"  Total plaintexts: {total_diags}")
    print(f"  Memory estimate: {total_diags * S * 8 / 1e9:.1f} GB "
          f"(raw plaintext floats)")
    if args.interleaved:
        n_std = 4 * hD + 2 * max(hD, fD)
        ratio = n_std / n_weights_per_layer
        print(f"  Savings vs standard: {n_std} → {n_weights_per_layer} "
              f"Ptx/layer ({ratio:.1f}x reduction)")
    print(f"  Absorption: LN gamma -> W columns (saves 2 ct*pt mults/layer = "
          f"{2 * n_layers} mults total)")


if __name__ == "__main__":
    main()
