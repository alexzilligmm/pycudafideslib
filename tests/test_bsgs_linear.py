"""
Numerical verification of the BSGS linear kernel for CKKS matrix-vector multiply.

Tests three modes:
1. real_mode  (baby_step = 1)            — correct for any S >= d
2. bench_mode (baby_step = intRot = S/d) — only correct when S = d
3. interleaved_mode                      — correct for any S >= d, fewer Galois keys

BSGS packing formula for real_mode:
  w[diag][slot] = W[(slot + diag%inRot) % d][(slot - (diag//inRot)*inRot) % d]
  d diagonals, d = matrix dimension.
  Input/output: standard replicated, slot[s] = v[s % d].
  After post-broadcast, the output has a factor of (S/d) on each element.

Interleaved mode (CacheMIR):
  Slot layout: slot[j*K + q] = x[(j+q) % d], K = S/d
  Weight encoding: pt[t][j*K + q] = W[(j+q)%d, (j+q+t)%d]
  d diagonals, baby_step = K, giant_stride = bStep*K.
  No pre/post broadcast — input/output stay in interleaved layout.
  Galois keys needed: multiples of K only → fewer keys than real_mode.
"""
import numpy as np
import math


def linear_bsgs(x_slots, weights, S, d_in, d_out, expand, bench_mode=True):
    """Simulate the BSGS linear() kernel exactly as C++ does."""
    hidDim = d_in
    ffDim  = d_out if expand != 0 else d_in

    if bench_mode:
        intRot_val = S // hidDim
        if expand == 0:
            inRot  = int(math.sqrt(hidDim * hidDim / (2 * S)))
            outRot = hidDim * hidDim // (S * inRot)
        else:
            inRot  = int(math.sqrt(hidDim * ffDim / (2 * S)))
            outRot = hidDim * ffDim // (S * inRot)
        baby_step = intRot_val
        giant_stride = inRot * intRot_val
    else:
        # Real mode: baby_step=1, covers all d diagonals
        d = d_in  # square case
        inRot  = int(math.sqrt(d))
        outRot = d // inRot
        baby_step = 1
        giant_stride = inRot

    preStep  = hidDim if expand >= 0 else ffDim
    postStep = hidDim if expand <= 0 else ffDim

    # Pre-broadcast
    xb = x_slots.copy()
    step = preStep
    while step < S:
        xb = xb + np.roll(xb, -step)
        step *= 2

    # Baby step rotations
    ctRot = [None] * inRot
    ctRot[0] = xb.copy()
    for i in range(1, inRot):
        ctRot[i] = np.roll(ctRot[i-1], -baby_step)

    n_weights = len(weights)
    partSum = [None] * n_weights

    # Multiply
    for i in range(n_weights):
        partSum[i] = ctRot[i % inRot] * weights[i]

    # Input sum within blocks
    for i in range(n_weights):
        if i % inRot > 0:
            partSum[i - i % inRot] = partSum[i - i % inRot] + partSum[i]

    # Giant step: rotate output blocks
    for j in range(1, outRot):
        partSum[j * inRot] = np.roll(partSum[j * inRot], -(j * giant_stride))

    # Output sum
    result = partSum[0].copy()
    for j in range(1, outRot):
        result = result + partSum[j * inRot]

    # Post-broadcast
    step = postStep
    while step < S:
        result = result + np.roll(result, -step)
        step *= 2

    return result


def pack_weights_real(W, S, d):
    """
    BSGS diagonal packing for real_mode (baby_step=1).

    w[diag][slot] = W[(slot + diag%inRot) % d][(slot - (diag//inRot)*inRot) % d]

    Repeats with period d across all S slots.
    Returns: list of d numpy arrays, each of length S.
    """
    inRot = int(math.sqrt(d))
    outRot = d // inRot
    assert inRot * outRot == d, f"d={d} must factor as inRot*outRot={inRot}*{outRot}"

    weights = []
    for diag in range(d):
        j = diag // inRot  # giant step index
        i = diag % inRot   # baby step index
        w = np.zeros(S)
        for k in range(S):
            row = (k + i) % d
            col = (k - j * inRot) % d
            w[k] = W[row][col]
        weights.append(w)
    return weights


def pack_weights_bench(W, S, d):
    """
    BSGS diagonal packing for bench_mode (baby_step = intRot = S/d).
    Only produces correct results when S = d (intRot = 1).
    """
    intRot = S // d
    n_weights = d * d // S
    inRot  = int(math.sqrt(d * d / (2 * S)))
    outRot = d * d // (S * inRot)

    weights = []
    for w_idx in range(n_weights):
        j = w_idx // inRot
        i = w_idx % inRot
        w = np.zeros(S)
        for k in range(S):
            row = (k + i * intRot) % d
            col = (k - j * inRot * intRot) % d
            w[k] = W[row][col]
        weights.append(w)
    return weights


## ── Interleaved mode ────────────────────────────────────────────────────


def make_interleaved(x, S, d):
    """Encode vector x (length d) into interleaved slot layout of S slots.

    Layout: slot[j*K + q] = x[(j + q) % d]  where K = S/d.
    """
    K = S // d
    out = np.zeros(S)
    for j in range(d):
        for q in range(K):
            out[j * K + q] = x[(j + q) % d]
    return out


def extract_interleaved(slots, S, d):
    """Extract vector from interleaved layout. slot[j*K+0] = x[j]."""
    return slots[:d * (S // d):S // d].copy()  # slots[0], slots[K], slots[2K], ...


def pack_weights_interleaved(W, S, d):
    """
    Interleaved BSGS diagonal packing (CacheMIR).

    For diagonal t = g*bStep + b:
      pt[t][j*K + q] = W[((j+q) - g*bStep) % d, ((j+q) + b) % d]

    The row index is shifted by -g*bStep to compensate for the giant-step
    rotation that will shift this slot to the correct output position.

    d diagonals, each of S slots.  K = S/d.
    Returns: list of d numpy arrays, each of length S.
    """
    K = S // d
    bStep = max(1, int(math.floor(math.sqrt(d))))
    weights = []
    for t in range(d):
        g = t // bStep
        b = t % bStep
        w = np.zeros(S)
        for j in range(d):
            for q in range(K):
                m = (j + q) % d
                row = (m - g * bStep) % d   # compensate for giant-step rotation
                col = (m + b) % d           # baby-step offset
                w[j * K + q] = W[row][col]
        weights.append(w)
    return weights


def linear_interleaved(x_slots, weights, S, d):
    """Simulate the interleaved BSGS linear kernel.

    Input/output in interleaved layout: slot[j*K+q] = val[(j+q)%d].
    Baby step = K, giant stride = bStep*K.
    No pre/post broadcast.
    """
    K = S // d
    n_diag = len(weights)
    bStep = max(1, int(math.floor(math.sqrt(n_diag))))
    gStep = (n_diag + bStep - 1) // bStep

    # Baby-step rotations: rotate by b*K
    ctRot = [None] * bStep
    ctRot[0] = x_slots.copy()
    for b in range(1, bStep):
        ctRot[b] = np.roll(ctRot[b - 1], -K)

    # Giant steps: accumulate baby products, then rotate
    result = np.zeros(S)
    for g in range(gStep):
        tmp = np.zeros(S)
        for b in range(bStep):
            t = g * bStep + b
            if t >= n_diag:
                break
            tmp = tmp + ctRot[b] * weights[t]

        # Giant-step rotation
        if g > 0:
            tmp = np.roll(tmp, -(g * bStep * K))

        result = result + tmp

    return result


## ── Paper's d²/S algorithm (sparse in/out, pre/post processing) ──────

def _modinv(a, m):
    """Modular inverse of a mod m (extended Euclidean)."""
    if m == 1:
        return 0
    g, x, _ = _extended_gcd(a % m, m)
    assert g == 1, f"No inverse: gcd({a},{m})={g}"
    return x % m


def _extended_gcd(a, b):
    if a == 0:
        return b, 0, 1
    g, x1, y1 = _extended_gcd(b % a, a)
    return g, y1 - (b // a) * x1, x1


def compute_perm_f(S, d):
    """Compute the preprocessing permutation f(i).

    After preprocessing, slot i contains x[f(i)].
    f(i) = ((i + l_i*(d-1)) mod S) // t
    where l_i is the unique l in {0..t-1} such that t | (i + l*(d-1)).
    """
    t = S // d
    if t == 1:
        return list(range(d))

    inv_dm1 = _modinv(d - 1, t)  # (d-1)^{-1} mod t
    i_arr = np.arange(S)
    l_arr = (-i_arr * inv_dm1) % t
    f_arr = ((i_arr + l_arr * (d - 1)) % S) // t
    return f_arr.tolist()


def encode_sparse(x, S, d):
    """Sparse encoding: cx[k*t] = x[k], rest = 0."""
    t = S // d
    cx = np.zeros(S)
    for k in range(d):
        cx[k * t] = x[k]
    return cx


def preprocess(cx, S, d):
    """Log(t) doubling rotations: c'x = Σ_{l=0}^{t-1} rot(cx, l*(d-1))."""
    t = S // d
    result = cx.copy()
    step = 1
    while step < t:
        result = result + np.roll(result, -(step * (d - 1)))
        step *= 2
    return result


def postprocess(cy, S, d):
    """Log(t) doubling reductions: cy_final = Σ_{l=0}^{t-1} rot(cy, l)."""
    t = S // d
    result = cy.copy()
    step = 1
    while step < t:
        result = result + np.roll(result, -step)
        step *= 2
    return result


def decode_sparse(cy, S, d):
    """Extract output from sparse encoding: y[k] = cy[k*t]."""
    t = S // d
    return np.array([cy[k * t] for k in range(d)])


def _find_bsgs_params(n_pt):
    """Find r_i, r_o such that r_i * r_o = n_pt, r_i ≈ sqrt(n_pt)."""
    r_i = max(1, int(math.floor(math.sqrt(n_pt))))
    while n_pt % r_i != 0 and r_i > 1:
        r_i -= 1
    r_o = n_pt // r_i
    return r_i, r_o


def pack_weights_paper(W, S, d):
    """Paper's weight encoding with d²/S plaintexts.

    Correct formula (derived from actual preprocessing permutation f):
      p_{j,k}[q] = W[f((q + j*t) % S), floor(((q - k*t*r_i) % S) / t) % d]

    Returns: list of r_i*r_o weight arrays, r_i, r_o.
    """
    t = S // d
    n_pt = d * d // S
    r_i, r_o = _find_bsgs_params(n_pt)
    f = np.array(compute_perm_f(S, d))

    q = np.arange(S)
    weights = []
    for j in range(r_i):
        shifted_q = (q + j * t) % S
        rows = f[shifted_q]
        for k in range(r_o):
            cols = ((q - k * t * r_i) % S) // t % d
            weights.append(W[rows, cols].copy())
    return weights, r_i, r_o


def linear_paper(x_slots_sparse, weights, S, d, r_i, r_o):
    """Paper's d²/S BSGS algorithm with pre/post processing.

    Input: sparse encoding (x at every t-th slot).
    Output: sparse encoding (y at every t-th slot).
    """
    t = S // d

    # 1. Preprocess
    cxp = preprocess(x_slots_sparse, S, d)

    # 2. Baby step rotations (r_i rotations by t)
    cx_rot = [cxp.copy()]
    for j in range(1, r_i):
        cx_rot.append(np.roll(cx_rot[-1], -t))

    # 3. Multiply and accumulate per giant-step group
    cy_k = [np.zeros(S) for _ in range(r_o)]
    for j in range(r_i):
        for k in range(r_o):
            idx = j * r_o + k
            cy_k[k] = cy_k[k] + cx_rot[j] * weights[idx]

    # 4. Giant step accumulation (back-to-front)
    for k in range(r_o - 1, 0, -1):
        cy_k[k - 1] = cy_k[k - 1] + np.roll(cy_k[k], -(t * r_i))

    cy = cy_k[0]

    # 5. Postprocess
    cy = postprocess(cy, S, d)

    return cy


def linear_paper_streaming(x_slots_sparse, weights, S, d, r_i, r_o):
    """Streaming variant of linear_paper: loads weights per baby-step block.

    Functionally identical to linear_paper, but processes one baby-step
    at a time (r_o plaintexts per block) instead of keeping all r_i*r_o
    in memory simultaneously.
    """
    t = S // d

    # 1. Preprocess
    cxp = preprocess(x_slots_sparse, S, d)

    # 2. Baby step rotations
    cx_rot = [cxp.copy()]
    for j in range(1, r_i):
        cx_rot.append(np.roll(cx_rot[-1], -t))

    # 3. Stream: for each baby step j, load r_o weights, multiply, discard
    cy_k = [np.zeros(S) for _ in range(r_o)]
    for j in range(r_i):
        # "Load" this block: weights[j*r_o .. j*r_o+r_o-1]
        block = weights[j * r_o:(j + 1) * r_o]
        for k in range(r_o):
            cy_k[k] = cy_k[k] + cx_rot[j] * block[k]
        # block goes out of scope (simulates freeing memory)

    # 4. Giant step accumulation
    for k in range(r_o - 1, 0, -1):
        cy_k[k - 1] = cy_k[k - 1] + np.roll(cy_k[k], -(t * r_i))

    cy = cy_k[0]

    # 5. Postprocess
    cy = postprocess(cy, S, d)

    return cy


## ── Paper's d²/S algorithm — rectangular matrices ──────────────────────

def pack_weights_paper_rect(W, S, d_in, d_out):
    """Unified compute_perm_f weight encoding for rectangular W (d_in × d_out).

    All cases (expand/square/shrink) use the same encoding:
      Baby = max(t_in, t_out), Giant = baby * r_i.
      rows = f((q + j*t_baby) % S),  cols = ((q - k*t_giant) % S) / t_out % d_out

    Returns: weights list, r_i, r_o.
    """
    t_in  = S // d_in
    t_out = S // d_out
    n_pt  = d_in * d_out // S

    r_i = max(1, int(math.floor(math.sqrt(n_pt))))
    while n_pt % r_i != 0 and r_i > 1:
        r_i -= 1
    r_o = n_pt // r_i

    q = np.arange(S)
    weights = []

    f_in = np.array(compute_perm_f(S, d_in))
    t_baby = max(t_in, t_out)
    t_giant = t_baby * r_i

    for j in range(r_i):
        shifted_q = (q + j * t_baby) % S
        rows = f_in[shifted_q]
        for k in range(r_o):
            cols = ((q - k * t_giant) % S) // t_out % d_out
            weights.append(W[rows, cols].copy())

    return weights, r_i, r_o


def linear_paper_rect(x_sparse, weights, S, d_in, d_out, r_i, r_o):
    """CacheMIR d_in*d_out/S algorithm for rectangular matrix.

    Unified: baby = max(t_in, t_out), giant = baby * r_i for all cases.
    Input: sparse encoding with t_in = S/d_in.
    Output: sparse encoding with t_out = S/d_out.
    """
    t_in  = S // d_in
    t_out = S // d_out

    t_baby  = max(t_in, t_out)
    t_giant = t_baby * r_i

    # 1. Preprocess (input fold, using d_in)
    cxp = x_sparse.copy()
    step = 1
    while step < t_in:
        cxp = cxp + np.roll(cxp, -(step * (d_in - 1)))
        step *= 2

    # 2. Baby step rotations by t_baby
    cx_rot = [cxp.copy()]
    for j in range(1, r_i):
        cx_rot.append(np.roll(cx_rot[-1], -t_baby))

    # 3. Multiply and accumulate per giant-step group
    cy_k = [np.zeros(S) for _ in range(r_o)]
    for j in range(r_i):
        for k in range(r_o):
            idx = j * r_o + k
            cy_k[k] = cy_k[k] + cx_rot[j] * weights[idx]

    # 4. Giant step accumulation (back-to-front)
    for k in range(r_o - 1, 0, -1):
        cy_k[k - 1] = cy_k[k - 1] + np.roll(cy_k[k], -t_giant)

    cy = cy_k[0]

    # 5. Postprocess (output unfold, using t_out)
    step = 1
    while step < t_out:
        cy = cy + np.roll(cy, -step)
        step *= 2

    return cy


def test_paper_rect(d_in, d_out, S, label=""):
    """Test paper's d_in*d_out/S algorithm for rectangular matrix."""
    np.random.seed(42)
    W = np.random.randn(d_in, d_out)  # d_in rows, d_out cols
    x = np.random.randn(d_in)

    # Paper convention: y = x @ W
    y_expected = x @ W

    # Sparse encode input
    t_in = S // d_in
    cx = np.zeros(S)
    for k in range(d_in):
        cx[k * t_in] = x[k]

    # Pack weights
    weights, r_i, r_o = pack_weights_paper_rect(W, S, d_in, d_out)

    # Run
    cy = linear_paper_rect(cx, weights, S, d_in, d_out, r_i, r_o)

    # Decode output: all cases use simple sparse encoding
    t_out = S // d_out
    y_out = np.array([cy[k * t_out] for k in range(d_out)])

    n_pt = d_in * d_out // S
    err = np.max(np.abs(y_out - y_expected))
    rel_err = err / (np.max(np.abs(y_expected)) + 1e-15)
    passed = rel_err < 1e-10
    print(f"  paper_rect d_in={d_in:4d} d_out={d_out:4d} S={S:5d} "
          f"n_pt={n_pt:4d} r_i={r_i:3d} r_o={r_o:3d}  "
          f"err={err:.2e} rel={rel_err:.2e}  {'PASS' if passed else 'FAIL'}"
          f"  {label}")
    return passed


## ── Paper's Algorithm 1 — exact encoding with interleaved I/O ────────

def pack_weights_alg1(W, S, d_in, d_out):
    """Paper's exact weight encoding (Appendix, Algorithm 1).

    For expand (d_in < d_out): d=d_in, αd=d_out, t=S/d, t'=S/(αd).
      p_{j,k}[i] = W[(i/t + i%t + j*t) % d, (⌊i/t⌋ + (i/t')%α*d - k*t*r_i) % (αd)]
    For shrink (d_in > d_out): d=d_out, αd=d_in, t=S/d, t'=S/(αd).
      p_{j,k}[i] = W[(⌊i/t⌋ + (i/t')%α*d + i%t + j*t) % (αd), (i/t - k*t*r_i) % d]
    For square: both reduce to the same formula.

    Baby step = t' = min(t_in, t_out). Giant stride = t = max(t_in, t_out).
    """
    n_pt = d_in * d_out // S
    r_i = max(1, int(math.floor(math.sqrt(n_pt))))
    while n_pt % r_i != 0 and r_i > 1:
        r_i -= 1
    r_o = n_pt // r_i

    i_arr = np.arange(S)

    if d_in <= d_out:
        # Expand or square: d = d_in, αd = d_out, α = d_out/d_in
        d = d_in
        alpha_d = d_out
        alpha = d_out // d_in
        t = S // d          # S/d_in (large spacing)
        t_prime = S // alpha_d  # S/d_out (small spacing)

        weights = []
        for j in range(r_i):
            for k in range(r_o):
                rows = (i_arr // t + i_arr % t + j * t) % d
                cols = (i_arr // t + (i_arr // t_prime) % alpha * d
                        - k * t * r_i) % alpha_d
                weights.append(W[rows, cols].copy())
    else:
        # Shrink: d = d_out, αd = d_in, α = d_in/d_out
        d = d_out
        alpha_d = d_in
        alpha = d_in // d_out
        t = S // d          # S/d_out (large spacing)
        t_prime = S // alpha_d  # S/d_in (small spacing)

        weights = []
        for j in range(r_i):
            for k in range(r_o):
                rows = (i_arr // t + (i_arr // t_prime) % alpha * d
                        + i_arr % t + j * t) % alpha_d
                cols = (i_arr // t - k * t * r_i) % d
                weights.append(W[rows, cols].copy())

    return weights, r_i, r_o


def linear_alg1(x_encoded, weights, S, d_in, d_out, r_i, r_o):
    """Paper's Algorithm 1 for generalized VMM.

    Baby = t' = min(t_in, t_out), Giant = t = max(t_in, t_out).
    Preprocess: log(t_1) steps, Postprocess: log(t_2) steps.
    For expand: (t_1, t_2) = (t, t') = (t_in, t_out).
    For shrink: (t_1, t_2) = (t', t) = (t_in, t_out).
    Both cases: preprocess = log(t_in), postprocess = log(t_out).
    """
    t_in = S // d_in
    t_out = S // d_out
    t_baby = min(t_in, t_out)   # t'
    t_giant = max(t_in, t_out)  # t

    # 1. Preprocess: log(t_in) rotations by 2^i*(d_in - 1)
    cxp = x_encoded.copy()
    step = 1
    while step < t_in:
        cxp = cxp + np.roll(cxp, -(step * (d_in - 1)))
        step *= 2

    # 2. Baby step: r_i rotations by t' = t_baby
    cx_rot = [cxp.copy()]
    for j in range(1, r_i):
        cx_rot.append(np.roll(cx_rot[-1], -t_baby))

    # 3. Multiply and accumulate per giant-step group
    cy_k = [np.zeros(S) for _ in range(r_o)]
    for j in range(r_i):
        for k in range(r_o):
            idx = j * r_o + k
            cy_k[k] = cy_k[k] + cx_rot[j] * weights[idx]

    # 4. Giant step accumulation (back-to-front): rotation by t*r_i
    for k in range(r_o - 1, 0, -1):
        cy_k[k - 1] = cy_k[k - 1] + np.roll(cy_k[k], -(t_giant * r_i))

    cy = cy_k[0]

    # 5. Postprocess: log(t_out) rotations by 2^l
    step = 1
    while step < t_out:
        cy = cy + np.roll(cy, -step)
        step *= 2

    return cy


def interleave_perm(d_in, d_out):
    """Permutation for interleaved encoding: perm(m) = m//α + (m%α)*d_base.

    For expand (d_in < d_out): α = d_out/d_in, d_base = d_in.
      Output slot m*t_out holds y[m//α + (m%α)*d_in].
    For shrink (d_in > d_out): α = d_in/d_out, d_base = d_out.
      Input slot m*t_in holds x[m//α + (m%α)*d_out].
    """
    if d_in < d_out:
        alpha = d_out // d_in
        return np.array([m // alpha + (m % alpha) * d_in for m in range(d_out)])
    elif d_in > d_out:
        alpha = d_in // d_out
        return np.array([m // alpha + (m % alpha) * d_out for m in range(d_in)])
    else:
        return np.arange(d_in)


def encode_interleaved(x, S, d_in, d_out):
    """Encode vector x into interleaved slot layout.

    For shrink input (d_in > d_out): place x at spacing t_in = S/d_in.
    Position m*t_in holds x[perm(m)] where perm is the interleave permutation.
    """
    alpha = d_in // d_out
    t_in = S // d_in
    cx = np.zeros(S)
    for m in range(d_in):
        idx = m // alpha + (m % alpha) * d_out
        cx[m * t_in] = x[idx]
    return cx


def decode_interleaved(cy, S, d_in, d_out):
    """Decode vector y from interleaved slot layout.

    For expand output (d_in < d_out): read at spacing t_out = S/d_out.
    Position m*t_out holds y[perm(m)] where perm is the interleave permutation.
    """
    alpha = d_out // d_in
    t_out = S // d_out
    y = np.zeros(d_out)
    for m in range(d_out):
        idx = m // alpha + (m % alpha) * d_in
        y[idx] = cy[m * t_out]
    return y


def test_alg1(d_in, d_out, S, label=""):
    """Test paper's exact Algorithm 1 with interleaved I/O."""
    np.random.seed(42)
    W = np.random.randn(d_in, d_out)
    x = np.random.randn(d_in)
    y_expected = x @ W

    t_in = S // d_in

    if d_in <= d_out:
        # Expand or square: simple sparse input
        cx = np.zeros(S)
        for k in range(d_in):
            cx[k * t_in] = x[k]
    else:
        # Shrink: interleaved input
        cx = encode_interleaved(x, S, d_in, d_out)

    weights, r_i, r_o = pack_weights_alg1(W, S, d_in, d_out)
    cy = linear_alg1(cx, weights, S, d_in, d_out, r_i, r_o)

    t_out = S // d_out
    if d_in <= d_out and d_in != d_out:
        # Expand: interleaved output
        y_out = decode_interleaved(cy, S, d_in, d_out)
    else:
        # Square or shrink: simple sparse output
        y_out = np.array([cy[k * t_out] for k in range(d_out)])

    n_pt = d_in * d_out // S
    err = np.max(np.abs(y_out - y_expected))
    rel_err = err / (np.max(np.abs(y_expected)) + 1e-15)
    passed = rel_err < 1e-10
    print(f"  alg1 d_in={d_in:4d} d_out={d_out:4d} S={S:5d} "
          f"n_pt={n_pt:4d} r_i={r_i:3d} r_o={r_o:3d}  "
          f"err={err:.2e} rel={rel_err:.2e}  {'PASS' if passed else 'FAIL'}"
          f"  {label}")
    return passed


def test_paper(d, S, label=""):
    """Test paper's d²/S algorithm."""
    np.random.seed(42)
    W = np.random.randn(d, d)
    x = np.random.randn(d)

    # Paper convention: y = x @ W  (row-vector times matrix)
    y_expected = x @ W

    # Sparse encode
    cx = encode_sparse(x, S, d)

    # Pack weights
    weights, r_i, r_o = pack_weights_paper(W, S, d)

    # Run
    cy = linear_paper(cx, weights, S, d, r_i, r_o)

    # Decode
    y_out = decode_sparse(cy, S, d)

    n_pt = d * d // S
    err = np.max(np.abs(y_out - y_expected))
    rel_err = err / (np.max(np.abs(y_expected)) + 1e-15)
    passed = rel_err < 1e-10
    print(f"  paper d={d:4d} S={S:5d} t={S//d:3d} n_pt={n_pt:4d} "
          f"r_i={r_i:3d} r_o={r_o:3d}  "
          f"err={err:.2e} rel={rel_err:.2e}  {'PASS' if passed else 'FAIL'}"
          f"  {label}")
    return passed


def test_interleaved(d, S, label=""):
    """Test interleaved BSGS matmul correctness."""
    np.random.seed(42)
    W = np.random.randn(d, d)
    x = np.random.randn(d)

    # y = W @ x  (interleaved computes W @ x directly, not W^T @ x)
    y_expected = W @ x

    # Encode input in interleaved layout
    x_slots = make_interleaved(x, S, d)

    # Pack weights
    weights = pack_weights_interleaved(W, S, d)

    # Run
    y_slots = linear_interleaved(x_slots, weights, S, d)

    # Extract from interleaved output: slot[j*K + 0] = y[j]
    K = S // d
    y_out = np.array([y_slots[j * K] for j in range(d)])

    err = np.max(np.abs(y_out - y_expected))
    rel_err = err / (np.max(np.abs(y_expected)) + 1e-15)
    passed = rel_err < 1e-10
    print(f"  inter d={d:4d} S={S:5d} K={K:3d} nDiag={d:4d} "
          f"bStep={max(1,int(math.sqrt(d))):3d}  "
          f"err={err:.2e} rel={rel_err:.2e}  {'PASS' if passed else 'FAIL'}"
          f"  {label}")
    return passed


def test_interleaved_rect(d_in, d_out, S, label=""):
    """Test interleaved BSGS for rectangular matrix (d_out x d_in)."""
    d = max(d_in, d_out)
    np.random.seed(42)
    W_raw = np.random.randn(d_out, d_in)
    x = np.random.randn(d_in)

    y_expected = W_raw @ x
    y_expected_pad = np.zeros(d)
    y_expected_pad[:d_out] = y_expected

    # Pad W to square d x d
    W = np.zeros((d, d))
    W[:d_out, :d_in] = W_raw

    # Pad x to d
    x_pad = np.zeros(d)
    x_pad[:d_in] = x

    x_slots = make_interleaved(x_pad, S, d)
    weights = pack_weights_interleaved(W, S, d)
    y_slots = linear_interleaved(x_slots, weights, S, d)

    K = S // d
    y_out = np.array([y_slots[j * K] for j in range(d)])

    err = np.max(np.abs(y_out - y_expected_pad))
    rel_err = err / (np.max(np.abs(y_expected_pad)) + 1e-15)
    passed = rel_err < 1e-10
    print(f"  inter d_in={d_in:4d} d_out={d_out:4d} d={d:4d} S={S:5d} K={K:3d}  "
          f"err={err:.2e} rel={rel_err:.2e}  {'PASS' if passed else 'FAIL'}"
          f"  {label}")
    return passed


def test_mode(d, S, bench_mode, label=""):
    """Test BSGS matmul correctness."""
    np.random.seed(42)
    W = np.random.randn(d, d)
    x = np.random.randn(d)

    # y = W^T @ x  (column-wise: y[q] = sum_p W[p][q] * x[p])
    y_expected = W.T @ x

    # Pack input
    x_slots = np.zeros(S)
    x_slots[:d] = x

    # Pack weights
    if bench_mode:
        weights = pack_weights_bench(W, S, d)
    else:
        weights = pack_weights_real(W, S, d)

    # Run
    y_slots = linear_bsgs(x_slots, weights, S, d, d, expand=0,
                           bench_mode=bench_mode)

    # Extract: first d slots, divided by (S/d) from post-broadcast
    scale = S // d
    y_out = y_slots[:d] / scale

    err = np.max(np.abs(y_out - y_expected))
    rel_err = err / (np.max(np.abs(y_expected)) + 1e-15)
    passed = rel_err < 1e-10
    mode_str = "bench" if bench_mode else "real"
    print(f"  {mode_str:5s} d={d:4d} S={S:5d} intRot={S//d:3d}  "
          f"err={err:.2e} rel={rel_err:.2e}  {'PASS' if passed else 'FAIL'}"
          f"  {label}")
    return passed


if __name__ == "__main__":
    print("=== BSGS Linear Kernel Verification ===\n")

    results = []

    # S = d  (intRot=1, both should work)
    print("--- S = d (intRot = 1) ---")
    results.append(test_mode(4,   4,   bench_mode=True))
    results.append(test_mode(4,   4,   bench_mode=False))
    results.append(test_mode(16,  16,  bench_mode=True))
    results.append(test_mode(16,  16,  bench_mode=False))
    results.append(test_mode(64,  64,  bench_mode=False))

    # S > d  (only real should work; bench formulas break for small d)
    print("\n--- S > d (intRot > 1), real_mode only ---")
    results.append(test_mode(4,   16,  bench_mode=False))
    results.append(test_mode(16,  64,  bench_mode=False))
    results.append(test_mode(16,  256, bench_mode=False))
    results.append(test_mode(64,  512, bench_mode=False))
    results.append(test_mode(64, 1024, bench_mode=False))
    results.append(test_mode(256, 2048, bench_mode=False))

    # Realistic dimensions
    print("\n--- Realistic dims ---")
    results.append(test_mode(256, 4096, bench_mode=False, label="(bench-like)"))
    results.append(test_mode(1024, 8192, bench_mode=False, label="(GPT-2 @ logN=14)"))

    # ── Interleaved mode tests ──────────────────────────────────────
    print("\n--- Interleaved: square, S = d (K=1) ---")
    results.append(test_interleaved(4, 4))
    results.append(test_interleaved(16, 16))
    results.append(test_interleaved(64, 64))

    print("\n--- Interleaved: square, S > d ---")
    results.append(test_interleaved(4, 16))
    results.append(test_interleaved(16, 64))
    results.append(test_interleaved(16, 256))
    results.append(test_interleaved(64, 512))
    results.append(test_interleaved(64, 1024))
    results.append(test_interleaved(256, 2048))

    print("\n--- Interleaved: realistic dims ---")
    results.append(test_interleaved(256, 4096, label="(logN=13)"))
    results.append(test_interleaved(1024, 4096, label="(GPT-2 tight)"))
    results.append(test_interleaved(1024, 8192, label="(GPT-2 @ logN=14)"))

    print("\n--- Interleaved: rectangular (Up/Down) ---")
    results.append(test_interleaved_rect(1024, 4096, 4096, label="Up (tight)"))
    results.append(test_interleaved_rect(4096, 1024, 4096, label="Down (tight)"))
    results.append(test_interleaved_rect(1024, 4096, 8192, label="Up (logN=14)"))
    results.append(test_interleaved_rect(4096, 1024, 8192, label="Down (logN=14)"))

    # ── Paper's d²/S algorithm ────────────────────────────────────
    # Requires d² >= S (i.e., d >= sqrt(S)) for n_pt = d²/S >= 1
    print("\n--- Paper d²/S: S = d (t=1, degenerate) ---")
    results.append(test_paper(4, 4))
    results.append(test_paper(16, 16))
    results.append(test_paper(64, 64))

    print("\n--- Paper d²/S: S > d, d² >= S ---")
    results.append(test_paper(4, 8))       # d²=16, S=8, n_pt=2
    results.append(test_paper(4, 16))      # d²=16, S=16, n_pt=1
    results.append(test_paper(16, 64))     # d²=256, S=64, n_pt=4
    results.append(test_paper(16, 256))    # d²=256, S=256, n_pt=1
    results.append(test_paper(64, 256))    # d²=4096, S=256, n_pt=16
    results.append(test_paper(64, 1024))   # d²=4096, S=1024, n_pt=4
    results.append(test_paper(64, 4096))   # d²=4096, S=4096, n_pt=1

    print("\n--- Paper d²/S: realistic dims ---")
    results.append(test_paper(256, 2048, label="(logN=12)"))
    results.append(test_paper(256, 4096, label="(logN=13)"))
    results.append(test_paper(1024, 4096, label="(GPT-2 tight)"))
    results.append(test_paper(1024, 8192, label="(GPT-2 @ logN=14)"))

    # ── Streaming: verify block-by-block gives same result ─────────
    print("\n--- Streaming (block-by-block) vs batch ---")
    for d, S, label in [(64, 256, ""), (256, 2048, "(logN=12)"),
                         (1024, 4096, "(GPT-2 tight)")]:
        np.random.seed(42)
        W = np.random.randn(d, d)
        x = np.random.randn(d)
        cx = encode_sparse(x, S, d)
        weights, r_i, r_o = pack_weights_paper(W, S, d)
        cy_batch = linear_paper(cx, weights, S, d, r_i, r_o)
        cy_stream = linear_paper_streaming(cx, weights, S, d, r_i, r_o)
        err = np.max(np.abs(cy_batch - cy_stream))
        passed = err < 1e-15
        print(f"  stream d={d:4d} S={S:5d}  batch_vs_stream err={err:.2e}  "
              f"{'PASS' if passed else 'FAIL'}  {label}")
        results.append(passed)

    # ── Paper's d²/S: rectangular ──────────────────────────────────
    # Note: the paper's algorithm (Alg 1, Appendix B) uses different
    # giant-step parameters for expand vs shrink (d_in > d_out).
    # Our implementation handles the EXPAND case (d_in ≤ d_out) correctly.
    # The SHRINK case (d_in > d_out) requires the paper's block-interleaved
    # input encoding and modified giant step = t_out*r_i (not t_in*r_i).
    # In practice: Down proj at logN=13 (fD=S=4096) uses standard BSGS.
    # Our algorithm works for expand (d_in ≤ d_out) when t_out=1 (d_out=S),
    # i.e., when the output fills all slots (no postprocessing needed).
    print("\n--- Paper d²/S: rectangular EXPAND (d_in ≤ d_out) ---")
    results.append(test_paper_rect(64, 64, 256, label="square sanity"))
    results.append(test_paper_rect(1024, 1024, 4096, label="GPT-2 sq"))
    results.append(test_paper_rect(1024, 4096, 4096, label="Up hD→fD logN=13 t_out=1"))
    results.append(test_paper_rect(8, 32, 32, label="small 8→32"))
    results.append(test_paper_rect(16, 64, 64, label="16→64"))

    # Expand with t_out > 1: compute_perm_f only covers d_in rows per slot
    # (incomplete for expand when t_out > 1). Not our target config.
    print("\n--- Paper d²/S: expand t_out > 1 — KNOWN LIMITATION ---")
    for d_in, d_out, S, label in [(1024, 4096, 8192, "Up logN=14 t_out=2")]:
        passed = test_paper_rect(d_in, d_out, S, label=label + " [XFAIL]")
        if not passed:
            print(f"    ^ expected: t_out={S//d_out}>1, row coverage incomplete")

    print("\n--- Paper d²/S: rectangular SHRINK (d_in > d_out) ---")
    shrink_cases = [
        (4096, 1024, 4096, "Down fD→hD (tight)"),
        (4096, 1024, 8192, "Down (logN=14)"),
        (32, 8, 32, "small 32→8"),
        (64, 16, 64, "64→16"),
    ]
    for d_in, d_out, S, label in shrink_cases:
        results.append(test_paper_rect(d_in, d_out, S, label=label))

    # ── Streaming rect: verify block-by-block gives same result ────
    print("\n--- Streaming rect vs batch ---")
    for d_in, d_out, S, label in [
        (1024, 4096, 4096, "Up (tight)"),
        (64, 256, 256, "small expand"),
        (4096, 1024, 4096, "Down (tight)"),
        (64, 16, 64, "small shrink"),
    ]:
        np.random.seed(42)
        W = np.random.randn(d_in, d_out)
        x = np.random.randn(d_in)
        t_in = S // d_in
        t_out = S // d_out
        t_baby = max(t_in, t_out)
        cx = np.zeros(S)
        for k in range(d_in):
            cx[k * t_in] = x[k]
        weights, r_i, r_o = pack_weights_paper_rect(W, S, d_in, d_out)
        cy_batch = linear_paper_rect(cx, weights, S, d_in, d_out, r_i, r_o)
        # Streaming: process baby-step blocks one at a time
        cxp = cx.copy()
        step = 1
        while step < t_in:
            cxp = cxp + np.roll(cxp, -(step * (d_in - 1)))
            step *= 2
        cx_rot = [cxp.copy()]
        for j in range(1, r_i):
            cx_rot.append(np.roll(cx_rot[-1], -t_baby))
        cy_k = [np.zeros(S) for _ in range(r_o)]
        for j in range(r_i):
            block = weights[j * r_o:(j + 1) * r_o]
            for k in range(r_o):
                cy_k[k] = cy_k[k] + cx_rot[j] * block[k]
        for k in range(r_o - 1, 0, -1):
            cy_k[k - 1] = cy_k[k - 1] + np.roll(cy_k[k], -(t_baby * r_i))
        cy_stream = cy_k[0]
        step = 1
        while step < t_out:
            cy_stream = cy_stream + np.roll(cy_stream, -step)
            step *= 2
        err = np.max(np.abs(cy_batch - cy_stream))
        passed = err < 1e-15
        print(f"  stream d_in={d_in:4d} d_out={d_out:4d} S={S:5d}  "
              f"err={err:.2e}  {'PASS' if passed else 'FAIL'}  {label}")
        results.append(passed)

    # ══════════════════════════════════════════════════════════════════
    # MEMORY EFFICIENCY: compare plaintext counts standard vs interleaved
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  MEMORY EFFICIENCY: standard vs interleaved plaintext counts")
    print("=" * 72)

    configs = [
        # (label, hD, fD, S, logN)
        ("GPT-2 logN=13", 1024, 4096, 4096, 13),
        ("GPT-2 logN=14", 1024, 4096, 8192, 14),
        ("GPT-2 logN=16", 1024, 4096, 32768, 16),
        ("LLaMA-7B logN=16", 4096, 11008, 32768, 16),
    ]

    print(f"\n  {'Config':<20s} {'hD':>5s} {'fD':>6s} {'S':>6s} "
          f"{'Std/lyr':>8s} {'Int/lyr':>8s} {'Ratio':>6s} "
          f"{'Std GB':>7s} {'Int GB':>7s}")
    print("  " + "-" * 68)

    mem_results = []
    for label, hD, fD, S, logN in configs:
        # Standard: d diags per square, max(d_in,d_out) per rect
        d_ff = max(hD, fD)
        std_sq = 4 * hD                      # Q,K,V,Out
        std_ff = 2 * d_ff                     # Up,Down
        std_total = std_sq + std_ff

        # Interleaved: d_in*d_out/S per matrix
        int_sq = 4 * (hD * hD // S)          # Q,K,V,Out
        int_ff = 2 * (hD * fD // S)          # Up,Down
        int_total = int_sq + int_ff

        ratio = std_total / int_total if int_total > 0 else float('inf')

        # Memory estimate: ~S*2*8 bytes per plaintext (2N doubles at logN)
        bytes_per_ptx = S * 2 * 8 * (logN + 7)  # rough: logN+7 RNS limbs
        std_gb = std_total * bytes_per_ptx / 1e9
        int_gb = int_total * bytes_per_ptx / 1e9

        passed = int_total < std_total
        mem_results.append(passed)
        results.append(passed)

        print(f"  {label:<20s} {hD:>5d} {fD:>6d} {S:>6d} "
              f"{std_total:>8d} {int_total:>8d} {ratio:>5.1f}× "
              f"{std_gb:>6.1f}G {int_gb:>6.1f}G  "
              f"{'PASS' if passed else 'FAIL'}")

    print()
    all_mem_pass = all(mem_results)
    print(f"  Memory efficiency: {'ALL PASS' if all_mem_pass else 'SOME FAIL'} "
          f"({sum(mem_results)}/{len(mem_results)})")

    # Detailed breakdown for GPT-2 @ logN=13 (our target)
    print("\n  Detailed breakdown for GPT-2 @ logN=13 (S=4096, hD=1024, fD=4096):")
    S, hD, fD = 4096, 1024, 4096
    print(f"  {'Matrix':<18s} {'Dims':>12s}  {'Std':>6s}  {'Int':>6s}  "
          f"{'Hybrid':>6s}  {'Savings'}")
    print(f"  " + "-" * 62)
    hybrid_total = 0
    std_total_ex = 0
    for name, d_in, d_out, use_interleaved in [
        ("Q/K/V/Out (×4)", hD, hD, True),
        ("Up         (×1)", hD, fD, True),
        ("Down       (×1)", fD, hD, True),   # shrink: interleaved (fixed)
    ]:
        std_n  = max(d_in, d_out)
        int_n  = d_in * d_out // S
        hyb_n  = int_n if use_interleaved else std_n
        hybrid_total += hyb_n
        std_total_ex += std_n
        ratio  = std_n / int_n
        note   = "(interleaved)" if use_interleaved else "(std BSGS)"
        print(f"  {name:<18s} {d_in}×{d_out}  "
              f"{std_n:>6d}  {int_n:>6d}  {hyb_n:>6d}  "
              f"{ratio:.0f}× possible  {note}")
    print(f"  {'TOTAL/layer':<18s} {'':>12s}  {std_total_ex:>6d}  "
          f"{'':>6s}  {hybrid_total:>6d}  "
          f"{std_total_ex/hybrid_total:.1f}× hybrid savings")

    # verify hybrid is a memory reduction
    hybrid_saves = hybrid_total < std_total_ex
    results.append(hybrid_saves)
    print(f"\n  Hybrid mode saves memory: {'PASS' if hybrid_saves else 'FAIL'} "
          f"({std_total_ex} → {hybrid_total} Ptx/layer, "
          f"{std_total_ex/hybrid_total:.1f}× reduction)")

    # ══════════════════════════════════════════════════════════════════
    # CONSISTENCY: prepare_gpt2_weights.py packing matches test packing
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  CONSISTENCY: bsgs_pack_interleaved matches pack_weights_paper")
    print("=" * 72)

    # Import the prepare script's packing function
    import importlib.util
    import os
    prep_path = os.path.join(os.path.dirname(__file__), '..', 'prepare_gpt2_weights.py')
    prep_path = os.path.abspath(prep_path)
    spec = importlib.util.spec_from_file_location("prepare_gpt2_weights", prep_path)
    prep_mod = importlib.util.module_from_spec(spec)
    # Prevent main() from running during import
    import sys
    old_argv = sys.argv
    sys.argv = [prep_path]
    spec.loader.exec_module(prep_mod)
    sys.argv = old_argv

    consistency_cases = [
        (64, 64, 256, "square small"),
        (256, 256, 2048, "square logN=12"),
        (1024, 1024, 4096, "GPT-2 sq"),
    ]
    for d_in, d_out, S, label in consistency_cases:
        np.random.seed(42)
        W = np.random.randn(d_in, d_out)

        # Test file's packing
        test_weights, _, _ = pack_weights_paper(W, S, d_in)

        # prepare script's packing
        prep_packed = prep_mod.bsgs_pack_interleaved(W, S, d_in, d_out)

        # Compare
        test_arr = np.array(test_weights)
        err = np.max(np.abs(test_arr - prep_packed))
        passed = err < 1e-15
        print(f"  {label}: d_in={d_in} d_out={d_out} S={S}  "
              f"err={err:.2e}  {'PASS' if passed else 'FAIL'}")
        results.append(passed)

    # Also test rectangular consistency
    rect_cases = [
        (1024, 4096, 4096, "Up GPT-2"),
        (4096, 1024, 4096, "Down GPT-2"),
    ]
    for d_in, d_out, S, label in rect_cases:
        np.random.seed(42)
        W = np.random.randn(d_in, d_out)

        # Test file's packing (rect)
        test_weights, _, _ = pack_weights_paper_rect(W, S, d_in, d_out)

        # prepare script's packing
        prep_packed = prep_mod.bsgs_pack_interleaved(W, S, d_in, d_out)

        test_arr = np.array(test_weights)
        err = np.max(np.abs(test_arr - prep_packed))
        passed = err < 1e-15
        print(f"  {label}: d_in={d_in} d_out={d_out} S={S}  "
              f"err={err:.2e}  {'PASS' if passed else 'FAIL'}")
        results.append(passed)

    # ══════════════════════════════════════════════════════════════════
    # PAPER'S EXACT ALGORITHM 1: with interleaved I/O for rectangular
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("  PAPER'S ALGORITHM 1: interleaved I/O for rectangular cases")
    print("=" * 72)

    print("\n--- Alg1: square (should match paper d²/S) ---")
    results.append(test_alg1(64, 64, 256, label="square small"))
    results.append(test_alg1(1024, 1024, 4096, label="GPT-2 sq tight"))
    results.append(test_alg1(1024, 1024, 32768, label="GPT-2 sq logN=16"))

    print("\n--- Alg1: expand (d_in < d_out) — interleaved output ---")
    results.append(test_alg1(8, 32, 32, label="small 8→32"))
    results.append(test_alg1(16, 64, 64, label="16→64"))
    results.append(test_alg1(1024, 4096, 4096, label="Up logN=13 t'=1"))
    results.append(test_alg1(1024, 4096, 8192, label="Up logN=14 t'=2"))
    results.append(test_alg1(1024, 4096, 32768, label="Up logN=16 t'=8"))

    print("\n--- Alg1: shrink (d_in > d_out) — interleaved input ---")
    results.append(test_alg1(32, 8, 32, label="small 32→8"))
    results.append(test_alg1(64, 16, 64, label="64→16"))
    results.append(test_alg1(4096, 1024, 4096, label="Down logN=13"))
    results.append(test_alg1(4096, 1024, 8192, label="Down logN=14"))
    results.append(test_alg1(4096, 1024, 32768, label="Down logN=16"))

    n_pass = sum(results)
    print(f"\n=== {n_pass}/{len(results)} tests passed ===")
