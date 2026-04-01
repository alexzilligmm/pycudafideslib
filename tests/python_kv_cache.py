import numpy as np
from math import log2


def rot(v, k):
    """Cyclic rotation: rot(v, k)[i] = v[(i+k) % len(v)]."""
    return np.roll(v, -k)

def compute_params(N, d, alpha, is_up):
    t  = N // d
    tp = N // (alpha * d)
    tp_in  = t  if is_up else tp
    tp_out = tp if is_up else t
    d_in   = d if is_up else alpha * d
    n_pt   = d_in // tp_out
    r_i    = max(1, d * d // N)
    r_i    = min(r_i, n_pt)
    r_o    = n_pt // r_i
    return t, tp, tp_in, tp_out, r_i, r_o, n_pt


def interleave_idx(m, d, dim):
    a = dim // d if dim > d else 1
    return (m // a + (m % a) * d) % dim


def encode_x(x, N, d, alpha, is_up):
    t, tp, _, _, _, _, _ = compute_params(N, d, alpha, is_up)
    d_in = d if is_up else alpha * d
    M    = N // tp
    ptx  = np.zeros(N)
    if is_up:
        for i in range(d):
            ptx[i * t] = x[i]
    else:
        for m in range(M):
            ptx[m * tp] = x[interleave_idx(m, d, d_in)]
    return ptx


def encode_W(W, N, d, alpha, is_up):
    d_in, d_out = W.shape
    t, tp, tp_in, tp_out, r_i, r_o, n_pt = compute_params(N, d, alpha, is_up)
    M_out = N // tp_out
    cascade_shift = (t * tp) // tp_out
    pt = np.zeros((n_pt, N))
    for j in range(r_i):
        for k in range(r_o):
            for i in range(N):
                row = ((i // t + j * t + i % tp_in) % d) \
                    + ((i % t) // tp_in) * d
                m_shifted = (i // tp_out - k * cascade_shift) % M_out
                col = interleave_idx(m_shifted, d, d_out)
                pt[j * r_o + k, i] = W[row, col]
    return pt


def decode_output(cy, N, t, tp, alpha, d, d_out, is_up):
    M = N // tp
    if is_up and alpha > 1:
        y = np.zeros(d_out)
        for m in range(M):
            idx = interleave_idx(m, d, d_out)
            if idx < d_out:
                y[idx] = cy[m * tp]
    else:
        y = cy[::t][:d_out]
    return y


def cachemir_vmm_raw(x, W, N):
    """VMM returning raw N-slot ciphertext (before decode/masking)."""
    d_in, d_out = W.shape
    assert len(x) == d_in
    if d_in <= d_out:
        d = d_in;  alpha = d_out // d;  is_up = True
    else:
        d = d_out; alpha = d_in  // d;  is_up = False
    assert alpha * d == max(d_in, d_out)
    assert N % d == 0 and alpha * d * d >= N
    t, tp, tp_in, tp_out, r_i, r_o, n_pt = compute_params(N, d, alpha, is_up)
    ptx   = encode_x(x, N, d, alpha, is_up)
    pts_W = encode_W(W, N, d, alpha, is_up)
    ptx_prime = ptx.copy()
    step = 1
    while step < tp_in:
        ptx_prime = ptx_prime + rot(ptx_prime, step * (t - 1))
        step *= 2
    rot2 = t * t
    cy_prime    = np.zeros((r_o, N))
    ptx_rotated = [rot(ptx_prime, j * rot2) for j in range(r_i)]
    for k in range(r_o):
        for j in range(r_i):
            cy_prime[k] += ptx_rotated[j] * pts_W[j * r_o + k]
    cascade_rot = t * tp
    for k in range(r_o - 1, 0, -1):
        cy_prime[k - 1] += rot(cy_prime[k], cascade_rot)
    cy = cy_prime[0]
    step = 1
    while step < tp_out:
        cy = cy + rot(cy, step)
        step *= 2
    return cy


def mask_interleaved(cy, N, d):
    t = N // d
    mask = np.zeros(N)
    mask[::t] = 1.0
    return cy * mask


def _pos(local_dim, head, tok, t, H):
    """Ciphertext slot index for a (local_dim, head, token) triple."""
    return local_dim * t * H + head * t + tok


def encode_q_interleaved(q, N, d, H=1):
    """
    Encode one query vector into head-interleaved N-slot format.
    Zeros in all token slots except tok=0.
    """
    t      = N // d
    d_head = d // H
    ct     = np.zeros(N)
    for h in range(H):
        for ld in range(d_head):
            ct[_pos(ld, h, 0, t, H)] = q[h * d_head + ld]
    return ct


def encode_kcache_group(keys, N, d, H=1):
    """
    Pack up to t key vectors into one head-interleaved K cache ciphertext.
    For H=1 reduces to: ct[dim * t + tok] = keys[tok][dim]
    """
    t      = N // d
    d_head = d // H
    assert len(keys) <= t, f"group has {len(keys)} keys but t={t}"
    ct = np.zeros(N)
    for tok_idx, k_vec in enumerate(keys):
        for h in range(H):
            for ld in range(d_head):
                ct[_pos(ld, h, tok_idx, t, H)] = k_vec[h * d_head + ld]
    return ct

def qkt_single_group(cq_rep, kcache_ct, N, d, H, num_tokens):
    t = N // d
    result = cq_rep * kcache_ct
    result = _accumulate_dims(result, N, d, H)
    return result * _group_mask(num_tokens, t, H, N)

class KCache:
    """Helper for building K cache ciphertexts from a key matrix."""
    def __init__(self, N, d, K=None, H=1):
        self.curr_keys = 0
        self.N = N
        self.d = d
        self.H = H
        self.groups = []
        if K is not None:
            self._build(K)
        

    def _build(self, K):
        """Build K cache ciphertexts from (n_k, d) matrix. Returns list."""
        t = self.N // self.d
        self.curr_keys = K.shape[0]
        self.groups = [
            encode_kcache_group(
                [K[i] for i in range(g * t, min((g + 1) * t, self.curr_keys))],
                self.N, self.d, self.H
            )
            for g in range((self.curr_keys + t - 1) // t)
        ]

    def push_back(self, key):
        if self.curr_keys % (self.N // self.d) == 0: # New group needed
            # Start new group
            self.groups.append(encode_kcache_group([key], self.N, self.d, self.H))
        else:
            # Add to current group
            g = len(self.groups) - 1
            tok_idx = self.curr_keys % (self.N // self.d)
            for h in range(self.H):
                for ld in range(self.d // self.H):
                    self.groups[g][_pos(ld, h, tok_idx, self.N // self.d, self.H)] = key[h * (self.d // self.H) + ld]
        self.curr_keys += 1

    def _qkt_single(self, q):
        """
        Compute QK^T for a single query vector against this cache.

        Args:
            q: shape (d,)

        Returns:
            ct_windows: list of N-slot ciphertext windows
        """
        assert q.ndim == 1, "q must be a single vector"
        assert q.shape[0] == self.d, f"q has {q.shape[0]} cols, expected d={self.d}"
        assert self.d % self.H == 0, f"d={self.d} not divisible by H={self.H}"
        assert self.N % self.d == 0, f"N={self.N} not divisible by d={self.d}"

        n_k = self.curr_keys
        t = self.N // self.d
        cap_keys = self.N // self.H

        cq = encode_q_interleaved(q, self.N, self.d, self.H)
        cq_rep = preprocess_q(cq, self.N, self.d)

        groups_per_window = max(1, cap_keys // t)
        ct_windows = []

        for g0 in range(0, len(self.groups), groups_per_window):
            g1 = min(g0 + groups_per_window, len(self.groups))
            attn_ct = np.zeros(self.N)
            for local_g, g in enumerate(range(g0, g1)):
                start = g * t
                num_tok = min(t, n_k - start)
                partial = qkt_single_group(cq_rep, self.groups[g], self.N, self.d, self.H, num_tok)
                attn_ct += rot(partial, -(local_g * t * self.H))

            ct_windows.append(attn_ct)

        return ct_windows

    def qkt(self, Q):
        """
        Compute QK^T for one or multiple query vectors against this cache.

        Args:
            Q: shape (d,) or (n_q, d)

        Returns:
            raw_cts: list of per-query ciphertext-window lists
        """
        if Q.ndim == 1:
            Q = Q.reshape(1, -1)

        n_q = Q.shape[0]
        assert Q.shape[1] == self.d, f"Q has {Q.shape[1]} cols, expected d={self.d}"

        raw_cts = []

        for qi in range(n_q):
            ct_windows = self._qkt_single(Q[qi])
            raw_cts.append(ct_windows)

        return raw_cts


def preprocess_q(cq, N, d):
    t   = N // d
    out = cq.copy()
    step = 1
    while step < t:
        out = out + rot(out, -step)
        step *= 2
    return out


def _accumulate_dims(result, N, d, H):
    t = N // d
    s = t * H
    while s < N:
        result = result + rot(result, s)
        s *= 2
    return result


def _group_mask(num_tokens, t, H, N):
    mask = np.zeros(N)
    for p in range(t * H):
        if p % t < num_tokens:
            mask[p] = 1.0
    return mask


def decode_qkt(raw_cts, n_k, N, d, H):
    """Decode qkt ciphertext-window output into (n_q, H, n_k) scores."""
    n_q = len(raw_cts)
    cap_keys = N // H
    scores = np.zeros((n_q, H, n_k))
    for qi, ct_windows in enumerate(raw_cts):
        out_start = 0
        for ct in ct_windows:
            chunk_keys = min(cap_keys, n_k - out_start)
            decoded_chunk = decode_attn_scores([ct], chunk_keys, N, d, H)[0]
            scores[qi, :, out_start:out_start + chunk_keys] = decoded_chunk
            out_start += chunk_keys
    return scores


def decode_attn_scores(raw_cts, n_k, N, d, H):
    """
    Extract per-head attention scores from packed ciphertexts.

    Output layout is head-interleaved in t*H blocks:
      [h0(t), h1(t) | h0(t), h1(t) | ...]
    Token i, head h -> position (i // t) * t * H + h * t + (i % t)

    For H=1: position = i (contiguous).
    Returns (n_q, H, n_k).
    """
    t   = N // d
    n_q = len(raw_cts)
    scores = np.zeros((n_q, H, n_k))
    for qi, ct in enumerate(raw_cts):
        for h in range(H):
            for i in range(n_k):
                pos = (i // t) * t * H + h * t + (i % t)
                scores[qi, h, i] = ct[pos]
    return scores


def _print_group_interleaving(ct, N, d, H, num_tokens, label):
    """Print one packed KCache group to inspect raw slot interleaving."""
    print(f"  {label}")
    print(f"    raw slots: {ct}")


def inner_rot(v, s, t, N):
    """
    Rotate token index by s within every t-sized block.
    v[block + tok] -> v[block + (tok + s) % t]

    A global rot(v, s) would cross block boundaries.
    """
    out = np.zeros(N)
    for i in range(N):
        blk = (i // t) * t
        tok = i % t
        out[i] = v[blk + (tok + s) % t]
    return out


def qkt_batched(Q, K, N, d, H=1):
    """
    Batched prefill QK^T: t queries per ciphertext, inner rotations
    to compute all token cross-products per (Q_group, K_group) pair.

    Same interface and output shape as qkt().
    """
    if Q.ndim == 1:
        Q = Q.reshape(1, -1)

    n_q, n_k, t = Q.shape[0], K.shape[0], N // d

    assert Q.shape[1] == d and K.shape[1] == d
    assert d % H == 0 and N % d == 0
    assert n_k <= N // H

    kcache_cts = KCache(N, d, K, H)
    scores     = np.zeros((n_q, H, n_k))

    n_q_groups = (n_q + t - 1) // t

    for qg in range(n_q_groups):
        q_start = qg * t
        q_end   = min(q_start + t, n_q)
        num_q   = q_end - q_start

        q_group_cache = KCache(N, d, Q[q_start:q_end], H)
        cq_group = q_group_cache.groups[0]

        for kg, ck in enumerate(kcache_cts.groups):
            k_start = kg * t
            k_end   = min(k_start + t, n_k)
            num_k   = k_end - k_start

            for s in range(t):
                cq_shifted = inner_rot(cq_group, s, t, N)
                result     = _accumulate_dims(cq_shifted * ck, N, d, H)

                for tok_k in range(num_k):
                    tok_q_local = (tok_k + s) % t
                    if tok_q_local < num_q:
                        qi = q_start + tok_q_local
                        ki = k_start + tok_k
                        for h in range(H):
                            scores[qi, h, ki] = result[_pos(0, h, tok_k, t, H)]

    return scores


def _gt(Q, K, H):
    """Per-head ground truth QK^T: (n_q, H, n_k)."""
    if Q.ndim == 1:
        Q = Q.reshape(1, -1)
    d_head = Q.shape[1] // H
    out    = np.zeros((Q.shape[0], H, K.shape[0]))
    for h in range(H):
        sl = slice(h * d_head, (h + 1) * d_head)
        out[:, h, :] = Q[:, sl] @ K[:, sl].T
    return out


def _run(label, Q, K, N, d, H, batched=False):
    gt = _gt(Q, K, H)
    kcache = KCache(N, d, K, H)
    if batched:
        scores = qkt_batched(Q, K, N, d, H)
    else:
        raw_cts = kcache.qkt(Q)
        scores = decode_qkt(raw_cts, K.shape[0], N, d, H)
    err = np.max(np.abs(gt - scores))
    tag = "OK" if err < 1e-10 else "FAIL"
    nq  = 1 if Q.ndim == 1 else Q.shape[0]
    nk  = K.shape[0]
    t   = N // d
    print(f"  {label:12s} N={N:4d} d={d:2d} H={H} t={t:2d} "
          f"nq={nq:2d} nk={nk:3d}: {err:.2e} {tag}")
    return err < 1e-10


def _same_kcache_layout(a, b):
    if a.curr_keys != b.curr_keys or len(a.groups) != len(b.groups):
        return False
    return all(np.allclose(ga, gb) for ga, gb in zip(a.groups, b.groups))


def _run_kcache_path_test(label, Q, K_base, K_new, N, d, H):
    K_full = np.vstack([K_base, K_new]) if K_new.shape[0] > 0 else K_base.copy()

    cache_full = KCache(N, d, K_full, H)
    cache_inc  = KCache(N, d, K_base, H)
    for i in range(K_new.shape[0]):
        cache_inc.push_back(K_new[i])

    raw_full = cache_full.qkt(Q)
    raw_inc  = cache_inc.qkt(Q)
    scores_full = decode_qkt(raw_full, K_full.shape[0], N, d, H)
    scores_inc  = decode_qkt(raw_inc, K_full.shape[0], N, d, H)
    gt = _gt(Q, K_full, H)

    err_full = np.max(np.abs(gt - scores_full))
    err_inc  = np.max(np.abs(gt - scores_inc))
    err_diff = np.max(np.abs(scores_full - scores_inc))
    same_layout = _same_kcache_layout(cache_full, cache_inc)

    ok = (err_full < 1e-10) and (err_inc < 1e-10) and (err_diff < 1e-10) and same_layout
    tag = "OK" if ok else "FAIL"
    nq  = 1 if Q.ndim == 1 else Q.shape[0]
    nk  = K_full.shape[0]
    print(
        f"  {label:12s} N={N:4d} d={d:2d} H={H} nq={nq:2d} nk={nk:3d} "
        f"(+{K_new.shape[0]}): full={err_full:.2e} inc={err_inc:.2e} "
        f"delta={err_diff:.2e} layout={'OK' if same_layout else 'FAIL'} {tag}"
    )
    return ok



def cachemir_vmm(x, W, N):
    d_in, d_out = W.shape
    assert len(x) == d_in

    if d_in <= d_out:
        d = d_in;  alpha = d_out // d;  is_up = True
    else:
        d = d_out; alpha = d_in  // d;  is_up = False
    assert alpha * d == max(d_in, d_out)
    assert N % d == 0 and alpha * d * d >= N

    t, tp, tp_in, tp_out, r_i, r_o, n_pt = compute_params(N, d, alpha, is_up)

    ptx = encode_x(x, N, d, alpha, is_up)
    pts_W = encode_W(W, N, d, alpha, is_up)

    ptx_prime = ptx.copy()
    step = 1
    while step < tp_in:
        ptx_prime = ptx_prime + rot(ptx_prime, step * (t - 1))
        step *= 2

    rot2 = t * t
    cy_prime = np.zeros((r_o, N))

    ptx_rotated = [rot(ptx_prime, j * rot2) for j in range(r_i)]
                 
    for k in range(r_o):
        for j in range(r_i):
            cy_prime[k] += ptx_rotated[j] * pts_W[j * r_o + k]

    cascade_rot = t * tp
    for k in range(r_o - 1, 0, -1):
        cy_prime[k - 1] += rot(cy_prime[k], cascade_rot)

    cy = cy_prime[0]

    step = 1
    while step < tp_out:
        cy = cy + rot(cy, step)
        step *= 2
    
    return decode_output(cy, N, t, tp, alpha, d, d_out, is_up)


class VCache:
    """Helper for building V cache ciphertexts from a value matrix."""
    def __init__(self, N, d, V=None):
        self.curr_values = 0
        self.N = N
        self.d = d
        self.values = None
        if V is not None:
            self._build(V)

    def _build(self, V):
        n_v = V.shape[0]
        n_metagroups = (n_v // self.N) + (n_v % self.N > 0)
        self.values = []
        for mg in range(n_metagroups):
            metagroup = [np.zeros(self.N) for _ in range(self.d)]
            start = mg * self.N
            end = min(start + self.N, n_v)
            for i in range(start, end):
                for j in range(self.d):
                    #metagroup[something][something] = V[i + start, j] Something
                    raise NotImplementedError("VCache building not implemented yet")
            self.values.append(metagroup)



def main():
    ok = True

    print("=== VMM sanity ===")
    for lab, N, di, do in [("sq", 32, 8, 8), ("up", 128, 16, 64), ("dn", 128, 64, 16)]:
        x = np.random.random(di); W = np.random.random((di, do))
        e = np.max(np.abs(x @ W - cachemir_vmm(x, W, N)))
        t = "OK" if e < 1e-10 else "FAIL"
        print(f"  {lab:3s} ({di}→{do}): {e:.2e} {t}")
        ok &= (e < 1e-10)

    # ── Single-head QK^T ──
    print("\n=== QK^T single-head decoding (1 query) ===")
    for lab, N, d, nk in [("toy",8,4,5),("exact",8,4,4),("1tok",8,4,1),("med",32,8,10),("big",128,16,20)]:
        np.random.seed(42)
        ok &= _run(lab, np.random.randn(d), np.random.randn(nk, d), N, d, 1)

    print("\n=== QK^T single-head classification (n queries) ===")
    for lab, N, d, n in [("tiny",8,4,3),("small",32,8,6),("med",64,8,15)]:
        np.random.seed(42)
        ok &= _run(lab, np.random.randn(n, d), np.random.randn(n, d), N, d, 1)

    # ── Multi-head QK^T ──
    print("\n=== QK^T multi-head decoding ===")
    for lab, N, d, H, nk in [("2h",8,4,2,3),("4h",64,8,4,10),("8h",128,16,8,15)]:
        np.random.seed(42)
        ok &= _run(lab, np.random.randn(d), np.random.randn(nk, d), N, d, H)

    print("\n=== QK^T multi-head classification ===")
    for lab, N, d, H, n in [("2h",8,4,2,3),("4h",64,8,4,12),("8h",128,16,8,8)]:
        np.random.seed(42)
        ok &= _run(lab, np.random.randn(n, d), np.random.randn(n, d), N, d, H)

    # ── Batched prefill (inner rotations) ──
    print("\n=== QK^T batched prefill ===")
    for lab, N, d, H, nq, nk in [
        ("1h",8,4,1,3,3),("2h",8,4,2,3,3),("4h",64,8,4,12,12),
        ("ragged",64,8,4,7,10),("8h",128,16,8,8,15)
    ]:
        np.random.seed(42)
        ok &= _run(lab, np.random.randn(nq, d), np.random.randn(nk, d), N, d, H, batched=True)

    # ── KCache construction paths ──
    print("\n=== KCache build paths (full vs incremental add) ===")
    np.random.seed(42)
    N, d, H = 128, 16, 8

    Q = np.random.randn(5, d)
    K_full = np.random.randn(11, d)
    cache_full = KCache(N, d, K_full, H)
    raw_cache = cache_full.qkt(Q)
    scores_cache = decode_qkt(raw_cache, K_full.shape[0], N, d, H)
    gt_full = _gt(Q, K_full, H)
    err_full_build = np.max(np.abs(gt_full - scores_cache))
    tag = "OK" if err_full_build < 1e-10 else "FAIL"
    print(f"  {'whole-k':12s} N={N:4d} d={d:2d} H={H} nq={Q.shape[0]:2d} nk={K_full.shape[0]:3d}: {err_full_build:.2e} {tag}")
    ok &= (err_full_build < 1e-10)

    K_base = np.random.randn(8, d)
    K_more = np.random.randn(4, d)
    ok &= _run_kcache_path_test("add-1", Q, K_base, K_more[:1], N, d, H)
    ok &= _run_kcache_path_test("add-many", Q, K_base, K_more, N, d, H)


    print("\n=== Visual interleaving demo (incremental add) ===")
    N, d, H = 8, 4, 1
    t = N // d
    n_demo = t + 2  # force spill into a second KCache group

    K_demo = np.vstack([np.full((d,), i + 1.0) for i in range(n_demo)])
    print(f"  params: N={N}, d={d}, H={H}, t={t}, n_keys={n_demo}")
    print("  input keys (row i is filled with i+1):")
    print(f"  {K_demo}")
    kcache_inc = KCache(N, d, K_demo[:-1], H)
    print(f"  before add: n_keys={kcache_inc.curr_keys}")
    for g, ct in enumerate(kcache_inc.groups):
        start = g * t
        num_tok = min(t, kcache_inc.curr_keys - start)
        _print_group_interleaving(ct, N, d, H, num_tok, f"group {g} (keys {start}..{start + num_tok - 1})")

    kcache_inc.push_back(K_demo[-1])
    print(f"  after add : n_keys={kcache_inc.curr_keys} (added key value {n_demo})")
    for g, ct in enumerate(kcache_inc.groups):
        start = g * t
        num_tok = min(t, kcache_inc.curr_keys - start)
        _print_group_interleaving(ct, N, d, H, num_tok, f"group {g} (keys {start}..{start + num_tok - 1})")

    print("\n=== QK^T with n_k > N (windowed from KCache) ===")
    np.random.seed(42)
    N, d, H = 64, 8, 1
    q_big = np.random.randn(d)
    n_k_big = N + 9
    K_big = np.random.randn(n_k_big, d)
    kcache_big = KCache(N, d, K_big, H)

    raw_big = kcache_big.qkt(q_big)
    scores_big = decode_qkt(raw_big, n_k_big, N, d, H)
    gt_big = _gt(q_big, K_big, H)
    err_big = np.max(np.abs(scores_big - gt_big))
    tag = "OK" if err_big < 1e-10 else "FAIL"
    print(f"  N={N} d={d} H={H} n_k={n_k_big} (>N): {err_big:.2e} {tag}")
    ok &= (err_big < 1e-10)

    # ── Trace ──
    print("\n=== Trace: N=8, d=4, H=2, t=2, nk=3 ===")
    N, d, H, t = 8, 4, 2, 2
    q  = np.array([1., 2., 3., 4.])
    Km = np.array([[.1,.2,.3,.4],[.5,.6,.7,.8],[.9,1.,1.1,1.2]])
    gt = _gt(q, Km, H)

    print(f"q = {q}  →  h0:[1,2]  h1:[3,4]")
    for i, ki in enumerate(Km):
        print(f"K{i} = {ki}")
    print(f"GT h0: {gt[0,0]}")
    print(f"GT h1: {gt[0,1]}")

    cq     = encode_q_interleaved(q, N, d, H)
    cq_rep = preprocess_q(cq, N, d)
    print(f"\ncq       = {cq}")
    print(f"cq_rep   = {cq_rep}")

    kcache_trace_01 = KCache(N, d, Km[:2], H)
    ck01 = kcache_trace_01.groups[0]
    print(f"K[0,1]   = {ck01}")

    prod = cq_rep * ck01
    acc  = _accumulate_dims(prod, N, d, H)
    r0   = acc * _group_mask(2, t, H, N)
    print(f"group0   = {r0}")

    kcache_trace_2 = KCache(N, d, Km[2:3], H)
    ck2 = kcache_trace_2.groups[0]
    r1  = _accumulate_dims(cq_rep * ck2, N, d, H) * _group_mask(1, t, H, N)
    r1s = rot(r1, -(1 * t * H))
    print(f"group1→  = {r1s}")

    final = r0 + r1s
    print(f"final    = {final}")
    # Decode using position formula: (i//t)*t*H + h*t + i%t
    for h in range(H):
        vals = [final[(i // t) * t * H + h * t + i % t] for i in range(3)]
        print(f"h{h}: {vals} == {gt[0,h].tolist()} ? {np.allclose(vals, gt[0,h])}")

    print(f"\n{'ALL PASSED' if ok else 'SOME FAILURES'}")


if __name__ == "__main__":
    main()