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

def qkt_single_group(cq_rep, kcache_ct, N, d, H, group_idx, num_tok):
    result = cq_rep * kcache_ct # elementwise product in N-slot
    result = _sum_by_rot(result, N, d, H)
    mask = _qkt_mask(group_idx, num_tok, N, d, H)
    return result * mask

class KCache:
    """Helper for building K cache ciphertexts from a key matrix."""
    def __init__(self, N, d, K=None, H=1):
        self.curr_keys = 0
        self.N = N
        self.d = d
        self.H = H
        self.ciphertexts = []
        if K is not None:
            self._build(K)
        

    def _build(self, K):
        """Build K cache ciphertexts from (n_k, d) matrix. Returns list."""
        t = self.N // self.d
        self.curr_keys = K.shape[0]
        self.ciphertexts = [
            encode_kcache_group(
                [K[i] for i in range(g * t, min((g + 1) * t, self.curr_keys))],
                self.N, self.d, self.H
            )
            for g in range((self.curr_keys + t - 1) // t)
        ]

    def push_back(self, key):
        if self.curr_keys % (self.N // self.d) == 0: # New group needed
            # Start new group
            self.ciphertexts.append(encode_kcache_group([key], self.N, self.d, self.H))
        else:
            # Add to current group
            g = len(self.ciphertexts) - 1
            tok_idx = self.curr_keys % (self.N // self.d)
            for h in range(self.H):
                for ld in range(self.d // self.H):
                    self.ciphertexts[g][_pos(ld, h, tok_idx, self.N // self.d, self.H)] = key[h * (self.d // self.H) + ld]
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

        keys_per_ct = self.N // self.d

        cq = encode_q_interleaved(q, self.N, self.d, self.H)
        cq_rep = preprocess_q(cq, self.N, self.d)


        attn_ct = np.zeros(self.N)
        for global_ct_indx in range(len(self.ciphertexts)):
            first_local_key = global_ct_indx * keys_per_ct
            num_tok = min(keys_per_ct, self.curr_keys - first_local_key)
            partial = qkt_single_group(cq_rep, self.ciphertexts[global_ct_indx], self.N, self.d, self.H, global_ct_indx, num_tok)
            # part = *, *,a3,a4 | *,*,a3,a4

            # r = a1, a2, a
            
            attn_ct += partial

        # # attn = a1, a2, *, * | a1, a2, *, *| a1, a2, *, *
        # for h in range(self.H):
        #     mask = ';[;ld]'
        #     attn = attn * (1-mask) + rot(attn) * mask

        return attn_ct

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
            pre_softmax_attn = self._qkt_single(Q[qi])
            raw_cts.append(pre_softmax_attn)

        raw_cts = np.stack(raw_cts, axis=0) # shape (n_q, N)

        return raw_cts


def preprocess_q(cq, N, d):
    t   = N // d
    out = cq.copy()
    step = 1
    while step < t:
        out = out + rot(out, -step)
        step *= 2
    return out


# 1,2,3,1,2,3,1,2,3,1,2,3 | ....
# 


def _sum_by_rot(result, N, d, H):
    s = (N // d) * H
    while s < N:
        result = result + rot(result, s)
        s *= 2
    return result


def _qkt_mask(group_idx, num_tok, N, d, H):
    t, tH, mask = N // d, (N // d) * H, np.zeros(N)
    for i in range(N):
        if i // tH == group_idx and i % t < num_tok:
            mask[i] = 1.0
    return mask


# def decode_qkt(raw_cts, n_k, N, d, H):
#     """Decode qkt ciphertext-window output into (n_q, H, n_k) scores."""
#     n_q = len(raw_cts)
#     cap_keys = N // H
#     scores = np.zeros((n_q, H, n_k))
#     for qi, pre_softmax_attn in enumerate(raw_cts):
#         chunk_keys = min(cap_keys, n_k)
#         decoded_chunk = decode_attn_scores([pre_softmax_attn], chunk_keys, N, d, H)[0]
#         scores[qi, :, :chunk_keys] = decoded_chunk
#     return scores


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


# def qkt_batched(Q, K, N, d, H=1):
#     """
#     Batched prefill QK^T: t queries per ciphertext, inner rotations
#     to compute all token cross-products per (Q_group, K_group) pair.

#     Same interface and output shape as qkt().
#     """
#     if Q.ndim == 1:
#         Q = Q.reshape(1, -1)

#     n_q, n_k, t = Q.shape[0], K.shape[0], N // d

#     assert Q.shape[1] == d and K.shape[1] == d
#     assert d % H == 0 and N % d == 0
#     assert n_k <= N // H

#     kcache_cts = KCache(N, d, K, H)
#     scores     = np.zeros((n_q, H, n_k))

#     n_q_groups = (n_q + t - 1) // t

#     for qg in range(n_q_groups):
#         q_start = qg * t
#         q_end   = min(q_start + t, n_q)
#         num_q   = q_end - q_start

#         q_group_cache = KCache(N, d, Q[q_start:q_end], H)
#         cq_group = q_group_cache.ciphertexts[0]

#         for kg, ck in enumerate(kcache_cts.ciphertexts):
#             k_start = kg * t
#             k_end   = min(k_start + t, n_k)
#             num_k   = k_end - k_start

#             for s in range(t):
#                 cq_shifted = inner_rot(cq_group, s, t, N)
#                 result     = _sum_by_rot(cq_shifted * ck, N, d, H)

#                 for tok_k in range(num_k):
#                     tok_q_local = (tok_k + s) % t
#                     if tok_q_local < num_q:
#                         qi = q_start + tok_q_local
#                         ki = k_start + tok_k
#                         for h in range(H):
#                             scores[qi, h, ki] = result[_pos(0, h, tok_k, t, H)]

#     return scores


def _gt(Q, K, H, N):
    """Per-head ground truth QK^T packed in head-interleaved slot layout."""
    if Q.ndim == 1:
        Q = Q[None]

    d = Q.shape[1]
    assert d % H == 0, f"d={d} must be divisible by H={H}"
    assert N % d == 0, f"N={N} must be divisible by d={d}"

    t = N // d
    mult_d = d // H
    out = np.zeros((Q.shape[0], N))

    for h in range(H):
        sl = slice(h * mult_d, (h + 1) * mult_d)
        A = Q[:, sl] @ K[:, sl].T
        for i in range(A.shape[1]):
            pos = (i // t) * t * H + h * t + (i % t)
            if pos < N:
                out[:, pos] = A[:, i]

    return out


def _run(label, Q, K, N, d, H, batched=False):
    nq  = 1 if Q.ndim == 1 else Q.shape[0]
    nk  = K.shape[0]
    t   = N // d
    print(f"  {label:12s} N={N:4d} d={d:2d} H={H} t={t:2d} "
          f"nq={nq:2d} nk={nk:3d}:", end=' ')
    gt = _gt(Q, K, H, N)
    kcache = KCache(N, d, K, H)
    scores = kcache.qkt(Q)
    err = np.max(np.abs(gt - scores))
    tag = "OK" if err < 1e-10 else "FAIL"
    print(f"{err:.2e} {tag}")
    return err < 1e-10


def _same_kcache_layout(a, b):
    if a.curr_keys != b.curr_keys or len(a.ciphertexts) != len(b.ciphertexts):
        return False
    return all(np.allclose(ga, gb) for ga, gb in zip(a.ciphertexts, b.ciphertexts))


def _run_kcache_path_test(label, Q, K_base, K_new, N, d, H):
    K_full = np.vstack([K_base, K_new]) if K_new.shape[0] > 0 else K_base.copy()

    cache_full = KCache(N, d, K_full, H)
    cache_inc  = KCache(N, d, K_base, H)
    for i in range(K_new.shape[0]):
        cache_inc.push_back(K_new[i])

    scores_full = cache_full.qkt(Q)
    scores_inc  = cache_inc.qkt(Q)
    gt = _gt(Q, K_full, H, N)

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
    def __init__(self, N, d, V=None, H=1):
        assert H==1, "Multi-head V cache not implemented yet"
        self.curr_values = 0
        self.N = N
        self.d = d
        self.H = H
        self.metaciphertexts = None
        if V is not None:
            self._build(V)

    def _build(self, V):
        self.curr_values = V.shape[0]
        n_metagroups = (self.curr_values // self.N) + (self.curr_values % self.N > 0)
        right_rot = self.N // self.d
        self.metaciphertexts = []
        for mg in range(n_metagroups):
            metacipher = [np.zeros(self.N) for _ in range(self.d)]
            start = mg * self.N
            end = min(start + self.N, self.curr_values)
            for i in range(start, end):
                mg_i = i - start
                curr_i = mg_i // right_rot
                for j in range(self.d):
                    curr_j = j * right_rot + (mg_i % right_rot)
                    metacipher[curr_i][curr_j] = V[i, j]
                    curr_i = (curr_i - 1) % self.d
            self.metaciphertexts.append(metacipher)

    def push_back(self, value):
        """Append one value vector using the same layout as _build."""
        assert value.shape[0] == self.d, f"value has {value.shape[0]} cols, expected d={self.d}"

        if self.metaciphertexts is None:
            self.metaciphertexts = []

        if self.curr_values % self.N == 0:
            self.metaciphertexts.append([np.zeros(self.N) for _ in range(self.d)])

        right_rot = self.N // self.d
        mg_i = self.curr_values % self.N
        curr_i = mg_i // right_rot
        metacipher = self.metaciphertexts[-1]
        for j in range(self.d):
            curr_j = j * right_rot + (mg_i % right_rot)
            metacipher[curr_i][curr_j] = value[j]
            curr_i = (curr_i - 1) % self.d

        self.curr_values += 1


def _print_vcache_layout(vcache, label):
    print(f"  {label}: metagroups={len(vcache.metaciphertexts)}")
    for mg_idx, metacipher in enumerate(vcache.metaciphertexts):
        print(f"    metacipher {mg_idx}")
        for lane_idx, lane_ct in enumerate(metacipher):
            print(f"      lane {lane_idx}: {lane_ct}")

def main():
    ok = True

    print("=== VMM sanity ===")
    for lab, N, di, do in [("sq", 32, 8, 8), ("up", 128, 16, 64), ("dn", 128, 64, 16)]:
        x = np.random.random(di); W = np.random.random((di, do))
        e = np.max(np.abs(x @ W - cachemir_vmm(x, W, N)))
        t = "OK" if e < 1e-10 else "FAIL"
        print(f"  {lab:3s} ({di}→{do}): {e:.2e} {t}")
        ok &= (e < 1e-10)

    # ── Multi-head QK^T ──
    print("\n=== QK^T multi-head decoding ===")
    for lab, N, d, H, nk in [("2h",8,4,2,3), ("4h",64,8,4,12),("8h",128,16,8,8)]:
        np.random.seed(42)
        ok &= _run(lab, np.random.randn(d), np.random.randn(nk, d), N, d, H)

    print("\n=== QK^T multi-head classification ===")
    for lab, N, d, H, n in [("2h",8,4,2,3),("4h",64,8,4,12),("8h",128,16,8,8)]:
        np.random.seed(42)
        ok &= _run(lab, np.random.randn(n, d), np.random.randn(n, d), N, d, H)

    # ── KCache construction paths ──
    print("\n=== KCache build paths (full vs incremental add) ===")
    np.random.seed(42)
    N, d, H = 128, 16, 8

    Q = np.random.randn(5, d)
    K_full = np.random.randn(11, d)
    cache_full = KCache(N, d, K_full, H)
    scores_cache = cache_full.qkt(Q)
    gt_full = _gt(Q, K_full, H, N)
    err_full_build = np.max(np.abs(gt_full - scores_cache))
    tag = "OK" if err_full_build < 1e-10 else "FAIL"
    print(f"  {'whole-k':12s} N={N:4d} d={d:2d} H={H} nq={Q.shape[0]:2d} nk={K_full.shape[0]:3d}: {err_full_build:.2e} {tag}")
    ok &= (err_full_build < 1e-10)

    K_base = np.random.randn(8, d)
    K_more = np.random.randn(4, d)
    ok &= _run_kcache_path_test("add-1", Q, K_base, K_more[:1], N, d, H)
    ok &= _run_kcache_path_test("add-many", Q, K_base, K_more, N, d, H)

    print("\n=== VCache storage demo (constant vectors) ===")
    N, d, H = 16, 4, 1
    n_vals = 7
    V_demo = np.vstack([np.full((d,), i + 1.0) for i in range(n_vals)])
    print(f"  params: N={N}, d={d}, H={H}, n_values={n_vals}")
    print("  input values (row i is filled with i+1):")
    print(f"  {V_demo}")
    vcache_demo = VCache(N, d, V_demo, H)
    _print_vcache_layout(vcache_demo, "stored V-cache")

    print("  incremental push_back view:")
    vcache_inc = VCache(N, d, V_demo[:-1], H)
    _print_vcache_layout(vcache_inc, "before push_back")
    vcache_inc.push_back(V_demo[-1])
    _print_vcache_layout(vcache_inc, "after push_back")


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
    for g, ct in enumerate(kcache_inc.ciphertexts):
        start = g * t
        num_tok = min(t, kcache_inc.curr_keys - start)
        _print_group_interleaving(ct, N, d, H, num_tok, f"group {g} (keys {start}..{start + num_tok - 1})")

    kcache_inc.push_back(K_demo[-1])
    print(f"  after add : n_keys={kcache_inc.curr_keys} (added key value {n_demo})")
    for g, ct in enumerate(kcache_inc.ciphertexts):
        start = g * t
        num_tok = min(t, kcache_inc.curr_keys - start)
        _print_group_interleaving(ct, N, d, H, num_tok, f"group {g} (keys {start}..{start + num_tok - 1})")

    # ── Trace ──
    print("\n=== Trace: N=8, d=4, H=2, t=2, nk=3 ===")
    N, d, H, t = 8, 4, 2, 2
    q  = np.array([1., 2., 3., 4.])
    Km = np.array([[.1,.2,.3,.4],[.5,.6,.7,.8],[.9,1.,1.1,1.2]])
    gt = _gt(q, Km, H, N)

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
    ck01 = kcache_trace_01.ciphertexts[0]
    print(f"K[0,1]   = {ck01}")

    prod = cq_rep * ck01
    acc  = _sum_by_rot(prod, N, d, H)
    r0   = acc * _qkt_mask(0, 2, N, d, H)
    print(f"group0   = {r0}")

    kcache_trace_2 = KCache(N, d, Km[2:3], H)
    ck2 = kcache_trace_2.ciphertexts[0]
    r1  = _sum_by_rot(cq_rep * ck2, N, d, H) * _qkt_mask(0, 1, N, d, H)
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