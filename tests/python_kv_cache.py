import numpy as np
from linear import *

def pre_multi_head_comp(X, W_Q, W_K, W_V):
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V

    Q = Q.reshape(Q.shape[0], H, -1)
    K = K.reshape(K.shape[0], H, -1)
    V = V.reshape(V.shape[0], H, -1)

    return Q, K, V
    

def multi_head_attention(Q, K, V, H):
    """Ground-truth QK^T and QKV outputs for testing.
    Args:
        Q, K, V: shape (n_q, H, d_head)
        H: number of heads
    Returns:
        gt_qkt: shape (n_q, n_k) raw QK^T scores
        gt_attn: shape (n_q, d) final attention output after softmax and V matmul
    """

    gt_qkt = np.einsum('qhd,khd->qkh', Q, K)
    attn_weights = np.exp(gt_qkt - np.max(gt_qkt, axis=-1, keepdims=True))
    attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
    gt_attn = np.einsum('qkh,khd->qhd', attn_weights, V).reshape(X.shape[0], -1)

    return gt_qkt, gt_attn

def gt_qkt_repack(gt_qkt, N, d):
    """Repack ground-truth QK^T into head-interleaved N-slot format."""
    n_q, n_k, H = gt_qkt.shape
    d_head = d // H
    t = N // d
    assert n_k <= t * H, f"Too many keys {n_k} for N={N}, d={d}, H={H}"
    qkt_repacked = np.zeros((n_q, N))
    for gt, rep in zip(gt_qkt, qkt_repacked):
        for k, k_row in enumerate(gt):
            for h, h_val in enumerate(k_row):
                rep[k//t * t * H + h * t + k%t] = h_val

    return qkt_repacked

def gt_attn_repack(gt_attn, H, N, d):
    """Repack ground-truth attention output into head-interleaved N-slot format."""
    n_q, d_total = gt_attn.shape
    real_head_d = d_total // H
    t = N // d_total
    attn_repacked = np.zeros((n_q, N))
    for gt, rep in zip(gt_attn, attn_repacked):
        for h in range(H):
            gt_head = gt[h * real_head_d:(h + 1) * real_head_d]
            for i, v in enumerate(gt_head):
                rep[_pos(i, h, 0, t, H)] = v
    return attn_repacked

def rearrange_q_k_v(W, H):
    # perm[r] = (r % H) · d_head + ⌊r / H⌋  →  old_col = h · d_head + ld

    new_W = np.zeros_like(W)
    d_in, d_out = W.shape
    d_head = d_out // H
    for r in range(d_out):
        h = r % H
        ld = r // H
        new_W[:, r] = W[:, h * d_head + ld]
    return new_W

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

def neg_inf_mask(nk, N, d, H):
    t = N // d
    m = np.full(N, -1e30) # TODO: use actual approximation min
    for h in range(H):
        for tok in range(nk):
            m[_pos(tok // t, h, tok % t, t, H)] = 0.0
    return m


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
    
    result = cq_rep * kcache_ct # elementwise product in N-slot
    result = _sum_by_rot(result, N, d, H)
    mask = _qkt_mask(group_idx, num_tok, N, d, H)
    return result * mask

def head_reduce_sum(ct, N, d, H):
    t  = N // d
    tH = t * H
    out = ct.copy()

    step = 1
    while step < t:
        mask_nw = np.array([1.0 if (i % t) + step < t else 0.0 for i in range(N)])
        mask_w  = 1.0 - mask_nw
        # shift left inner head by step, shift right inner head by step - t to align values, mask to be sure we are not pulliting wrap around blocks
        shifted = rot(out, step) * mask_nw + rot(out, step - t) * mask_w
        # aggregate
        out = out + shifted
        step *= 2

    # inter block aggregations, standard heap
    step = tH
    while step < N:
        out = out + rot(out, step)
        step *= 2

    return out

def softmax(scores, nk, N, d, H, gt_max):
    out = np.zeros_like(scores)
    ninf = neg_inf_mask(nk, N, d, H)
    for qi in range(scores.shape[0]):
        ct = scores[qi] + ninf # to get zeros in empty positions
        e = np.exp(ct - gt_max[qi]) # exp for every row, interleaved by nature
        s = head_reduce_sum(e, N, d, H) # 
        out[qi] = e / s
    return out

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

        def _encode_q_interleaved(q, N, d, H=1):
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

        def _preprocess_q(cq, N, d):
            t   = N // d
            out = cq.copy()
            step = 1
            while step < t:
                out = out + rot(out, -step)
                step *= 2
            return out

        keys_per_ct = self.N // self.d

        cq = _encode_q_interleaved(q, self.N, self.d, self.H)
        cq_rep = _preprocess_q(cq, self.N, self.d)


        attn_ct = np.zeros(self.N)
        for global_ct_indx in range(len(self.ciphertexts)):
            first_local_key = global_ct_indx * keys_per_ct
            num_tok = min(keys_per_ct, self.curr_keys - first_local_key)
            partial = qkt_single_group(cq_rep, self.ciphertexts[global_ct_indx], self.N, self.d, self.H, global_ct_indx, num_tok)
            
            attn_ct += partial

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



class VCache:
    """Helper for building V cache ciphertexts from a value matrix."""
    def __init__(self, N, d, V=None, H=1):
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


def _print_group_interleaving(ct, N, d, H, num_tokens, label):
    """Print one packed KCache group to inspect raw slot interleaving."""
    print(f"  {label}")
    print(f"    raw slots: {ct}")

def _run(label, Q, K, N, d, H):
    nq  = 1 if Q.ndim == 1 else Q.shape[0]
    nk  = K.shape[0]
    t   = N // d
    print(f"  {label:12s} N={N:4d} d={d:2d} H={H} t={t:2d} "
          f"nq={nq:2d} nk={nk:3d}:", end=' ')
    gt = _gt(Q, K, H, N)
    kcache = KCache(N, d, K, H)
    scores = kcache.qkt(Q)
    print(scores)
    err = np.max(np.abs(gt - scores))
    tag = "OK" if err < 1e-10 else "FAIL"
    print(f"{err:.2e} {tag}")
    return err < 1e-10

def _run_softmax(label, Q, K, N, d, H):
    nk = K.shape[0]
    nq = 1 if Q.ndim == 1 else Q.shape[0]
    if Q.ndim == 1: Q = Q[None]
    t = N // d
    print(f"  {label:12s} N={N:4d} d={d:2d} H={H} t={t:2d} nq={nq:2d} nk={nk:3d}:", end=' ')
    scores = KCache(N, d, K, H).qkt(Q)
    gt_sm, gt_sum, gt_max = _soft_gt(Q, K, H, N)
    fhe_sm = softmax(scores, nk, N, d, H, gt_max)
    err = np.max(np.abs(gt_sm - fhe_sm))
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


def _print_vcache_layout(vcache, label):
    print(f"  {label}: metagroups={len(vcache.metaciphertexts)}")
    for mg_idx, metacipher in enumerate(vcache.metaciphertexts):
        print(f"    metacipher {mg_idx}")
        for lane_idx, lane_ct in enumerate(metacipher):
            print(f"      lane {lane_idx}: {lane_ct}")
            

def main():
    ok = True

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
    
    print("\n=== Softmax (oracle max, FHE sum) ===")
    for lab, N, d, H, nk in [
        ("small",      16, 4, 2, 5),
        ("gpt2", 32768, 1024, 16, 5)
    ]:
        np.random.seed(42)
        Q = np.random.randn(12, d)
        K = np.random.randn(nk, d)
        ok &= _run_softmax(lab, Q, K, N, d, H)
    
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

    print(f"\n{'ALL PASSED' if ok else 'SOME FAILURES'}")


if __name__ == "__main__":
    # main()

    N = 8
    d = 4
    H = 2
    n = 1
    assert n <= N // H, "Too many samples for demo parameters"

    X = np.random.randn(n, d)
    W_Q = np.random.randn(d, d)
    W_K = np.random.randn(d, d)
    W_V = np.random.randn(d, d)
    Q, K, V = pre_multi_head_comp(X, W_Q, W_K, W_V)
    gt_qkt, gt_attn = multi_head_attention(Q, K, V, H=H)
    print("gt_qkt shape:", gt_qkt.shape)
    print("gt_attn shape:", gt_attn.shape)

    qkt_repacked = gt_qkt_repack(gt_qkt, N, d)
    # attn_repacked = gt_attn_repack(gt_attn, H, N, d)
    print("qkt_repacked shape:", qkt_repacked.shape)
    # print("attn_repacked shape:", attn_repacked.shape)

    test_gt_qkt = np.zeros_like(gt_qkt)
    for q, q_row in enumerate(test_gt_qkt):
        for k, k_row in enumerate(q_row):
            for h in range(H):
                k_row[...,h] = h + 1
            k_row[...] += (k + 1) * 10
        q_row[...] += (q + 1) * 100

    print("test_gt_qkt:")
    print(test_gt_qkt)

    test_qkt_repacked = gt_qkt_repack(test_gt_qkt, N, d)
    print('test_qkt_repacked:')
    print(test_qkt_repacked)


    test_gt_attn = np.zeros_like(gt_attn)
    for q, q_row in enumerate(test_gt_attn):
        for h in range(H):
            for i in range(d // H):
                q_row[h * (d // H) + i] = i + 1
            q_row[h * (d // H): (h + 1) * (d // H)] += (h + 1) * 10
        q_row[...] += (q + 1) * 100

    print("test_gt_attn:")
    print(test_gt_attn)

    test_attn_repacked = gt_attn_repack(test_gt_attn, H, N, d)
    print("test_attn_repacked:")    
    print(test_attn_repacked)


    # W_Q = np.arange(d * d).reshape(d, d) + 1
    # W_K = np.arange(d * d).reshape(d, d) + 1

    rearranged_w_q = rearrange_q_k_v(W_Q, H) # d x d == d x (H * d_head)
    rearranged_w_k = rearrange_q_k_v(W_K, H) # d x d == d x (H * d_head)
    rearranged_w_v = rearrange_q_k_v(W_V, H) # d x d == d x (H * d_head)

    # rearranged_w_v = rearrange_v(W_V, H) # d x d == (H * d_head) x d

    encoded_W_q = encode_W(rearranged_w_q, N, d, alpha=1, is_up=True)
    encoded_W_k = encode_W(rearranged_w_k, N, d, alpha=1, is_up=True)
    encoded_W_v = encode_W(rearranged_w_v, N, d, alpha=1, is_up=True)

    computed_params = compute_params(N, d, alpha=1, is_up=True)

    print("encoded_W_k shape:", encoded_W_k.shape)
    print("W_K")
    print(W_K)
    print("encoded_W_k:")
    print(encoded_W_k)

    encoded_x = encode_x(X[0], N, d, alpha=1, is_up=True)

    encoded_K = cachemir_vmm(encoded_x, encoded_W_k, N, d, d, alpha=1, is_up=True, computed_params=computed_params)

    print(K)
    print(encoded_K)

