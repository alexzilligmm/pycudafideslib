import math

import numpy as np
from math import sqrt


def pmod(a, n):
    return ((a % n) + n) % n


def rot(v, k):
    return np.roll(v, -k)


def bsgs_params(d_in, d_out, N):
    """Compute BSGS parameters.

    n_pt = d_in·d_out / N         (Go util.go:213  PrepareWeights)
    r_i  ∈ {2^n}                  (Paper Alg.1 line 38)
    r_i · r_o = n_pt              (Paper proof line 137: r_i·r_o·t' = d)
    """
    expand = -1 if d_in > d_out else (1 if d_in < d_out else 0)
    d = min(d_in, d_out)                    # paper's d
    alpha = max(d_in, d_out) // d           # paper's α
    t = N // d                              # paper's t  = N/d
    t_prime = N // (alpha * d)              # paper's t' = N/(αd)

    n_pt = d_in * d_out // N               

    if expand == 0:
        r_i = int(math.sqrt(d * d / (2 * N)))
        r_o = int(d * d / (N * r_i))
    else:
        r_i = int(math.sqrt(d_in * d_out / (2 * N)))
        r_o = int(d_in * d_out / (N * r_i))  

    assert math.log2(r_i).is_integer()            

    return n_pt, r_i, r_o, d, alpha, t, t_prime



def encode_weights(W, d_in, d_out, N):
    """Encode W into n_pt plaintexts p_{j,k}.

    Each plaintext slot i stores W[row, col] where row and col are chosen
    to compensate for the baby-step rotation j·t' and giant-step rotation
    k·r_i·t' that will be applied during gemv.
    """
    expand = -1 if d_in > d_out else (1 if d_in < d_out else 0)
    n_pt, r_i, r_o, d, alpha, t, t_prime = bsgs_params(d_in, d_out, N)

    t1 = N // d_in                          # = t for up/square, t' for down
    stride = d_in // t1                     # interleave step within t1

    t2 = N // d_out                         # = t' for up, t for down/square

    plaintexts = []
    for idx in range(n_pt):
        j = idx % r_i                       # baby-step index
        k = idx // r_i                      # giant-step index
        p_jk = np.zeros(N)
        for i in range(N):
            if expand == 0:
                row = (math.floor(i // t) + i % t + j * t) % d
                col = (i // t - k * t * r_i) % d
            if expand == 1:
                row = (i // t + i % t + j * t) % d
                col = (math.floor(i // t) + ((i // t_prime) % alpha) * d - k * t * r_i) % (alpha * d)
            # TODO: down part is broken
            if expand == -1:
                row = (math.floor(i // t) + ((i // t_prime) % alpha) * d + i % t + j * t) % (alpha * d)
                col = (i // t - k * t * r_i) % d

            p_jk[i] = W[row][col]
        plaintexts.append(p_jk)

    seen = set()
    for idx in range(n_pt):
        j = idx % r_i
        k = idx // r_i
        for i in range(N):
            if expand == 0:
                row = (math.floor(i // t) + i % t + j * t) % d
                col = (i // t - k * t * r_i) % d
            if expand == 1:
                row = (i // t + i % t + j * t) % d
                col = (math.floor(i // t) + ((i // t_prime) % alpha) * d - k * t * r_i) % (alpha * d)
            # TODO: down part is broken don't know why
            if expand == -1:
                row = (math.floor(i // t) + ((i // t_prime) % alpha) * d + i % t + j * t) % (alpha * d)
                col = (i // t - k * t * r_i) % d
            seen.add((row, col))
    expected_pairs = {(r, c) for r in range(d_in) for c in range(d_out)}
    missing = expected_pairs - seen
    extra = seen - expected_pairs
    if missing:
        print(f"  ENCODE CHECK: {len(missing)} missing (row,col) pairs out of {d_in*d_out}")
        for p in sorted(missing)[:20]:
            print(f"    missing: row={p[0]} col={p[1]}")
    if extra:
        print(f"  ENCODE CHECK: {len(extra)} out-of-range (row,col) pairs")
    if not missing and not extra:
        print(f"  ENCODE CHECK: all {d_in*d_out} (row,col) pairs covered OK")

    return plaintexts


def gemv(cx, plaintexts, d_in, d_out, N):
    n_pt, r_i, r_o, d, alpha, t, t_prime = bsgs_params(d_in, d_out, N)
    expand = -1 if d_in > d_out else (1 if d_in < d_out else 0)

    if expand >= 0:                              
        t1 = t
    else:                                       
        t1 = t_prime

    if expand <= 0:                             
        t2 = t_prime
    else:                                       
        t2 = t

    cx_rot = [None] * r_i
    cy_partial = [None] * n_pt

    step = 1
    for i in range(int(math.log2(t1))):   
        cx = cx + rot(cx, (step ** i) * (d - 1))

    for j in range(r_i):                         # c''^x_j ← rot(c'^x, j·t')
        cx_rot[j] = rot(cx, j * t_prime)

    for i in range(n_pt):
        j = i % r_i
        cy_partial[i] = cx_rot[j] * plaintexts[i]
        if j > 0:
            cy_partial[i - j] = cy_partial[i - j] + cy_partial[i]

    for k in range(1, r_o):                      # rot(c'^y_k, k·t·r_i)  (paper line 54)
        cy_partial[r_o - k - 1] = rot(cy_partial[k * r_i], k * d * r_i)

    for k in range(1, r_o):
        cy_partial[0] = cy_partial[0] + cy_partial[k * r_i]

    cy = cy_partial[0]
    for i in range(int(math.log2(t2))):
        cy = cy + rot(cy, 2 ** i)

    return cy                                  

def encode_input(x, d, N):
    t = N // d
    cx = np.zeros(N)

    for i in range(N):
        if i % t == 0:                  # I{t | i}
            cx[i] = x[i // t]
    used = set()

    for i in range(N):
        if i % t == 0:
            idx = i // t
            used.add(idx)

    expected = set(range(d))

    missing = expected - used
    extra = used - expected

    if missing:
        print(f"INPUT CHECK: missing indices: {sorted(missing)}")
    if extra:
        print(f"INPUT CHECK: extra indices: {sorted(extra)}")
    if not missing and not extra:
        print(f"INPUT CHECK: all {d} entries used exactly once")
    return cx


# TODO: fix this
def encode_input_reduce(x, d_in, d_out, N):
    """Down:  c^x[i] = x[⌊i/t⌋ + (i/t')%alpha · d] · I{t'|i},  t' = N/d_in."""
    alpha = d_in // d_out
    t = N // d_out
    t_prime = N // d_in
    
    cx = np.zeros(N)
    for i in range(N):
        if i % t_prime == 0:                          # I{t'|i}
            idx = i // t + (i // t_prime) % alpha * d_out
            cx[i] = x[int(idx)]
    return cx


# TODO: check this 
def decode_output(cy, d, N):
    """Extract:  y[m] = c^y[m·t],  t = N/d."""
    t = N // d
    return np.array([cy[m * t] for m in range(d)])

# TODO: fix this
def decode_output_expanded(cy, d_in, d_out, N):
    """Extract up-projection result with alpha-interleaving."""
    alpha = d_out // d_in
    t_prime = N // d_out
    y = np.zeros(d_out)
    for m in range(d_out):
        idx = m // alpha + (m % alpha) * d_in
        y[idx] = cy[m * t_prime]
    return y


def run_test(d_in, d_out, N, seed=42, test_mode="basis"):
    rng = np.random.default_rng(seed + d_in * 1000 + d_out)
    if test_mode == "identity":
        W = np.eye(d_in, d_out)
        x = np.zeros(d_in); x[0] = 1.0
    elif test_mode == "basis":
        W = np.eye(d_in, d_out)
        x = np.zeros(d_in); x[1] = 1.0
    elif test_mode == "ones":
        W = np.ones((d_in, d_out))
        x = np.ones(d_in)
    elif test_mode == "randW_e0":
        W = rng.standard_normal((d_in, d_out)) / sqrt(d_in)
        x = np.zeros(d_in); x[0] = 1.0
    else:
        W = rng.standard_normal((d_in, d_out)) / sqrt(d_in)
        x = rng.standard_normal(d_in)
    y_exp = x @ W

    plaintexts = encode_weights(W, d_in, d_out, N)

    if d_in > d_out:
        cx = encode_input_reduce(x, d_in, d_out, N)
    else:
        cx = encode_input(x, d_in, N)

    cy = gemv(cx, plaintexts, d_in, d_out, N)


    if d_in < d_out:
        y_out = decode_output_expanded(cy, d_in, d_out, N)
    else:
        y_out = decode_output(cy, d_out, N)


    err = np.max(np.abs(y_out - y_exp))
    diff = y_out - y_exp
    bad = np.where(np.abs(diff) > 1e-10)[0]
    if len(bad) > 0:
        print(f"  {len(bad)}/{len(y_out)} slots wrong:")
        for b in bad[:20]:
            print(f"    y[{b:3d}]: got {y_out[b]:+.6f}, exp {y_exp[b]:+.6f}, diff {diff[b]:+.6f}")
        ok_slots = np.where(np.abs(diff) <= 1e-10)[0]
        if len(ok_slots) <= 20:
            print(f"  correct slots: {ok_slots.tolist()}")
    return err


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=120)

    print("=" * 60)
    print("CUDA-matching simulation (Algorithm 1, Appendix C)")
    print("=" * 60)

    print("\n--- SQUARE ---")
    for d, N in [(16, 128)]:
        n_pt, r_i, r_o, _, _, t, t_prime = bsgs_params(d, d, N)
        err = run_test(d, d, N, test_mode="random")
        ok = err < 1e-10
        print(f"  d={d:4d} N={N:5d} t={t:2d} n_pt={n_pt:3d} "
              f"r_i={r_i:2d} r_o={r_o:2d}  "
              f"err={err:.2e}  {'OK' if ok else 'FAIL'}")

    print("\n--- UP  (d_out = 4·d_in) ---")
    for d_in, N in [(32, 128)]:
        d_out = d_in * 4
        for mode in ["randW_e0", "random"]:
            n_pt, r_i, r_o, d, alpha, t, t_prime = bsgs_params(d_in, d_out, N)
            err = run_test(d_in, d_out, N, test_mode=mode)
            ok = err < 1e-10
            print(f"  [{mode:6s}] d_in={d_in:4d} d_out={d_out:5d} N={N:5d} "
                  f"r_i={r_i:2d} r_o={r_o:2d}  "
                  f"err={err:.2e}  {'OK' if ok else 'FAIL'}")

    # print("\n--- DOWN  (d_in = 4·d_out) ---")
    # for d_out, N in [(16, 256)]:
    #     d_in = d_out * 4
    #     for mode in ["randW_e0", "random"]:
    #         n_pt, r_i, r_o, d, alpha, t, t_prime = bsgs_params(d_in, d_out, N)
    #         err = run_test(d_in, d_out, N, test_mode=mode)
    #         ok = err < 1e-10
    #         print(f"  [{mode:6s}] d_in={d_in:4d} d_out={d_out:5d} N={N:5d} "
    #               f"r_i={r_i:2d} r_o={r_o:2d}  "
    #               f"err={err:.2e}  {'OK' if ok else 'FAIL'}")

if __name__ == "__main__":
    main()
