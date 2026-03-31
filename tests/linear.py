import numpy as np
from math import log2


def rot(v, k):
    """Cyclic rotation: rot(v, k)[i] = v[(i+k) % len(v)]."""
    return np.roll(v, -k)


def compute_params(N, d, alpha, is_up):
    """Compute all algorithm parameters from N, d, alpha, direction."""
    t  = N // d                        
    tp = N // (alpha * d)              

    tp_in  = t  if is_up else tp                # (t1 and t2 in Cachemir)
    tp_out = tp if is_up else t                 # (t1 and t2 in Cachemir)

    d_in  = d if is_up else alpha * d
    n_pt  = d_in // tp_out                      # from Chachemir's go code this coincides
    r_i   = max(1, d * d // N)          
    r_i   = min(r_i, n_pt)
    r_o   = n_pt // r_i

    return t, tp, tp_in, tp_out, r_i, r_o, n_pt


def interleave_idx(m, d, dim):
    a = dim // d if dim > d else 1
    return (m // a + (m % a) * d) % dim


def encode_x(x, N, d, alpha, is_up):
    """
    Encode activation vector into N-slot ciphertext.

    Up  (x ∈ R^d):   c^x[i] = x[i/t] · 1{t|i}
    Down (x ∈ R^alpha* d):  c^x[i] = x[interleave(i/t')] · 1{t'|i}
    """
    t, tp, _, _, _, _, _ = compute_params(N, d, alpha, is_up)
    d_in = d if is_up else alpha * d
    M = N // tp  # = αd

    ptx = np.zeros(N)
    if is_up:
        for i in range(d):
            ptx[i * t] = x[i]
    else:
        for m in range(M):
            ptx[m * tp] = x[interleave_idx(m, d, d_in)]
    return ptx


def encode_W(W, N, d, alpha, is_up):
    """
    Encode weight matrix into n_pt = r_i·r_o plaintext vectors.

    Row formula (index into d_in):
        ((i//t + j·t + i%tp_in) % d) + ((i%t)//tp_in) · d

    Col formula (index into d_out, with interleaved output mapping):
        interleave_idx((i//tp_out - k·cascade_shift) % M_out, d_out)
    """
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

def main():
    print("testing square matrix results")
    N = 32
    d = 8
    
    x = np.random.random(d)
    W = np.random.random((d, d))
    
    gt = x @ W
    out = cachemir_vmm(x, W, N)
    
    print(f"Error {gt - out}")
    
    print("testing up proj matrix results")
    N = 128
    d = 16
    d_out = 64
    
    x = np.random.random(d)
    W = np.random.random((d, d_out))
    
    gt = x @ W
    out = cachemir_vmm(x, W, N)
    
    print(f"Error {gt - out}")
    
    print("testing down proj matrix results")
    N = 128
    d = 64
    d_out = 16
    
    x = np.random.random(d)
    W = np.random.random((d, d_out))
    
    gt = x @ W
    out = cachemir_vmm(x, W, N)
    
    print(f"Error {gt - out}")
    
    print("Real params test")
    
    print("testing square matrix results")
    N = 65536
    d = 1024
    
    x = np.random.random(d)
    W = np.random.random((d, d))
    
    gt = x @ W
    out = cachemir_vmm(x, W, N)
    
    print(f"Error {sum((gt - out)>1e-10)}")
    
    print("testing up proj matrix results")
    N = 65536
    d = 1024
    d_out = 4096
    
    x = np.random.random(d)
    W = np.random.random((d, d_out))
    
    gt = x @ W
    out = cachemir_vmm(x, W, N)
    
    print(f"Error {sum((gt - out)>1e-10)}")
    
    print("testing down proj matrix results")
    N = 65536
    d = 1024
    d_out = 4096
    
    x = np.random.random(d)
    W = np.random.random((d, d_out))
    
    gt = x @ W
    out = cachemir_vmm(x, W, N)
    
    print(f"Error {sum((gt - out)>1e-10)}")
    
    

main()