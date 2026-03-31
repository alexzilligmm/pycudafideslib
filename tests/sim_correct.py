import math

import numpy as np
from sympy import O


def rot(v, k):
    return np.roll(v, -k)


def easy():
    # Let's try the case N = d^2
    # we a have vector x \in R^d
    # a matrix in W \in R^(dxd)
    # in this an input vector is packed into the N- degree polynomial ring
    # replicating the input for t times in each **block** where t = N / d
    # so that the input vector is x = [x_1, x_1, ... x_2, ...] where for every i it appears t times.
    # so in code 
    N = 16
    d = 4
    x = np.random.random(d)
    
    def encode(x, N, d):
        t = N // d
        ptx = np.zeros(N)
        for i in range(N):
            ptx[i] = x[int(i // t)]
        return ptx
    
    print(x)
    print(encode(x, N, d))
    
    # let's say we now want to use the interleaved replicated packing to multiply
    # for a matrix d * d. Crucially we are considering x @ W. The packing for the weights
    # works as follow: to encode a d x d matrix we need d^2 / N plain text arrays, so 
    # in this case one plaintext. Would be enough. The algorithm works as follows we first
    # multiply each (in this case the) plaintext element wise for the input vector x. Then we perform
    # the log_2(d) rotations and then sum, so that we will have N / d entries with the correct results 
    # we only have to collect the i | t indices.
    
    n_pt = int(d * d / N)
    # o anche meglio
    # n_pt = d / t ossia per encodare una matrice servono tanto più vettori quanti
    # meno ripetizioni di x_i ci sono
    W = np.random.random((d, d))
    print(f"We will need {n_pt} to encode the matrix")
    
    def encode_W(W, N, d):
        t = N // d
        n_pt = d // t
        pt = np.zeros((n_pt, N))
        for g in range(n_pt):
            for i in range(N):
                pt[g, i] = W[i // t, g * t + (i % t)]
        return pt
    
    gt = x @ W
    
    # okay with this encoding is enough to multiply and then rotate log_2 d times
    
    def vmm(enc_x, pts_W, N, d):
        t    = N // d
        n_pt = d // t
        y    = np.zeros(d)

        for g, pt in enumerate(pts_W):
            acc = enc_x * pt

            step = t
            while step < N:
                acc += rot(acc, step)
                step *= 2

        return acc
    
    # here we are!
    def decode(ptx, N, d):
        return ptx[: d]
    print(gt)
    print(decode(vmm(encode(x, N, d), encode_W(W, N, d), N, d), N, d))
        
    print(f"Error: {gt- decode(vmm(encode(x, N, d), encode_W(W, N, d), N, d), N, d)}")
    

def cachemir_test():
    N = 32
    d = 8
    t = N // d        # = 4
    n_pt = d * d // N # = 2
    
    def compute_r_i_r_o(N, d):
        r_i = d * d // (2 * N)
        r_o = d * d // (N * r_i)
        return r_i, r_o
    
    r_i, r_o = compute_r_i_r_o(N, d)

    x = np.random.random(d)
    W = np.random.random((d, d))

    print(f"N={N}, d={d}, t={t}, r_i={r_i}, r_o={r_o}, n_pt={n_pt}")

    def encode_x(x, N, d):
        t = N // d
        ptx = np.zeros(N)
        for i in range(N):
            if i % t == 0:
                ptx[i] = x[i // t]
        return ptx

    def encode_W(W, N, d, r_i, r_o):
        t = N // d
        n_pt = r_i * r_o
        pt = np.zeros((n_pt, N))
        for j in range(r_i):
            for k in range(r_o):
                for i in range(N):
                    row = (i//t + i%t + j) % d
                    col = (i//t - k*t)     % d
                    pt[j*r_o+k, i] = W[row, col]
        return pt

    ptx = encode_x(x, N, d)
    ptx_prime = ptx.copy()
    step = 1
    while step < t:
        ptx_prime = ptx_prime + rot(ptx_prime, step * (t - 1))
        step *= 2

    pts_W = encode_W(W, N, d, r_i, r_o)
    cy_prime = np.zeros((r_o, N))
    for k in range(r_o):
        for j in range(r_i):
            cx_rot = rot(ptx_prime, j * t)
            cy_prime[k] += cx_rot * pts_W[j * r_o + k]

    for k in range(r_o - 1, 0, -1):
        cy_prime[k-1] += rot(cy_prime[k], t * t)
    cy = cy_prime[0]

    step = 1
    while step < t:
        cy = cy + rot(cy, step)
        step *= 2

    y_true = x @ W
    y_fhe  = cy[::t][:d]
    print(f"match: {np.allclose(y_fhe, y_true)}")
    print(f"error: {np.max(np.abs(y_fhe - y_true)):.2e}")
    
def main():
    cachemir_test()

if __name__ == "__main__":
    main()
    