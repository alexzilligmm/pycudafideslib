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
    
def cachemir():
    # still square case
    N = 32
    d = 8
    n_pt = d * d // N 
    t = N // d
    x = np.random.random(d)
    W = np.random.random((d, d))
    
    # cachemir encodes with holes and fills later during loop 1 i.e. Algorithm 1 line 6
    def encode(x, N, d):
        t = N // d
        ptx = np.zeros(N)
        for i in range(N):
            if i % t == 0:
                ptx[i] = x[int(i // t)]
        return ptx
    
    def valid_ri(N, d):
        r_i = int(math.sqrt( d * d / (2 * N)))
        r_o = d * d // (N * r_i)
        return r_i, r_o
    
    def encode_W(W, N, d):
        t = N // d
        n_pt = d * d // N 
        pt = np.zeros((n_pt, N))
        r_i, r_o = valid_ri(N, d)
        print(r_i, r_o)
        for j in range(r_i):
            for k in range(r_o):
                for i in range(N):
                    row = (i // t + i % t + j * t) % d
                    col = (i // t - k * t * r_i)   % d
                    pt[j * r_o + k, i] = W[row, col]
        return pt, r_i, r_o
    
    ptx = encode(x, N, d)
    ptx_W, r_i, r_o = encode_W(W, N, d)
    
    # popoulate holes first for of algorithm 1
    ptx_prime = ptx.copy()
    
    step = 1
    while step < t:
        ptx_prime = ptx_prime + rot(ptx_prime, step * (t - 1)) # algo 1 is wrong it is ptx_prime instead of ptx
        step *= 2
    print(ptx_prime)
    print(ptx_W)

    
    
    
    
    
    
def main():
    cachemir()


if __name__ == "__main__":
    main()
