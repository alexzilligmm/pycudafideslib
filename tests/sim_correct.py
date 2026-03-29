#!/usr/bin/env python3
"""
Python simulation of linear_interleaved from linear.cu + install_weights from test_linear.cu.
Mirrors the CUDA code exactly, variable names match 1:1.
"""
import numpy as np
from math import sqrt


def pmod(a, n):
    return ((a % n) + n) % n


def rot(v, k):
    return np.roll(v, -k)


def install_weights(W, d_in, d_out, s):
    """Mirrors test_linear.cu::install_weights exactly."""
    expand = -1 if d_in > d_out else (1 if d_in < d_out else 0)

    d_pre  = d_in                         # line 48: always d_in
    K_pre  = s // d_pre
    stride = d_pre // K_pre

    intRot = s // max(d_in, d_out)        # line 52
    K_post = s // d_out                   # line 54-57: always S/d_out

    if expand == 0:                       # line 60-66
        inRot  = max(1, int(sqrt(d_in * d_in / (2 * s))))
        outRot = d_in * d_in // (s * inRot)
    else:
        inRot  = max(1, int(sqrt(d_in * d_out / (2 * s))))
        outRot = d_in * d_out // (s * inRot)
    nPt   = inRot * outRot
    alpha = 1 if expand == 0 else max(d_in, d_out) // min(d_in, d_out)

    weights = []
    for t in range(nPt):                  # line 71-98
        b = t % inRot
        g = t // inRot
        sl = np.zeros(s)
        for q in range(s):
            q_x = (q + b * intRot) % s
            row_raw = pmod(q_x // K_pre + (q_x % K_pre) * stride, d_pre)

            if expand == -1:
                row = row_raw // alpha + (row_raw % alpha) * d_out
            else:
                row = row_raw

            q_shifted = (q + s - g * inRot * intRot) % s
            m = q_shifted // K_post

            if expand == 1:
                col = m // alpha + (m % alpha) * d_in
            else:
                col = m % d_out

            sl[q] = W[row][col]
        weights.append(sl)
    return weights


def linear_interleaved(x_enc, weights, d_in, d_out, numSlots):
    """Mirrors linear.cu::linear_interleaved exactly."""
    expand = -1 if d_in > d_out else (1 if d_in < d_out else 0)

    intRot = numSlots // max(d_in, d_out)

    if expand >= 0:
        preProc = numSlots // d_in        # hidDim for sq/up
    else:
        preProc = numSlots // d_in        # expDim for down (d_in IS expDim)

    if expand <= 0:
        postProc = numSlots // d_out      # hidDim for sq/down
    else:
        postProc = numSlots // d_out      # expDim for up (d_out IS expDim)

    if expand == 0:
        inRot  = max(1, int(sqrt(d_in * d_in / (2 * numSlots))))
        outRot = d_in * d_in // (numSlots * inRot)
    else:
        inRot  = max(1, int(sqrt(d_in * d_out / (2 * numSlots))))
        outRot = d_in * d_out // (numSlots * inRot)

    d_pre = d_in if expand >= 0 else d_in  # line 52: hidDim for sq/up, expDim for down

    # Preprocessing: line 59-62
    x = x_enc.copy()
    i = 1
    while i < preProc:
        x = x + rot(x, i * (d_pre - 1))
        i *= 2

    # Baby steps from base: line 64-67
    ctRot = [None] * inRot
    ctRot[0] = x.copy()
    for i in range(1, inRot):
        ctRot[i] = rot(x, i * intRot)

    # Multiply + baby-step accumulate: line 69-74
    nPt = len(weights)
    partSum = [None] * nPt
    for i in range(nPt):
        partSum[i] = ctRot[i % inRot] * weights[i]
        if i % inRot > 0:
            partSum[i - (i % inRot)] = partSum[i - (i % inRot)] + partSum[i]

    # Giant step: line 76-84
    for i in range(1, outRot):
        partSum[i * inRot] = rot(partSum[i * inRot], i * intRot * inRot)
    for i in range(1, outRot):
        partSum[0] = partSum[0] + partSum[i * inRot]

    # Postprocessing: line 86-89
    result = partSum[0]
    i = 1
    while i < postProc:
        result = result + rot(result, i)
        i *= 2

    return result


def enc_simple(x, d, s):
    """Mirrors test_linear.cu::enc_simple."""
    t = s // d
    sl = np.zeros(s)
    for m in range(d):
        sl[m * t] = x[m]
    return sl


def enc_down(x, d_in, d_out, s):
    """Mirrors test_linear.cu::enc_down."""
    alpha = d_in // d_out
    kp = s // d_in
    sl = np.zeros(s)
    for m in range(d_in):
        idx = m // alpha + (m % alpha) * d_out
        sl[m * kp] = x[idx]
    return sl


def extract_simple(raw, d, s):
    """Mirrors test_linear.cu::extract_simple."""
    t = s // d
    return np.array([raw[m * t] for m in range(d)])


def extract_up(raw, d_in, d_out, s):
    """Mirrors test_linear.cu::extract_up."""
    alpha = d_out // d_in
    kp = s // d_out
    y = np.zeros(d_out)
    for m in range(d_out):
        idx = m // alpha + (m % alpha) * d_in
        y[idx] = raw[m * kp]
    return y


def run_test(d_in, d_out, s, seed=42):
    """Mirrors test_linear.cu::run exactly."""
    rng = np.random.default_rng(seed + d_in * 1000 + d_out)
    W = rng.standard_normal((d_in, d_out)) / sqrt(d_in)
    x = rng.standard_normal(d_in)
    y_exp = x @ W

    weights = install_weights(W, d_in, d_out, s)

    if d_in > d_out:
        ct_in = enc_down(x, d_in, d_out, s)
    else:
        ct_in = enc_simple(x, d_in, s)

    raw = linear_interleaved(ct_in, weights, d_in, d_out, s)

    if d_in < d_out:
        y_out = extract_up(raw, d_in, d_out, s)
    else:
        y_out = extract_simple(raw, d_out, s)

    err = np.max(np.abs(y_out - y_exp))
    return err


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=120)

    print("=" * 60)
    print("CUDA-matching simulation (linear.cu + test_linear.cu)")
    print("=" * 60)

    print("\n--- SQUARE ---")
    for d, s in [(32, 128), (64, 256), (128, 512), (256, 2048),
                  (1024, 32768)]:
        K = s // d
        nPt = d * d // s
        inRot = max(1, int(sqrt(d * d / (2 * s))))
        outRot = d * d // (s * inRot)
        err = run_test(d, d, s)
        ok = err < 1e-10
        print(f"  d={d:4d} S={s:5d} K={K:2d} nPt={nPt:3d} "
              f"inRot={inRot:2d} outRot={outRot:2d}  "
              f"err={err:.2e}  {'OK' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
