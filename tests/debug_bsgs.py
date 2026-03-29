"""
Diagnostic: load test_gpt2_real vectors and simulate C++ BSGS
to find where the mismatch is.
"""
import numpy as np
import math
import sys

DIR = "test_vectors/layer0"
S = 32768
HD = 1024

def load_vec(path):
    return np.loadtxt(path).flatten()

def load_mat(path):
    return np.loadtxt(path)

def find_bsgs_factors(d):
    r = int(math.sqrt(d))
    while r > 1 and d % r != 0:
        r -= 1
    return r, d // r

print("Loading test vectors...")
norm1 = load_vec(f"{DIR}/expected_norm1.txt")
Wq = load_mat(f"{DIR}/Wq.txt")
bq = load_vec(f"{DIR}/bq.txt")
expected_q = load_vec(f"{DIR}/expected_q.txt")

print(f"  norm1: shape={norm1.shape}, first 4: {norm1[:4]}")
print(f"  Wq: shape={Wq.shape}")
print(f"  bq: shape={bq.shape}, first 4: {bq[:4]}")
print(f"  expected_q: first 4: {expected_q[:4]}")

# Check periodicity of input
print(f"\n  norm1 periodic? norm1[0]={norm1[0]:.6f}, norm1[{HD}]={norm1[HD]:.6f}, norm1[{2*HD}]={norm1[2*HD]:.6f}")

# Check periodicity of weights
print(f"  Wq[0] periodic? Wq[0,0]={Wq[0,0]:.6f}, Wq[0,{HD}]={Wq[0,HD]:.6f}")

inRot, outRot = find_bsgs_factors(HD)
print(f"\n  BSGS: inRot={inRot}, outRot={outRot}, d={HD}")

# ── Simulate C++ BSGS (real mode, no pre/post broadcast) ──
print("\nSimulating C++ BSGS (real mode, no pre/post broadcast)...")
baby_step = 1
giant_stride = inRot

# Baby step rotations
ctRot = [None] * inRot
ctRot[0] = norm1.copy()
for i in range(1, inRot):
    ctRot[i] = np.roll(ctRot[i-1], -baby_step)

n_weights = Wq.shape[0]
print(f"  n_weights={n_weights}")
partSum = [None] * n_weights

# Multiply
for i in range(n_weights):
    partSum[i] = ctRot[i % inRot] * Wq[i]

# Input sum within blocks
for i in range(n_weights):
    if i % inRot > 0:
        partSum[i - i % inRot] = partSum[i - i % inRot] + partSum[i]

# Giant step
for j in range(1, outRot):
    partSum[j * inRot] = np.roll(partSum[j * inRot], -(j * giant_stride))

# Output sum
result = partSum[0].copy()
for j in range(1, outRot):
    result = result + partSum[j * inRot]

# Add bias
result_with_bias = result + bq

print(f"\n  BSGS result first 4: {result[:4]}")
print(f"  BSGS+bias first 4: {result_with_bias[:4]}")
print(f"  Expected first 4: {expected_q[:4]}")

err_no_bias = np.max(np.abs(result[:HD] - expected_q[:HD]))
err_with_bias = np.max(np.abs(result_with_bias[:HD] - expected_q[:HD]))
print(f"\n  Max error (no bias): {err_no_bias:.6e}")
print(f"  Max error (with bias): {err_with_bias:.6e}")

# ── Also verify: direct matmul of the absorbed W against norm1 ──
# The expected_q should be Wq_abs @ norm1 + bq_abs
# But we only have the BSGS-packed weights. Let's unpack diagonal 0 to check.
print(f"\n  Wq[0][:4] = {Wq[0,:4]}")  # diagonal 0 at slots 0-3
print(f"  Wq[0][{HD}:{HD+4}] = {Wq[0,HD:HD+4]}")  # should be same (periodic)

# Check: what does the test output file say (expected output from Python pipeline)?
# expected_q = Wq_abs @ norm1 + bq_abs (replicated to S)
# So expected_q[s] should be periodic with period HD
print(f"\n  expected_q periodic? [0]={expected_q[0]:.6f}, [{HD}]={expected_q[HD]:.6f}")

# Check result periodicity
print(f"  BSGS result periodic? [0]={result[0]:.6f}, [{HD}]={result[HD]:.6f}")

# ── Test with pack_weights_real to verify packing consistency ──
print("\n--- Verifying packing consistency ---")
# Re-derive what W.T should be from the packed Wq
# Wq[k][s] = W.T[(s + k%inRot) % d, (s - (k//inRot)*inRot) % d]
# At s=0, k=0: Wq[0][0] = W.T[0, 0]
# At s=0, k=1 (i=1, j=0): Wq[1][0] = W.T[1, 0]
# At s=0, k=32 (i=0, j=1): Wq[32][0] = W.T[0, -32 % 1024] = W.T[0, 992]
print(f"  Wq[0][0] = {Wq[0,0]:.6f}  (should be W.T[0,0])")
print(f"  Wq[1][0] = {Wq[1,0]:.6f}  (should be W.T[1,0])")
print(f"  Wq[32][0] = {Wq[32,0]:.6f}  (should be W.T[0,992])")
