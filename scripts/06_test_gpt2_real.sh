#!/bin/bash
#SBATCH --job-name test_gpt2
#SBATCH -A IscrC_eff-SAM2
#SBATCH --time 00:30:00
#SBATCH --qos normal
#SBATCH -p boost_usr_prod
#SBATCH --mem=128G
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --error=logs/slurm/%j.err
#SBATCH --output=logs/slurm/%j.out

# Run test_gpt2_real: validates C++ linear ops against PyTorch ground truth.
# Requires: test vectors from 03c_generate_test_vectors.sh + build from 02_build.sh
#
# Usage:
#   sbatch scripts/03d_test_gpt2_real.sh
#   # Or chained:
#   JID=$(sbatch --parsable scripts/03c_generate_test_vectors.sh)
#   sbatch --dependency=afterok:$JID scripts/03d_test_gpt2_real.sh

set -e
echo "=== test_gpt2_real (C++ vs PyTorch) ==="
echo "Host: $(hostname)  |  Date: $(date)"

REPO="$(pwd)"
DEPS="$REPO/deps"
BUILD_DIR="$REPO/build"
BIN="$BUILD_DIR/bin/test_gpt2_real"

module load cuda/12.6 gcc cmake
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export LD_LIBRARY_PATH="$DEPS/lib:$DEPS/lib64:${LD_LIBRARY_PATH:-}"

if [ ! -f "$BIN" ]; then
    echo "ERROR: $BIN not found. Run 02_build.sh first."
    exit 1
fi

if [ ! -d "$REPO/test_vectors_logN16/layer0" ]; then
    echo "ERROR: No test vectors found. Run 03c_generate_test_vectors.sh first."
    exit 1
fi
echo "Test vector dir: test_vectors_logN16/layer0"
ls "$REPO/test_vectors_logN16/layer0/" | wc -l
echo "files"

echo ""
# Run each test suite in a separate process to avoid FIDESlib GPU
# memory corruption when creating 3+ CKKS contexts in one process.
# Run each test suite in its own process to avoid FIDESlib GPU
# heap corruption when 3+ CKKS contexts are created sequentially.
# FIDESlib also corrupts the heap during cleanup AFTER all tests pass,
# so we check gtest output for "[  PASSED  ]" to distinguish real
# failures from exit-time crashes.
# Paper-params pipeline: logN=16 only
SUITES=(GPT2LinearTest GPT2DecoderTest)
FAILED=0
for SUITE in "${SUITES[@]}"; do
    echo "--- Running $SUITE ---"
    OUTPUT=$("$BIN" --gtest_filter="${SUITE}.*" 2>&1) || true
    echo "$OUTPUT"
    # Check if any test actually FAILED (not just exit-time crash)
    if echo "$OUTPUT" | grep -q "\[  FAILED  \]"; then
        echo "  $SUITE: FAIL (test assertion failed)"
        FAILED=$((FAILED + 1))
    else
        echo "  $SUITE: PASS"
    fi
    echo ""
done

if [ $FAILED -gt 0 ]; then
    echo "=== $FAILED suite(s) had test failures ==="
    exit 1
fi
echo "=== All suites PASSED ==="
echo "Date: $(date)"
