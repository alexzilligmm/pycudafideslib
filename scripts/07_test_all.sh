#!/bin/bash
#SBATCH --job-name run_tests
#SBATCH -A IscrC_eff-SAM2
#SBATCH --time 02:00:00
#SBATCH --qos normal
#SBATCH -p boost_usr_prod
#SBATCH --mem=32G
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --error=logs/slurm/%j.err
#SBATCH --output=logs/slurm/%j.out

# Build + run all test binaries individually (not via ctest) so each runs
# even if a previous one crashes at teardown (heap corruption).
#
# Submit:   sbatch scripts/03_run_tests.sh

echo "=== Build + run cuda_cachemir tests ==="
echo "Host: $(hostname)  |  Date: $(date)"

REPO="$(pwd)"
DEPS="$REPO/deps"
BUILD_DIR="$REPO/build"
BIN="$BUILD_DIR/bin"

module load cuda/12.6 gcc cmake
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export LD_LIBRARY_PATH="$DEPS/lib:$DEPS/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1

# --- Build ---
echo ""
echo "--- Building ---"
cmake --build "$BUILD_DIR" -j$(nproc) 2>&1 | tail -20
BUILD_RC=$?
if [ $BUILD_RC -ne 0 ]; then
    echo "ERROR: Build failed (exit $BUILD_RC)"
    exit 1
fi
echo "Build OK"

# --- Run tests ---
echo ""
echo "--- Test binaries ---"
ls -lh "$BIN/"test_* 2>/dev/null || echo "(none found)"

PASS=0
FAIL=0
for t in "$BIN"/test_*; do
    name=$(basename "$t")
    echo ""
    echo "=========================================="
    echo "  $name"
    echo "=========================================="
    "$t" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "  → $name: PASS"
        ((PASS++))
    else
        echo "  → $name: EXIT $rc"
        ((FAIL++))
    fi
done

echo ""
echo "=========================================="
echo "  SUMMARY: $PASS passed, $FAIL failed"
echo "=========================================="
echo "Date: $(date)"
