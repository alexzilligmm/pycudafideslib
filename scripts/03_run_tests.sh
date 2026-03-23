#!/bin/bash
#SBATCH --job-name run_tests
#SBATCH -A IscrC_eff-SAM2
#SBATCH --time 0:30:00
#SBATCH --qos normal
#SBATCH -p boost_usr_prod
#SBATCH --mem=32G
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --error=logs/slurm/%j.err
#SBATCH --output=logs/slurm/%j.out

# Run all ctest suites against the pre-built binaries.
#
# Run from: cuda-backend/
# Submit:   sbatch scripts/03_run_tests.sh
# Or with dependency on build job:
#   JID=$(sbatch --parsable scripts/02_build.sh)
#   sbatch --dependency=afterok:$JID scripts/03_run_tests.sh

set -e
echo "=== Running cuda_cachemir tests ==="
echo "Host: $(hostname)  |  Date: $(date)"

# ── Paths ────────────────────────────────────────────────────────────────
REPO="$(pwd)"
DEPS="$REPO/deps"
BUILD_DIR="$REPO/build"

# ── Modules ──────────────────────────────────────────────────────────────
export CUDA_HOME="/usr/local/cuda-12.6"

# ── Sanitize environment (remove Conda libstdc++ conflicts) ─────────────────
# Unset Conda variables that may inject older compiler stubs
# Remove /opt/miniconda3 from PATH to force system compilers/libraries
export PATH="/usr/local/cuda-12.6/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
# Ensure system libraries are prioritized (e.g., system libstdc++.so.6 with newer GLIBCXX symbols)
if [ -d "/usr/lib/x86_64-linux-gnu" ]; then
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
fi

# ── Verify build exists ───────────────────────────────────────────────────
if [ ! -d "$BUILD_DIR" ]; then
    echo "ERROR: Build directory $BUILD_DIR not found."
    echo "Run 02_build.sh first."
    exit 1
fi

# Export shared-library path so FIDESlib/OpenFHE .so files are found
export LD_LIBRARY_PATH="$DEPS/lib:$DEPS/lib64:${LD_LIBRARY_PATH:-}"

# ── List available test binaries ──────────────────────────────────────────
echo ""
echo "--- Test binaries ---"
ls -lh "$BUILD_DIR/bin/"test_* 2>/dev/null || echo "(no test_* binaries found in bin/)"

# ── ctest (runs all registered tests) ────────────────────────────────────
echo ""
echo "--- ctest (verbose) ---"
ctest --test-dir "$BUILD_DIR" -V --output-on-failure 2>&1

# ── Quick smoke-test of the main binary ───────────────────────────────────
echo ""
echo "--- Smoke: cuda_cachemir -test Ops -logN 12 ---"
"$BUILD_DIR/bin/cuda_cachemir" -test Ops -logN 12 2>&1

echo ""
echo "=== All tests complete ==="
echo "Date: $(date)"
