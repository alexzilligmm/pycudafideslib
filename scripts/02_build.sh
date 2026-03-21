#!/bin/bash
#SBATCH --job-name build_cachemir
#SBATCH -A IscrC_eff-SAM2
#SBATCH --time 0:30:00
#SBATCH --qos normal
#SBATCH -p boost_usr_prod
#SBATCH --mem=32G
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --error=logs/slurm/%j.err
#SBATCH --output=logs/slurm/%j.out

# Build cuda_cachemir + all tests against the pre-installed FIDESlib.
#
# Run from: cuda-backend/
# Submit:   sbatch scripts/02_build.sh
# Or with dependency on install job:
#   JID=$(sbatch --parsable scripts/01_install_deps.sh)
#   sbatch --dependency=afterok:$JID scripts/02_build.sh

set -e
echo "=== Building cuda_cachemir ==="
echo "Host: $(hostname)  |  Date: $(date)"

export http_proxy='http://login01:3140'
export https_proxy='http://login01:3140'
export OMP_NUM_THREADS=16

# ── Paths ────────────────────────────────────────────────────────────────
DEPS=/leonardo_work/IscrC_eff-SAM2/azirilli/deps
REPO=/leonardo_work/IscrC_eff-SAM2/azirilli/cuda-cachemir/cuda-backend
BUILD_DIR="$REPO/build_sm80"

# ── Modules ──────────────────────────────────────────────────────────────
module load cuda/12.6 gcc cmake nvhpc
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export CUDAHOSTCXX=$(which g++)
export CXX=$(which g++)
export CC=$(which gcc)
echo "CUDA_HOME:   $CUDA_HOME"
echo "CUDAHOSTCXX: $CUDAHOSTCXX"
cmake --version | head -1
g++ --version | head -1

# ── GPU architecture (auto-detect, default A100=80) ───────────────────────
_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1)
if [[ "$_cap" =~ ^[0-9]+\.[0-9]+$ ]]; then
    GPU_ARCH="$(echo "$_cap" | tr -d '.')-real"
    echo "Auto-detected GPU arch: $GPU_ARCH"
else
    GPU_ARCH="80-real"
    echo "nvidia-smi unavailable, defaulting to $GPU_ARCH"
fi

# ── Verify FIDESlib is installed ─────────────────────────────────────────
if ! ls "$DEPS/lib/"*fideslib* 2>/dev/null && \
   ! ls "$DEPS/lib/"*OPENFHE* 2>/dev/null && \
   ! ls "$DEPS/lib/"*openfhe* 2>/dev/null; then
    echo "ERROR: FIDESlib not found in $DEPS"
    echo "Run 01_install_deps.sh first."
    exit 1
fi
echo "FIDESlib: $DEPS"

# ── Configure ────────────────────────────────────────────────────────────
echo ""
echo "--- cmake configure ---"
cd "$REPO"
cmake -S . -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$CC" \
    -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_CUDA_HOST_COMPILER="$CUDAHOSTCXX" \
    -DCUDA_PATH="$CUDA_HOME" \
    -DFIDESLIB_ARCH="$GPU_ARCH" \
    -DFIDESLIB_ROOT="$DEPS" \
    -DCACHEMIR_BUILD_TESTS=ON \
    2>&1

# ── Build ────────────────────────────────────────────────────────────────
echo ""
echo "--- cmake build (16 jobs) ---"
cmake --build "$BUILD_DIR" --parallel 16 2>&1

# ── Inventory ────────────────────────────────────────────────────────────
echo ""
echo "--- Built binaries ---"
ls -lh "$BUILD_DIR/bin/" 2>/dev/null || ls -lh "$BUILD_DIR/" | grep -v "^d"

echo ""
echo "=== Build complete ==="
echo "Date: $(date)"
