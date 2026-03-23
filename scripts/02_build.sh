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

export OMP_NUM_THREADS=16

# ── Paths ────────────────────────────────────────────────────────────────
REPO="$(pwd)"
DEPS="$REPO/deps"
BUILD_DIR="$REPO/build"

# ── Modules ──────────────────────────────────────────────────────────────
if [ -z "$CUDA_HOME" ] &>/dev/null; then
    export CUDA_HOME="/usr/local/cuda-12.6"
fi
export CUDA_NVCC="$CUDA_HOME/bin/nvcc"
export CUDAHOSTCXX=$(which g++)
export CXX=$(which g++)
export CC=$(which gcc)
echo "CUDA_HOME:   $CUDA_HOME"
echo "CUDA_NVCC:   $CUDA_NVCC"
echo "CUDAHOSTCXX: $CUDAHOSTCXX"
cmake --version | head -1
g++ --version | head -1

# ── NCCL preflight ─────────────────────────────────────────────────────────
_nccl_lib=""
_nccl_inc=""

if [ -n "$NCCL_HOME" ]; then
    if [ -f "$NCCL_HOME/lib/libnccl.so" ]; then
        _nccl_lib="$NCCL_HOME/lib"
    fi
    if [ -f "$NCCL_HOME/include/nccl.h" ]; then
        _nccl_inc="$NCCL_HOME/include"
    fi
fi

for d in \
    "$CUDA_HOME/lib64" \
    "$CUDA_HOME/targets/x86_64-linux/lib" \
    /usr/lib/x86_64-linux-gnu \
    /usr/lib \
    /usr/local/lib
do
    if [ -z "$_nccl_lib" ] && [ -f "$d/libnccl.so" ]; then
        _nccl_lib="$d"
    fi
done

for d in \
    "$CUDA_HOME/include" \
    "$CUDA_HOME/targets/x86_64-linux/include" \
    /usr/include \
    /usr/local/include
do
    if [ -z "$_nccl_inc" ] && [ -f "$d/nccl.h" ]; then
        _nccl_inc="$d"
    fi
done

if [ -z "$_nccl_lib" ] || [ -z "$_nccl_inc" ]; then
    echo ""
    echo "ERROR: NCCL not found (required for this FIDESlib build)."
    echo "Missing lib path:    ${_nccl_lib:-<not found>}"
    echo "Missing include path:${_nccl_inc:-<not found>}"
    echo ""
    echo "Install NCCL runtime and headers (e.g. libnccl2 + libnccl-dev),"
    echo "or export NCCL_HOME to a valid prefix containing lib/libnccl.so"
    echo "and include/nccl.h before running this script."
    exit 1
fi

export NCCL_HOME="$(dirname "$_nccl_inc")"
echo "NCCL_HOME:   $NCCL_HOME"
echo "NCCL lib:    $_nccl_lib"
echo "NCCL include:$_nccl_inc"


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
    -DCMAKE_CUDA_COMPILER="$CUDA_NVCC" \
    -DCMAKE_CUDA_HOST_COMPILER="$CUDAHOSTCXX" \
    -DCMAKE_LIBRARY_PATH="$_nccl_lib" \
    -DCMAKE_INCLUDE_PATH="$_nccl_inc" \
    -DCUDA_PATH="$CUDA_HOME" \
    -DCMAKE_CUDA_ARCHITECTURES="$GPU_ARCH" \
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
