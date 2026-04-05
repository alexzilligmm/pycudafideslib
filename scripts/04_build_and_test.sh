#!/bin/bash
#SBATCH --job-name build_test_mha
#SBATCH -A IscrC_eff-SAM2
#SBATCH --time 00:30:00
#SBATCH --qos=boost_qos_dbg
#SBATCH -p boost_usr_prod
#SBATCH --mem=128G
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --error=logs/slurm/%j.err
#SBATCH --output=logs/slurm/%j.out

set -e
echo "=== Build & Test (linear + MHA) ==="
echo "Host: $(hostname)  |  Date: $(date)"

REPO="$(pwd)"
DEPS="$REPO/deps"
BUILD_DIR="$REPO/build"

module load cuda/12.6 gcc cmake
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export CUDA_NVCC="$CUDA_HOME/bin/nvcc"
export CUDAHOSTCXX=$(which g++)
export CXX=$(which g++)
export CC=$(which gcc)
export OMP_NUM_THREADS=16
export LD_LIBRARY_PATH="$DEPS/lib:$DEPS/lib64:${LD_LIBRARY_PATH:-}"

echo "CUDA_HOME: $CUDA_HOME"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || true

# ── NCCL ──────────────────────────────────────────────────────────────────
_nccl_lib=""
_nccl_inc=""
for d in "$CUDA_HOME/lib64" "$CUDA_HOME/targets/x86_64-linux/lib" /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib; do
    [ -z "$_nccl_lib" ] && [ -f "$d/libnccl.so" ] && _nccl_lib="$d"
done
for d in "$CUDA_HOME/include" "$CUDA_HOME/targets/x86_64-linux/include" /usr/include /usr/local/include; do
    [ -z "$_nccl_inc" ] && [ -f "$d/nccl.h" ] && _nccl_inc="$d"
done
export NCCL_HOME="$(dirname "$_nccl_inc")"
echo "NCCL_HOME: $NCCL_HOME"

# ── GPU arch ──────────────────────────────────────────────────────────────
_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1)
if [[ "$_cap" =~ ^[0-9]+\.[0-9]+$ ]]; then
    GPU_ARCH="$(echo "$_cap" | tr -d '.')-real"
else
    GPU_ARCH="80-real"
    echo "nvidia-smi unavailable, defaulting to $GPU_ARCH"
fi
echo "GPU_ARCH: $GPU_ARCH"

# ── Configure + Build ────────────────────────────────────────────────────
echo ""
echo "--- cmake configure ---"
cmake -S "$REPO" -B "$BUILD_DIR" \
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

echo ""
echo "--- cmake build (16 jobs) ---"
cmake --build "$BUILD_DIR" --parallel 16 2>&1

echo ""
echo "--- Built test binaries ---"
ls -lh "$BUILD_DIR/bin/"test_* 2>/dev/null

# ── Run tests ────────────────────────────────────────────────────────────
export OMP_NUM_THREADS=1

echo ""
echo "=========================================="
echo "--- Running mlp e2e ---"
echo "=========================================="
"$BUILD_DIR/bin/test_mlp_e2e" 2>&1 || true
echo "Exit code: $?"

echo ""
echo "=== All tests complete ==="
echo "Date: $(date)"
