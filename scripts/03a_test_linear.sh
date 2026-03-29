#!/bin/bash
#SBATCH --job-name test_linear
#SBATCH -A IscrC_eff-SAM2
#SBATCH --time 01:00:00
#SBATCH --qos normal
#SBATCH -p boost_usr_prod
#SBATCH --mem=128G
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --error=logs/slurm/%j.err
#SBATCH --output=logs/slurm/%j.out

set -e
echo "=== test_linear ==="
echo "Host: $(hostname)  |  Date: $(date)"

REPO="$(pwd)"
DEPS="$REPO/deps"
BUILD_DIR="$REPO/build"
BIN="$BUILD_DIR/bin/test_linear"

module load cuda/12.6 gcc cmake
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export LD_LIBRARY_PATH="$DEPS/lib:$DEPS/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1

echo "--- Rebuild ---"
cmake --build "$BUILD_DIR" --target test_linear -j$(nproc) 2>&1 | tail -5

echo ""
echo "--- Running test_linear ---"
"$BIN" 2>&1
echo "Exit code: $?"
echo "Date: $(date)"
