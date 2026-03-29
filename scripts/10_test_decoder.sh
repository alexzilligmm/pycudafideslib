#!/bin/bash
#SBATCH --job-name test_decoder
#SBATCH -A IscrC_eff-SAM2
#SBATCH --time 03:00:00
#SBATCH --qos normal
#SBATCH -p boost_usr_prod
#SBATCH --mem=128G
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --error=logs/slurm/%j.err
#SBATCH --output=logs/slurm/%j.out

# Run a single GPT-2 decoder layer with real interleaved weights.
# Good smoke test before running the full model.
#
# Prerequisites:
#   - weights-gpt2-interleaved/ (run 04c_generate_weights_interleaved.sh)
#   - build/bin/cuda_cachemir (run 02_build.sh)
#
# Usage:
#   sbatch scripts/05_test_decoder.sh
#   # Chained after build:
#   JID=$(sbatch --parsable scripts/02_build.sh)
#   sbatch --dependency=afterok:$JID scripts/05_test_decoder.sh

set -e
echo "=== GPT-2 Single Decoder Test (interleaved) ==="
echo "Host: $(hostname)  |  Date: $(date)"
echo "SLURM job: ${SLURM_JOB_ID:-local}"

REPO="$(pwd)"
DEPS="$REPO/deps"
BIN="$REPO/build/bin/cuda_cachemir"
WEIGHT_DIR="$REPO/weights-gpt2-interleaved"

module load cuda/12.6 gcc cmake
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export LD_LIBRARY_PATH="$DEPS/lib:$DEPS/lib64:${LD_LIBRARY_PATH:-}"

if [ ! -f "$BIN" ]; then
    echo "ERROR: $BIN not found. Run 02_build.sh first."; exit 1
fi
if [ ! -d "$WEIGHT_DIR" ]; then
    echo "ERROR: $WEIGHT_DIR not found. Run 04c_generate_weights_interleaved.sh first."; exit 1
fi

echo "Weights: $WEIGHT_DIR ($(ls $WEIGHT_DIR/ | wc -l) files)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null | sed 's/^/GPU: /' || true
echo ""

"$BIN" \
    -test Decoder \
    -model gpt2 \
    -weights "$WEIGHT_DIR/" \
    -interleaved true \
    -logN 16 \
    -hidDim 1024 -ffDim 4096 \
    -numHeads 16 -seqLen 4 \
    -numLayers 1 \
    -parallel false \
    2>&1

echo ""
echo "=== Done — $(date) ==="
