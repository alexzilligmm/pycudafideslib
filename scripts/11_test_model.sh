#!/bin/bash
#SBATCH --job-name test_model
#SBATCH -A IscrC_eff-SAM2
#SBATCH --time 08:00:00
#SBATCH --qos normal
#SBATCH -p boost_usr_prod
#SBATCH --mem=128G
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --error=logs/slurm/%j.err
#SBATCH --output=logs/slurm/%j.out

# Run the full GPT-2 model (all 12 decoder layers) with real interleaved weights.
# Use DiagModel for verbose per-layer diagnostics.
#
# Prerequisites:
#   - weights-gpt2-interleaved/ (run 04c_generate_weights_interleaved.sh)
#   - build/bin/cuda_cachemir (run 02_build.sh)
#
# Usage:
#   sbatch scripts/06_test_model.sh
#   sbatch scripts/06_test_model.sh --layers 3    # fewer layers for quicker test
#   sbatch scripts/06_test_model.sh --diag         # diagnostic mode (verbose per-layer)

set -e
echo "=== GPT-2 Full Model Test (interleaved) ==="
echo "Host: $(hostname)  |  Date: $(date)"
echo "SLURM job: ${SLURM_JOB_ID:-local}"

REPO="$(pwd)"
DEPS="$REPO/deps"
BIN="$REPO/build/bin/cuda_cachemir"
WEIGHT_DIR="$REPO/weights-gpt2-interleaved"

NUM_LAYERS=12
TEST_MODE="Model"
for arg in "$@"; do
    case $arg in
        --layers=*) NUM_LAYERS="${arg#*=}" ;;
        --diag)     TEST_MODE="DiagModel" ;;
    esac
done

module load cuda/12.6 gcc cmake
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export LD_LIBRARY_PATH="$DEPS/lib:$DEPS/lib64:${LD_LIBRARY_PATH:-}"

if [ ! -f "$BIN" ]; then
    echo "ERROR: $BIN not found. Run 02_build.sh first."; exit 1
fi
if [ ! -d "$WEIGHT_DIR" ]; then
    echo "ERROR: $WEIGHT_DIR not found. Run 04c_generate_weights_interleaved.sh first."; exit 1
fi

echo "Mode: $TEST_MODE  Layers: $NUM_LAYERS"
echo "Weights: $WEIGHT_DIR ($(ls $WEIGHT_DIR/ | wc -l) files)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null | sed 's/^/GPU: /' || true
echo ""

T0=$(date +%s)
"$BIN" \
    -test "$TEST_MODE" \
    -model gpt2 \
    -weights "$WEIGHT_DIR/" \
    -interleaved true \
    -logN 16 \
    -hidDim 1024 -ffDim 4096 \
    -numHeads 16 -seqLen 4 \
    -numLayers "$NUM_LAYERS" \
    -parallel false \
    2>&1
T1=$(date +%s)

echo ""
echo "Wall-clock: $((T1 - T0)) seconds"
echo "=== Done — $(date) ==="
