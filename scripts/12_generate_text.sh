#!/bin/bash
#SBATCH --job-name gpt2_generate
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

# End-to-end encrypted GPT-2 text generation from real tokenized input.
#
# Pipeline:
#   [CLIENT/Python] tokenize prompt → token IDs
#   [CLIENT/C++]    embed last token + position → encrypt
#   [SERVER/C++]    run N decoder layers → return hidden state
#   [CLIENT/C++]    compute lm_head logits → argmax → generated token
#
# Prerequisites:
#   - weights-gpt2/ (run 04a_generate_weights.sh)
#   - include/gpt2_optimized_config.h (run gen_config.py)
#   - built binary (run 02_build.sh)
#
# Usage:
#   sbatch scripts/11_generate_text.sh
#   sbatch scripts/11_generate_text.sh --text "Once upon a time"
#   sbatch scripts/11_generate_text.sh --text "The capital of Italy is" --layers 12

set -e
T0=$(date +%s)
echo "=== Encrypted GPT-2 Text Generation ==="
echo "Host: $(hostname)  |  Date: $(date)"
echo "SLURM job: ${SLURM_JOB_ID:-local}  Node: ${SLURM_NODELIST:-$(hostname)}"

REPO="$(pwd)"
DEPS="$REPO/deps"
BUILD_DIR="$REPO/build"
BIN="$BUILD_DIR/bin/cuda_cachemir"

# ── Parse args ──
PROMPT_TEXT="The capital of France is"
NUM_LAYERS=12
NUM_GEN=1
LOG_N=16
for arg in "$@"; do
    case $arg in
        --text=*)   PROMPT_TEXT="${arg#*=}" ;;
        --text)     shift; PROMPT_TEXT="$1" ;;
        --layers=*) NUM_LAYERS="${arg#*=}" ;;
        --layers)   shift; NUM_LAYERS="$1" ;;
        --numgen=*) NUM_GEN="${arg#*=}" ;;
        --numgen)   shift; NUM_GEN="$1" ;;
        --logN=*)   LOG_N="${arg#*=}" ;;
        --logN)     shift; LOG_N="$1" ;;
    esac
done

# Weight dir: always use interleaved (CacheMIR sparse-to-sparse)
WEIGHT_DIR="$REPO/weights-gpt2-interleaved"

module load cuda/12.6 gcc cmake
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export LD_LIBRARY_PATH="$DEPS/lib:$DEPS/lib64:${LD_LIBRARY_PATH:-}"
export http_proxy='http://login02:3140'
export https_proxy='http://login02:3140'

# ── Checks ──
if [ ! -f "$BIN" ]; then
    echo "ERROR: $BIN not found. Run 02_build.sh first."; exit 1
fi
if [ ! -d "$WEIGHT_DIR" ]; then
    echo "ERROR: $WEIGHT_DIR not found."
    echo "  Run: sbatch scripts/04c_generate_weights_interleaved.sh"
    exit 1
fi

# ── Pre-flight info ──
echo ""
echo "--- Configuration ---"
echo "  Binary  : $BIN  ($(date -r "$BIN" '+%Y-%m-%d %H:%M:%S'))"
echo "  Weights : $WEIGHT_DIR  ($(ls $WEIGHT_DIR/ | wc -l) files)"
echo "  Layers  : $NUM_LAYERS   NumGen: $NUM_GEN"
echo "  logN=$LOG_N  hidDim=1024  ffDim=4096  realHidDim=768  realFfDim=3072"
CONFIG_H="$REPO/include/gpt2_optimized_config.h"
if [ -f "$CONFIG_H" ]; then
    echo "  Config  : $(head -3 $CONFIG_H | tr '\n' ' ')"
fi
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null | sed 's/^/  GPU     : /' || true

# ── Step 1: Tokenize prompt ──
echo ""
echo "=== [$(date '+%H:%M:%S')] Step 1/3 — Tokenizing prompt ==="
echo "  Prompt: \"$PROMPT_TEXT\""
INPUT_DIR="$REPO/input_data/generate"
uv run python scripts/tokenize_input.py \
    --text "$PROMPT_TEXT" \
    --out_dir "$INPUT_DIR"

PROMPT_IDS=$(uv run python -c "
import json; m=json.load(open('$INPUT_DIR/metadata.json'))
print(m['prompt_ids_csv'])
")
N_TOKENS=$(echo "$PROMPT_IDS" | tr ',' '\n' | wc -l)
echo "  Token IDs ($N_TOKENS tokens): $PROMPT_IDS"

# ── Step 2: Run encrypted inference ──
echo ""
echo "=== [$(date '+%H:%M:%S')] Step 2/3 — Encrypted GPT-2 generation ==="
echo "  [CLIENT] Embedding last token + encrypting → ciphertext"
echo "  [SERVER] Running $NUM_LAYERS decoder layer(s) under encryption..."
echo "  (each layer: Norm1 → QKV → Softmax → AttnV → Out → Norm2 → Up → GELU → Down)"
echo ""
T_INF=$(date +%s)
"$BIN" -test Generate -model gpt2 \
    -weights "$WEIGHT_DIR/" \
    -interleaved true \
    -logN "$LOG_N" -hidDim 1024 -ffDim 4096 \
    -realHidDim 768 -realFfDim 3072 \
    -numLayers "$NUM_LAYERS" \
    -numGen "$NUM_GEN" \
    -prompt "$PROMPT_IDS" \
    -parallel false \
    2>&1
T_INF_END=$(date +%s)

# ── Step 3: Report ──
echo ""
echo "=== [$(date '+%H:%M:%S')] Step 3/3 — Done ==="
echo "  Inference wall-clock : $((T_INF_END - T_INF)) s"
echo "  Total wall-clock     : $((T_INF_END - T0)) s"
echo ""
echo "=== Done — $(date) ==="
