#!/bin/bash
#SBATCH --job-name gen_wts_int
#SBATCH -A IscrC_eff-SAM2
#SBATCH --time 00:30:00
#SBATCH --qos normal
#SBATCH -p boost_usr_prod
#SBATCH --mem=32G
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --error=logs/slurm/%j.err
#SBATCH --output=logs/slurm/%j.out

# Generate full-interleaved GPT-2 weights for logN=16 (S=32768).
# All 6 projections → CacheMIR interleaved (d_in*d_out/S plaintexts).
#   Q,K,V,Out (square): 4 × hD²/S = 4×32 = 128 Ptx
#   Up (expand hD→fD):      hD*fD/S = 128 Ptx
#   Down (shrink fD→hD):    fD*hD/S = 128 Ptx
# Total: 384 Ptx/layer  (vs 4096 standard → huge reduction)

set -e
echo "=== Generate Hybrid-Interleaved GPT-2 Weights — logN=16 (S=32768) ==="
echo "Host: $(hostname)  |  Date: $(date)"

REPO="$(pwd)"
WEIGHT_DIR="$REPO/weights-gpt2-interleaved"

export http_proxy='http://login02:3140'
export https_proxy='http://login02:3140'

echo ""
echo "--- Output: $WEIGHT_DIR ---"
echo "--- Params: slots=32768  hidDim=1024  ffDim=4096  --interleaved ---"
echo ""

uv run python prepare_gpt2_weights.py \
    --model gpt2 \
    --out_dir "$WEIGHT_DIR" \
    --slots 32768 \
    --hidDim 1024 \
    --ffDim 4096 \
    --interleaved \
    2>&1

echo ""
echo "--- Weight directory ---"
du -sh "$WEIGHT_DIR" 2>/dev/null || echo "(not found)"
ls "$WEIGHT_DIR/" 2>/dev/null | head -5
echo ""
echo "--- Ptx counts (per-layer sample) ---"
ls "$WEIGHT_DIR/layer0/" 2>/dev/null | wc -l || true
du -sh "$WEIGHT_DIR/layer0/" 2>/dev/null || true

echo ""
echo "=== Done — $(date) ==="
