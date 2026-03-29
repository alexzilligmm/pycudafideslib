#!/bin/bash
#SBATCH --job-name gen_vectors
#SBATCH -A IscrC_eff-SAM2
#SBATCH --time 00:15:00
#SBATCH --qos normal
#SBATCH -p boost_usr_prod
#SBATCH --mem=16G
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --error=logs/slurm/%j.err
#SBATCH --output=logs/slurm/%j.out

# Generate test vectors from PyTorch GPT-2 (layer 0).
# Downloads HuggingFace GPT-2 weights, runs one layer in plaintext,
# saves input + expected outputs as .txt files for the C++ test.
#
# Output: test_vectors/layer0/*.txt
#
# Usage:  sbatch scripts/03c_generate_test_vectors.sh

set -e
echo "=== Generate GPT-2 Test Vectors ==="
echo "Host: $(hostname)  |  Date: $(date)"

REPO="$(pwd)"
TEST_VEC_DIR="$REPO/test_vectors"

export http_proxy='http://login02:3140'
export https_proxy='http://login02:3140'

if [ -f "$REPO/.venv/bin/activate" ]; then
    source "$REPO/.venv/bin/activate"
fi

# Paper params: logN=16 only (S=32768).
# GPT-2 small padded dims: hid=1024, ff=4096.

LOGN=16
SLOTS=$((1 << (LOGN - 1)))
OUTDIR="${TEST_VEC_DIR}_logN${LOGN}"
echo ""
echo "--- logN=${LOGN}  slots=${SLOTS}  out=${OUTDIR} ---"
uv run python tests/generate_gpt2_test_vectors.py \
    --out_dir "$OUTDIR" \
    --layers 1 \
    --slots "$SLOTS" \
    --hid 1024 \
    --ff 4096 \
    2>&1
echo "  Files:"
ls -lh "$OUTDIR/layer0/" 2>/dev/null | head -5 || echo "  (no files)"
echo "  Interleaved:"
ls -lh "$OUTDIR/layer0/interleaved/" 2>/dev/null | head -5 || echo "  (no files)"

echo ""
echo "=== Done ==="
echo "Date: $(date)"
