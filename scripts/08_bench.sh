#!/bin/bash
#SBATCH --job-name bench_fhe
#SBATCH -A IscrC_eff-SAM2
#SBATCH --time 08:00:00
#SBATCH --qos normal
#SBATCH -p boost_usr_prod
#SBATCH --mem=128G
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --error=logs/slurm/%j.err
#SBATCH --output=logs/slurm/%j.out

# Benchmark all FHE operations per-level for GPT-2.
# Uses BenchAll mode (single CKKS context, sweeps all ops) to avoid
# regenerating the ~31GB GPU context for each (op, level) pair.
#
# Produces: bench_all_output.txt → parsed into new_data.csv
#
# Usage:
#   sbatch scripts/02b_bench.sh                  # full bench + placement
#   sbatch scripts/02b_bench.sh --skip-boot      # skip bootstrap.py at end
#
# Requires: build/bin/cuda_cachemir (run 02_build.sh first)
#   JID=$(sbatch --parsable scripts/02_build.sh)
#   sbatch --dependency=afterok:$JID scripts/02b_bench.sh

set -euo pipefail
echo "=== FHE Operation Benchmark (BenchAll) ==="
echo "Host: $(hostname)  |  Date: $(date)"
echo "SLURM job: ${SLURM_JOB_ID:-local}"

REPO="$(pwd)"
DEPS="$REPO/deps"
LOGN=16
MAX_LEVEL=13

SKIP_BOOT=0
for arg in "$@"; do
    case $arg in
        --skip-boot) SKIP_BOOT=1 ;;
    esac
done

# ── Modules & env ────────────────────────────────────────────────────────
module load cmake/3.27.9 cuda/12.6 gcc/12.2.0
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export OMP_NUM_THREADS=16
export http_proxy='http://login02:3140'
export https_proxy='http://login02:3140'
export LD_LIBRARY_PATH="$DEPS/lib:$DEPS/lib64:${LD_LIBRARY_PATH:-}"

if [ -f "$REPO/.venv/bin/activate" ]; then
    source "$REPO/.venv/bin/activate"
fi

# ── Verify binary exists ────────────────────────────────────────────────
BIN="$REPO/build/bin/cuda_cachemir"
if [ ! -x "$BIN" ]; then
    echo "ERROR: $BIN not found. Run 02_build.sh first."
    exit 1
fi

# ── Run BenchAll (single context, all ops) ──────────────────────────────
BENCH_RAW="$REPO/bench_all_output.txt"
echo ""
if [ -f "$BENCH_RAW" ]; then
    DONE=$(grep -c "^Consumed" "$BENCH_RAW" 2>/dev/null || echo 0)
    echo "Resuming BenchAll ($DONE sections already done in $BENCH_RAW)"
else
    echo "Starting fresh BenchAll"
fi
echo "Model=gpt2, logN=$LOGN, maxLevel=$MAX_LEVEL"
echo ""

"$BIN" \
    -test BenchAll \
    -model gpt2 \
    -logN $LOGN \
    -hidDim 1024 \
    -ffDim 4096 \
    -seqLen 4 \
    -numHeads 16 \
    -maxLevel $MAX_LEVEL \
    -parallel true \
    -resumeFile "$BENCH_RAW" \
    2>&1 | tee -a "$BENCH_RAW"

# ── Parse BenchAll output into new_data.csv ─────────────────────────────
echo ""
echo "Parsing bench output → new_data.csv..."
uv run python -c "
import re

lines = open('$BENCH_RAW').readlines()
data = {}
current_op = None
current_level = 0

for line in lines:
    m = re.match(r'--- (\w+) @ (?:btp)?[Ll]evel (\d+) ---', line)
    if m:
        current_op = m.group(1)
        current_level = int(m.group(2))
        continue
    m = re.match(r'Consumed\s+([\d.]+)\s+seconds', line)
    if m and current_op:
        t = float(m.group(1))
        data.setdefault(current_op, {})[current_level] = t

max_lvl = $MAX_LEVEL
with open('$REPO/new_data.csv', 'w') as f:
    f.write('module')
    for i in range(1, max_lvl+1):
        f.write(f'\t{i:.2f}')
    f.write('\n')
    for op in ['QKV','QK_T','AttnV','GELU','Up','Down','Cache','CtMult','Softmax','SqrtNt','SqrtGold']:
        f.write(op)
        for i in range(1, max_lvl+1):
            t = data.get(op, {}).get(i, 0.0)
            f.write(f'\t{t:.6f}')
        f.write('\n')
print('new_data.csv written with', len(data), 'ops')
" 2>&1

echo ""
cat "$REPO/new_data.csv"

# ── Optional: run bootstrap placement optimizer ─────────────────────────
if [ $SKIP_BOOT -eq 0 ]; then
    echo ""
    echo "=== Running bootstrap.py placement optimizer ==="
    uv run python bootstrap.py \
        --file "$REPO/new_data.csv" \
        --model gpt2 \
        --max-level $MAX_LEVEL \
        --logN $LOGN \
        --uniform \
        --json "$REPO/new_data_placement.json" \
        2>&1 || echo "bootstrap.py failed (non-fatal)"
fi

echo ""
echo "=== Done — $(date) ==="
