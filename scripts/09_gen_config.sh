#!/bin/bash
#SBATCH --job-name gen_config
#SBATCH -A IscrC_eff-SAM2
#SBATCH --time 00:30:00
#SBATCH --qos normal
#SBATCH -p boost_usr_prod
#SBATCH --mem=16G
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --error=logs/slurm/%j.err
#SBATCH --output=logs/slurm/%j.out

# Generate optimized C++ config header from benchmark placement JSON.
# Runs: bootstrap.py (if needed) → gen_config.py → gpt2_optimized_config.h
#
# Usage:
#   sbatch scripts/02c_gen_config.sh                    # uses existing placement JSON
#   sbatch scripts/02c_gen_config.sh --run-bootstrap    # re-run bootstrap.py first
#
# Requires: new_data_placement.json (from 02b_bench.sh) OR new_data.csv + --run-bootstrap
#
# Chain from bench:
#   JID=$(sbatch --parsable scripts/02b_bench.sh)
#   sbatch --dependency=afterok:$JID scripts/02c_gen_config.sh

set -euo pipefail
echo "=== Generate Optimized Config ==="
echo "Host: $(hostname)  |  Date: $(date)"
echo "SLURM job: ${SLURM_JOB_ID:-local}"

REPO="$(pwd)"
LOGN=16
MAX_LEVEL=13

RUN_BOOTSTRAP=0
for arg in "$@"; do
    case $arg in
        --run-bootstrap) RUN_BOOTSTRAP=1 ;;
    esac
done

export http_proxy='http://login02:3140'
export https_proxy='http://login02:3140'

if [ -f "$REPO/.venv/bin/activate" ]; then
    source "$REPO/.venv/bin/activate"
fi

PLACEMENT_JSON="$REPO/new_data_placement.json"
CONFIG_H="$REPO/include/gpt2_optimized_config.h"

# ── Optional: re-run bootstrap.py ────────────────────────────────────────
if [ $RUN_BOOTSTRAP -eq 1 ]; then
    if [ ! -f "$REPO/new_data.csv" ]; then
        echo "ERROR: new_data.csv not found. Run 02b_bench.sh first."
        exit 1
    fi
    echo "Running bootstrap.py (placement optimizer)..."
    uv run python bootstrap.py \
        --file "$REPO/new_data.csv" \
        --model gpt2 \
        --max-level $MAX_LEVEL \
        --logN $LOGN \
        --uniform \
        --json "$PLACEMENT_JSON" \
        2>&1
fi

# ── Generate C++ header ─────────────────────────────────────────────────
if [ ! -f "$PLACEMENT_JSON" ]; then
    echo "ERROR: $PLACEMENT_JSON not found."
    echo "Run 02b_bench.sh first, or use --run-bootstrap with new_data.csv."
    exit 1
fi

echo "Running gen_config.py → $CONFIG_H"
uv run python gen_config.py \
    "$PLACEMENT_JSON" \
    --logN $LOGN \
    --num-layers 12 \
    -o "$CONFIG_H" \
    2>&1

echo ""
echo "Generated config header:"
head -30 "$CONFIG_H"
echo ""
echo "=== Done — rebuild with 02_build.sh to use new config ==="
