#!/bin/bash
# submit_all.sh – chain all four jobs with afterok dependencies.
#
# Usage (run from cuda-backend/):
#   bash scripts/submit_all.sh            # full pipeline
#   bash scripts/submit_all.sh --skip-check   # skip env check
#   bash scripts/submit_all.sh --skip-install # skip dep install (already done)
#
# The script prints each job ID and the final ctest job ID so you can
# monitor with:  sacct -j <JID>  or  tail -f logs/slurm/<JID>.out

set -e

SKIP_CHECK=false
SKIP_INSTALL=false
for arg in "$@"; do
    case "$arg" in
        --skip-check)   SKIP_CHECK=true ;;
        --skip-install) SKIP_INSTALL=true ;;
    esac
done

# ── Ensure log directory exists ───────────────────────────────────────────
mkdir -p logs/slurm

DEP=""  # current dependency chain

# ── 00: check environment ─────────────────────────────────────────────────
if [ "$SKIP_CHECK" = false ]; then
    JID0=$(sbatch --parsable scripts/00_check_env.sh)
    echo "Submitted 00_check_env     job=$JID0"
    DEP="--dependency=afterok:${JID0}"
fi

# ── 01: install OpenFHE + FIDESlib ────────────────────────────────────────
if [ "$SKIP_INSTALL" = false ]; then
    JID1=$(sbatch --parsable $DEP scripts/01_install_deps.sh)
    echo "Submitted 01_install_deps  job=$JID1"
    DEP="--dependency=afterok:${JID1}"
fi

# ── 02: build cuda_cachemir + tests ───────────────────────────────────────
JID2=$(sbatch --parsable $DEP scripts/02_build.sh)
echo "Submitted 02_build         job=$JID2"
DEP="--dependency=afterok:${JID2}"

# ── 03: run tests ─────────────────────────────────────────────────────────
JID3=$(sbatch --parsable $DEP scripts/03_run_tests.sh)
echo "Submitted 03_run_tests     job=$JID3"

echo ""
echo "Pipeline submitted.  Monitor with:"
echo "  squeue -u \$USER"
echo "  sacct -j $JID3"
echo "  tail -f logs/slurm/${JID3}.out"
