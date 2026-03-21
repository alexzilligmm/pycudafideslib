#!/bin/bash
#SBATCH --job-name check_env
#SBATCH -A IscrC_eff-SAM2
#SBATCH --time 0:10:00
#SBATCH --qos boost_qos_dbg
#SBATCH -p boost_usr_prod
#SBATCH --mem=16G
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --error=logs/slurm/%j.err
#SBATCH --output=logs/slurm/%j.out

# Probe the build environment before committing to a full install.
# Run from: cuda-backend/scripts/
# Submit:   sbatch scripts/00_check_env.sh

set -e
echo "=== Environment check ==="
echo "Host: $(hostname)"
echo "Date: $(date)"

export http_proxy='http://login01:3140'
export https_proxy='http://login01:3140'

# ── CUDA ─────────────────────────────────────────────────────────────────
echo ""
echo "--- CUDA ---"
module avail cuda 2>&1 | grep -i cuda || echo "(no cuda modules found)"
module load cuda/12.1 2>/dev/null && echo "Loaded cuda/12.1" || \
  { module load cuda 2>/dev/null && echo "Loaded default cuda"; }
nvcc --version 2>/dev/null || echo "nvcc not found"
nvidia-smi 2>/dev/null | head -12 || echo "nvidia-smi not found"

# ── CMake ────────────────────────────────────────────────────────────────
echo ""
echo "--- CMake ---"
module avail cmake 2>&1 | grep -i cmake | head -5 || true
cmake --version 2>/dev/null | head -1 || echo "cmake not in PATH"
# Try loading a newer cmake if needed (requires 3.25+)
for v in 3.29 3.28 3.27 3.26 3.25; do
    module load cmake/${v} 2>/dev/null && echo "Loaded cmake/${v}" && break
done
cmake --version 2>/dev/null | head -1 || echo "cmake still not found"

# ── Compilers ────────────────────────────────────────────────────────────
echo ""
echo "--- Compilers ---"
# Clang (FIDESlib prefers 17+)
for v in 18 17 16; do
    if command -v clang++-${v} &>/dev/null; then
        echo "Found clang++-${v}"
        break
    fi
done
clang++ --version 2>/dev/null | head -1 || echo "clang++ not in PATH"

# GCC fallback (needs 11+ for C++20)
g++ --version 2>/dev/null | head -1 || echo "g++ not in PATH"
module avail gcc 2>&1 | grep -i gcc | head -5 || true
for v in 13 12 11; do
    module load gcc/${v} 2>/dev/null && echo "Loaded gcc/${v}" && break
done
g++ --version 2>/dev/null | head -1 || echo "g++ still not found"

# ── Python (for bootstrap.py) ────────────────────────────────────────────
echo ""
echo "--- Python ---"
python3 --version 2>/dev/null || echo "python3 not in PATH"

# ── Git ──────────────────────────────────────────────────────────────────
echo ""
echo "--- Git ---"
git --version 2>/dev/null || echo "git not found"
# Test proxy connectivity
echo "Testing proxy connectivity..."
curl -s --proxy http://login01:3140 \
     https://api.github.com 2>&1 | grep -q '"current_user_url"' \
     && echo "GitHub reachable via proxy" \
     || echo "GitHub NOT reachable – check proxy"

# ── OpenFHE / FIDESlib already installed? ────────────────────────────────
echo ""
echo "--- Existing installs ---"
DEPS=/leonardo_work/IscrC_eff-SAM2/azirilli/deps
ls "$DEPS/lib/libOPENFHE"*.so 2>/dev/null && echo "OpenFHE found in $DEPS" || echo "OpenFHE not yet installed"
ls "$DEPS/lib/libfideslib"*.a  2>/dev/null && echo "FIDESlib found in $DEPS" || echo "FIDESlib not yet installed"

echo ""
echo "=== Check complete ==="
