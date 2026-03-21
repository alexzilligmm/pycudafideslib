#!/bin/bash
#SBATCH --job-name install_deps
#SBATCH -A IscrC_eff-SAM2
#SBATCH --time 1:30:00
#SBATCH --qos normal
#SBATCH -p boost_usr_prod
#SBATCH --mem=64G
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --error=logs/slurm/%j.err
#SBATCH --output=logs/slurm/%j.out

# Install OpenFHE 1.4.2 + FIDESlib into $DEPS.
# FIDESlib lives as a git submodule in third_party/FIDESlib, already patched
# to remove the hardcoded clang++ requirement in its CMakeLists.txt.
#
# Run from: pycudafhe/
# Submit:   sbatch scripts/01_install_deps.sh

set -e
echo "=== Installing OpenFHE + FIDESlib ==="
echo "Host: $(hostname)  |  Date: $(date)"

# ── Proxy ──────────────────────────────────────────────────────────────────
export http_proxy='http://login01:3140'
export https_proxy='http://login01:3140'
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"

export OMP_NUM_THREADS=16

# ── Paths ─────────────────────────────────────────────────────────────────
REPO=/leonardo_work/IscrC_eff-SAM2/azirilli/cuda-cachemir/pycudafhe
DEPS=/leonardo_work/IscrC_eff-SAM2/azirilli/deps
FIDESLIB_SRC="$REPO/third_party/FIDESlib"
FIDESLIB_BUILD="$FIDESLIB_SRC/build"

mkdir -p "$DEPS"

# ── Modules ───────────────────────────────────────────────────────────────
module load cuda/12.6 gcc cmake nvhpc
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export CUDAHOSTCXX=$(which g++)
export CXX=$(which g++)
export CC=$(which gcc)
echo "CUDA_HOME:   $CUDA_HOME"
echo "CUDAHOSTCXX: $CUDAHOSTCXX"
cmake --version | head -1
g++ --version | head -1

# ── GPU architecture ───────────────────────────────────────────────────────
_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1)
if [[ "$_cap" =~ ^[0-9]+\.[0-9]+$ ]]; then
    GPU_ARCH="$(echo "$_cap" | tr -d '.')-real"
    echo "Auto-detected GPU arch: $GPU_ARCH (compute cap ${_cap})"
else
    GPU_ARCH="80-real"
    echo "nvidia-smi unavailable, defaulting to $GPU_ARCH"
fi

# ── Init submodule (FIDESlib already patched in repo) ─────────────────────
echo ""
echo "--- Initialising FIDESlib submodule ---"
cd "$REPO"
GIT_SSL_NO_VERIFY=true git submodule update --init third_party/FIDESlib 2>&1

# Remove any stale git lock files
find "$FIDESLIB_SRC/.git" -name "*.lock" -delete 2>/dev/null || true

echo "FIDESlib source: $FIDESLIB_SRC"
grep -n 'CMAKE_C_COMPILER\|CMAKE_CXX_COMPILER' "$FIDESLIB_SRC/CMakeLists.txt" | head -5 || true

# ── Skip OpenFHE build if already installed ───────────────────────────────
OPENFHE_VER="$DEPS/lib/cmake/OpenFHE/OpenFHEConfigVersion.cmake"
OPENFHE_LIB="$DEPS/lib/libOPENFHEcore_static.a"
INSTALL_OPENFHE=ON
if [ -f "$OPENFHE_VER" ] && grep -qE 'PACKAGE_VERSION.*1\.4\.2' "$OPENFHE_VER" && [ -f "$OPENFHE_LIB" ]; then
    echo "OpenFHE 1.4.2 already installed, skipping."
    INSTALL_OPENFHE=OFF
fi

# ── Configure ─────────────────────────────────────────────────────────────
echo ""
echo "--- Configuring FIDESlib (FIDESLIB_INSTALL_OPENFHE=${INSTALL_OPENFHE}) ---"
rm -rf "$FIDESLIB_BUILD"
cmake -S "$FIDESLIB_SRC" -B "$FIDESLIB_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$DEPS" \
    -DFIDESLIB_INSTALL_PREFIX="$DEPS" \
    -DOPENFHE_INSTALL_PREFIX="$DEPS" \
    -DCMAKE_C_COMPILER="$CC" \
    -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_CUDA_HOST_COMPILER="$CUDAHOSTCXX" \
    -DCUDA_PATH="$CUDA_HOME" \
    -DFIDESLIB_ARCH="$GPU_ARCH" \
    -DFIDESLIB_INSTALL_OPENFHE="$INSTALL_OPENFHE" \
    -DFIDESLIB_BUILD_TESTS=OFF \
    -DFIDESLIB_BUILD_BENCH=OFF \
    -DFIDESLIB_BUILD_EXAMPLES=OFF \
    -DFIDESLIB_COMPILE_PYTHON_BINDINGS=OFF \
    2>&1

# ── Build & Install ────────────────────────────────────────────────────────
echo ""
echo "--- Building FIDESlib (16 parallel jobs) ---"
cmake --build "$FIDESLIB_BUILD" --parallel 16 2>&1

echo ""
echo "--- Installing to $DEPS ---"
cmake --install "$FIDESLIB_BUILD" 2>&1

# ── Verify ────────────────────────────────────────────────────────────────
echo ""
echo "--- Installed files ---"
ls "$DEPS/lib/" | grep -E "fideslib|OPENFHE|openfhe" || echo "WARNING: no lib files found"
ls "$DEPS/include/" | head -10

echo ""
echo "=== Install complete: $DEPS ==="
echo "Date: $(date)"
