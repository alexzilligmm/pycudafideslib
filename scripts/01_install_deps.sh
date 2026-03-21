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

# Two-stage install:
#   1. Build OpenFHE v1.4.2 (with FIDESlib's patch) → $DEPS
#   2. Build FIDESlib against that OpenFHE → $DEPS
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
OPENFHE_PATCH="$FIDESLIB_SRC/deps/openfhe-1.4.2.patch"
OPENFHE_TMP=/tmp/openfhe_build_$$

mkdir -p "$DEPS"

# ── Modules ───────────────────────────────────────────────────────────────
module load cuda/12.6 gcc cmake
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

# ── Init FIDESlib submodule ────────────────────────────────────────────────
echo ""
echo "--- Initialising FIDESlib submodule ---"
cd "$REPO"
GIT_SSL_NO_VERIFY=true git submodule update --init third_party/FIDESlib 2>&1

# ══════════════════════════════════════════════════════════════════════════
# STAGE 1: OpenFHE v1.4.2
# ══════════════════════════════════════════════════════════════════════════
OPENFHE_VER_FILE="$DEPS/lib/cmake/OpenFHE/OpenFHEConfigVersion.cmake"
OPENFHE_LIB="$DEPS/lib/libOPENFHEcore_static.a"

if [ -f "$OPENFHE_VER_FILE" ] && grep -qE 'PACKAGE_VERSION.*1\.4\.2' "$OPENFHE_VER_FILE" && [ -f "$OPENFHE_LIB" ]; then
    echo ""
    echo "--- OpenFHE 1.4.2 already installed, skipping ---"
else
    echo ""
    echo "--- Cloning OpenFHE v1.4.2 ---"
    mkdir -p "$OPENFHE_TMP"
    GIT_SSL_NO_VERIFY=true git clone \
        https://github.com/openfheorg/openfhe-development.git \
        --branch v1.4.2 --depth 1 "$OPENFHE_TMP/src" 2>&1

    echo "--- Applying FIDESlib patch ---"
    cd "$OPENFHE_TMP/src"
    git config user.email "build@leonardo"
    git config user.name "build"
    git am "$OPENFHE_PATCH" 2>&1

    echo "--- Building OpenFHE ---"
    cmake -S "$OPENFHE_TMP/src" -B "$OPENFHE_TMP/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$DEPS" \
        -DCMAKE_C_COMPILER="$CC" \
        -DCMAKE_CXX_COMPILER="$CXX" \
        -DBUILD_STATIC=ON \
        -DWITH_BE2=OFF -DWITH_BE4=OFF \
        2>&1
    cmake --build "$OPENFHE_TMP/build" --parallel 16 2>&1

    echo "--- Installing OpenFHE to $DEPS ---"
    cmake --install "$OPENFHE_TMP/build" 2>&1
    rm -rf "$OPENFHE_TMP"
    echo "OpenFHE installed."
fi

cd "$REPO"   # return after possible cd into deleted OPENFHE_TMP

# ══════════════════════════════════════════════════════════════════════════
# STAGE 2: FIDESlib (against already-installed OpenFHE)
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "--- Configuring FIDESlib (FIDESLIB_INSTALL_OPENFHE=OFF) ---"
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
    -DNVTX3_INCLUDE="$CUDA_HOME/targets/x86_64-linux/include" \
    -DFIDESLIB_INSTALL_OPENFHE=OFF \
    -DFIDESLIB_COMPILE_TESTS=OFF \
    -DFIDESLIB_COMPILE_BENCHMARKS=OFF \
    -DFIDESLIB_COMPILE_PYTHON_BINDINGS=OFF \
    2>&1

echo ""
echo "--- Building FIDESlib (16 parallel jobs) ---"
cmake --build "$FIDESLIB_BUILD" --parallel 16 2>&1

echo ""
echo "--- Installing FIDESlib to $DEPS ---"
cmake --install "$FIDESLIB_BUILD" 2>&1

# ── Verify ────────────────────────────────────────────────────────────────
echo ""
echo "--- Installed files ---"
ls "$DEPS/lib/" | grep -E "fideslib|OPENFHE|openfhe" || echo "WARNING: no lib files found"
ls "$DEPS/include/" | head -10

echo ""
echo "=== Install complete: $DEPS ==="
echo "Date: $(date)"
