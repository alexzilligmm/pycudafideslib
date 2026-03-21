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
# FIDESlib is vendored under third_party/FIDESlib (cloned once, patched to
# remove hardcoded clang++ requirement) so we have full control over its build.
#
# Run from: cuda-backend/
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
REPO=/leonardo_work/IscrC_eff-SAM2/azirilli/cuda-cachemir/cuda-backend
DEPS=/leonardo_work/IscrC_eff-SAM2/azirilli/deps
FIDESLIB_SRC="$REPO/third_party/FIDESlib"
FIDESLIB_BUILD="$FIDESLIB_SRC/build"
PATCH="$REPO/third_party/patches/fideslib_remove_clang_hardcode.patch"

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

# ── Clone FIDESlib into third_party/ (once) ──────────────────────────────
if [ ! -d "$FIDESLIB_SRC/.git" ]; then
    echo ""
    echo "--- Cloning FIDESlib into third_party/FIDESlib ---"
    GIT_SSL_NO_VERIFY=true git clone https://github.com/CAPS-UMU/FIDESlib.git \
        --depth 1 "$FIDESLIB_SRC" 2>&1
else
    echo ""
    echo "--- FIDESlib already cloned at $FIDESLIB_SRC ---"
fi

# Remove any stale git lock files
find "$FIDESLIB_SRC/.git" -name "*.lock" -delete 2>/dev/null || true

# ── Apply patch (idempotent: skip if already applied) ─────────────────────
echo ""
echo "--- Patching FIDESlib (remove hardcoded clang++) ---"
cd "$FIDESLIB_SRC"
if git apply --check "$PATCH" 2>/dev/null; then
    git apply "$PATCH"
    echo "Patch applied."
elif grep -q 'set(CMAKE_C_COMPILER "clang")' CMakeLists.txt; then
    # git apply --check failed but the lines are still there: apply with patch(1)
    patch -p1 < "$PATCH"
    echo "Patch applied via patch(1)."
else
    echo "Patch already applied or not needed, skipping."
fi

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
