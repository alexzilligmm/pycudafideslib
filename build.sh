#!/bin/bash
# build.sh – configure and compile cuda_cachemir (FIDESlib backend)
#
# Usage:
#   bash build.sh [arch] [extra cmake args]
#   bash build.sh 80                         # A100
#   bash build.sh 86                         # RTX 30xx
#   bash build.sh 90                         # H100
#   bash build.sh 80 -DFIDESLIB_ROOT=/opt/fideslib   # pre-installed FIDESlib
#   bash build.sh 80 -DCACHEMIR_BUILD_TESTS=OFF       # skip tests

set -e

ARCH=${1:-80}; shift 2>/dev/null || true
BUILD_DIR="build_sm${ARCH}"

echo "=== cuda-cachemir (FIDESlib backend, sm_${ARCH}) ==="

# Load Leonardo HPC modules if available
if command -v module &>/dev/null; then
    module load cuda/12.1  2>/dev/null || true
    module load cmake/3.25 2>/dev/null || true
    # FIDESlib requires Clang 17+; try to load it
    module load llvm/17    2>/dev/null || module load clang/17 2>/dev/null || true
fi

# FIDESlib requires Clang 17+ as C++ compiler
# If clang-17 is on PATH, use it; otherwise let CMake use whatever is found.
CXX_COMPILER="${CXX:-clang++}"
if command -v clang++-17 &>/dev/null; then
    CXX_COMPILER=clang++-17
elif command -v clang++ &>/dev/null; then
    CLANG_VER=$(clang++ --version 2>&1 | grep -oP '(?<=version )\d+' | head -1)
    if [ "${CLANG_VER:-0}" -lt 17 ] 2>/dev/null; then
        echo "WARNING: clang++ version $CLANG_VER < 17 – FIDESlib may not compile"
    fi
fi

cmake -S . -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
    -DCMAKE_CXX_COMPILER="$CXX_COMPILER" \
    -DCMAKE_CUDA_COMPILER=nvcc \
    "$@"

cmake --build "$BUILD_DIR" --parallel "$(nproc)"

echo ""
echo "Binary : ${BUILD_DIR}/bin/cuda_cachemir"
echo "Tests  : ${BUILD_DIR}/bin/test_ops  test_bootstrap  test_nonlinear  test_llama"
echo ""
echo "Run all tests:"
echo "  ctest --test-dir ${BUILD_DIR} -V"
echo ""
echo "Quick benchmarks:"
echo "  ./${BUILD_DIR}/bin/cuda_cachemir -test Ops     -logN 12"
echo "  ./${BUILD_DIR}/bin/cuda_cachemir -test Decoder -logN 12 -hidDim 256"
echo "  ./${BUILD_DIR}/bin/cuda_cachemir -test Model   -logN 16 -hidDim 4096 -expDim 16384"
