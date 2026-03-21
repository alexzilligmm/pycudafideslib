cmake_minimum_required(VERSION 3.25)
project(cuda_cachemir LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# ── GPU architecture (override at cmake invocation time) ───────────────────
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)   # A100 default
endif()

# ── FIDESlib ───────────────────────────────────────────────────────────────
# FIDESlib wraps OpenFHE 1.4.2 with GPU acceleration + bootstrapping.
# If pre-installed: cmake -DFIDESLIB_ROOT=<prefix> ..
# If not found, fall back to FetchContent (requires internet).
find_package(fideslib QUIET CONFIG
    HINTS ${FIDESLIB_ROOT} /usr/local /opt/fideslib)

if(NOT fideslib_FOUND)
    message(STATUS "fideslib not found via find_package – fetching from GitHub")
    include(FetchContent)
    FetchContent_Declare(
        fideslib
        GIT_REPOSITORY https://github.com/CAPS-UMU/FIDESlib.git
        GIT_TAG        main
    )
    set(FIDESLIB_INSTALL_OPENFHE ON  CACHE BOOL "" FORCE)
    set(FIDESLIB_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
    set(FIDESLIB_BUILD_BENCH     OFF CACHE BOOL "" FORCE)
    set(FIDESLIB_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(fideslib)
endif()

find_package(OpenMP REQUIRED)

# ── compile options ────────────────────────────────────────────────────────
set(COMMON_CUDA_FLAGS
    --use_fast_math
    -Xcompiler=-fopenmp
    -lineinfo
)
set(COMMON_CXX_FLAGS -O3 -fopenmp)

# ── main library (object lib shared by exe + tests) ───────────────────────
add_library(cachemir_lib STATIC
    src/bootstrap.cu
    src/linear.cu
    src/nonlinear.cu
    src/llama.cu
)
target_include_directories(cachemir_lib PUBLIC include)
target_link_libraries(cachemir_lib PUBLIC fideslib::fideslib OpenMP::OpenMP_CXX)
target_compile_options(cachemir_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${COMMON_CUDA_FLAGS}>
    $<$<COMPILE_LANGUAGE:CXX>:${COMMON_CXX_FLAGS}>
)
set_target_properties(cachemir_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ── main executable ────────────────────────────────────────────────────────
add_executable(cuda_cachemir src/main.cu)
target_link_libraries(cuda_cachemir PRIVATE cachemir_lib)
target_compile_options(cuda_cachemir PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${COMMON_CUDA_FLAGS}>
)
set_target_properties(cuda_cachemir PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ── tests ──────────────────────────────────────────────────────────────────
option(CACHEMIR_BUILD_TESTS "Build test suite" ON)
if(CACHEMIR_BUILD_TESTS)
    # Use GTest if available, otherwise download it
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.14.0
        )
        set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()
    add_subdirectory(tests)
endif()
