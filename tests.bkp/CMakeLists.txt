cmake_minimum_required(VERSION 3.25)

# ── test helper ──────────────────────────────────────────────────────────
function(add_cachemir_test name)
    add_executable(${name} ${name}.cu)
    target_link_libraries(${name} PRIVATE cachemir_lib GTest::gtest_main)
    target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    set_target_properties(${name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    target_compile_options(${name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -Xcompiler=-fopenmp -lineinfo>
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -fopenmp>
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

enable_testing()

add_cachemir_test(test_ops)
add_cachemir_test(test_bootstrap)
add_cachemir_test(test_primitives)
add_cachemir_test(test_nonlinear)
add_cachemir_test(test_llama)
