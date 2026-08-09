cmake_minimum_required(VERSION 3.24)

# The CUDA implementation is deliberately an independent project.  The base
# Python wheel and the Rust CPU workspace must not need a CUDA Toolkit merely
# to configure or build.
project(renewable_huber_cuda LANGUAGES CXX CUDA)

option(RH_CUDA_BUILD_SHARED "Build the renewable-huber CUDA C ABI as a shared library" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)

# Listed explicitly rather than globbed: rh-cuda-ffi/build.rs relies on CMake
# reconfiguring deterministically when this list changes.
set(RH_CUDA_SOURCES
    src/c_api.cu
    src/engine_internal.cu
    src/workspace.cu
    src/batch.cu
    src/objective.cu
    src/linear_solver.cu
    src/pipeline.cu
    src/huber_kernels.cu
    # Defines no symbols; holds the static_asserts that pin the public struct
    # layout to native/contracts/rh_cuda_contract.json.
    src/abi_contract.cpp
)

if(RH_CUDA_BUILD_SHARED)
    add_library(rh_cuda SHARED ${RH_CUDA_SOURCES})
else()
    add_library(rh_cuda STATIC ${RH_CUDA_SOURCES})
endif()

add_library(renewable_huber::cuda ALIAS rh_cuda)

target_include_directories(rh_cuda
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(rh_cuda
    PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::cusolver
)

target_compile_definitions(rh_cuda PRIVATE RH_CUDA_BUILD=1)
if(NOT RH_CUDA_BUILD_SHARED)
    target_compile_definitions(rh_cuda PUBLIC RH_CUDA_STATIC=1)
endif()

# Whole-program device compilation is a release requirement, not a preference.
# Separable compilation routes every architecture through nvlink, and nvlink
# emits SASS only: the device-linked image that the runtime registers has no
# PTX, so `CMAKE_CUDA_ARCHITECTURES` entries without a `-real` suffix silently
# lose their virtual half.  The individual translation units still carry PTX,
# which makes the loss invisible in object-level inspection.  Asking the device
# linker for it directly does not help either -- `nvcc -dlink` with
# `code=compute_120` succeeds and produces an object holding neither SASS nor
# PTX.  No translation unit here calls a `__device__` function defined in
# another one, so whole-program compilation costs nothing and keeps the
# forward-compatible PTX the release fat-binary policy promises.
set_target_properties(rh_cuda PROPERTIES
    OUTPUT_NAME renewable_huber_cuda
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION OFF
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

# The implementation uses the CUDA libraries for dense linear algebra and
# intentionally avoids implicit reduced-precision math.  Consumers decide the
# supported SM set when producing distribution wheels through CMAKE_CUDA_ARCHITECTURES.

option(RH_CUDA_BUILD_TESTS "Build native CUDA ABI smoke tests" OFF)
if(RH_CUDA_BUILD_TESTS)
    include(CTest)
    add_executable(rh_cuda_smoke tests/rh_cuda_smoke.cpp)
    target_link_libraries(rh_cuda_smoke PRIVATE rh_cuda)
    add_test(NAME rh_cuda_smoke COMMAND rh_cuda_smoke)
endif()

include(GNUInstallDirs)
install(TARGETS rh_cuda
    EXPORT renewable_huber_cuda_targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(FILES include/rh_cuda.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
