cmake_minimum_required(VERSION 3.24)
project(OmniField LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(WITH_CUDA "Enable CUDA marcher kernels" ON)
option(WITH_USD "Enable USD/OpenUSD loader support" OFF)   # OFF by default for portability
option(WITH_TESTS "Build unit tests" OFF)
option(BUILD_EXAMPLES "Build example applications" ON)

if (WITH_CUDA)
  find_package(CUDAToolkit 12.0 REQUIRED)
  enable_language(CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89) # RTX 4060 default
  endif()
endif()

# ---------------- Core marcher ----------------
add_library(sdf_marcher_cuda STATIC
  src/core/marcher/cuda_marcher_core.h
  src/core/marcher/cuda_marcher_impl.cu
  src/core/marcher/sdf_primitives_impl.cu
  src/core/marcher/debug_utils.h
)
target_include_directories(sdf_marcher_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
if (WITH_CUDA)
  target_link_libraries(sdf_marcher_cuda PUBLIC CUDA::cudart)
  target_compile_options(sdf_marcher_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --use_fast_math
      -Xptxas -O3
      --expt-relaxed-constexpr
      --extended-lambda
    >
  )
endif()
target_compile_features(sdf_marcher_cuda PUBLIC cxx_std_20)
target_compile_options(sdf_marcher_cuda PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>)

# ---------------- USD loader (stub-friendly) ----------------
add_library(usd_loader OBJECT
  src/host/usd_loader/UsdDeviceSceneLoader.h
  src/host/usd_loader/UsdDeviceSceneLoader.cpp
)
target_link_libraries(usd_loader PUBLIC sdf_marcher_cuda)
target_include_directories(usd_loader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(usd_loader PUBLIC $<$<NOT:$<BOOL:${WITH_USD}>>:OMNIFIELD_USD_FALLBACK=1>)

# ---------------- Example app ----------------
if (BUILD_EXAMPLES)
  add_executable(omnifield_demo app/example_demo.cpp)
  target_link_libraries(omnifield_demo PRIVATE sdf_marcher_cuda usd_loader)
  target_include_directories(omnifield_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
endif()
