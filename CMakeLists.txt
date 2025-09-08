cmake_minimum_required(VERSION 3.24)
project(OmniField LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(WITH_CUDA "Enable CUDA" ON)

if (WITH_CUDA)
  find_package(CUDAToolkit 12.0 REQUIRED)
  enable_language(CUDA)
endif()

add_library(omnifield_core STATIC
  src/core/marcher/cuda_marcher_core.h
  src/core/marcher/cuda_marcher_impl.cu
  src/core/marcher/cuda_sphere_tracer.cuh
  src/core/marcher/cuda_sphere_tracer.cu
  src/core/marcher/cutlass_transforms.cuh
  src/core/marcher/cutlass_transforms.cu
  src/core/marcher/cufft_sdf_generator.cuh
  src/core/marcher/cufft_sdf_generator.cu
  src/core/marcher/warp_optimizations.cuh
  src/core/marcher/sdf_primitives_impl.cu
  src/core/marcher/sdf_eval.h
  src/host/usd_loader/UsdDeviceSceneLoader.h
  src/host/usd_loader/UsdDeviceSceneLoader.cpp
  app/demo_scenes.h
)

target_include_directories(omnifield_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/app)
if (WITH_CUDA)
  target_link_libraries(omnifield_core PUBLIC CUDA::cudart CUDA::cufft CUDA::curand)
endif()
# Set this to your GPU: 86 (Ampere), 89 (Ada). Avoid 75 with CUDA 13.
set_target_properties(omnifield_core PROPERTIES CUDA_ARCHITECTURES 89)

# Enable CUTLASS if available (requires manual CUTLASS installation)
# target_link_libraries(omnifield_core PUBLIC cutlass)

add_executable(omnifield_demo app/example_demo.cpp)
target_link_libraries(omnifield_demo PRIVATE omnifield_core)
target_include_directories(omnifield_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/app)
