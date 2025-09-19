cmake_minimum_required(VERSION 3.24)
project(OmniField LANGUAGES C CXX CUDA)

# Toolchain / standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
find_package(CUDAToolkit 12.0 REQUIRED)

# Target your RTX 4060 (Ada, SM_89)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

# ---- Single EXE (everything compiled in) ----
add_executable(omnifield_demo
  app/example_demo.cpp
  src/core/marcher/cuda_marcher_impl.cu
  src/core/marcher/sdf_primitives_impl.cu
  src/host/usd_loader/UsdDeviceSceneLoader.cpp
)

target_include_directories(omnifield_demo PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/app
)

# Make multi-.cu builds link correctly
set_target_properties(omnifield_demo PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Clean per-language flags (no leakage into NVCC)
target_compile_options(omnifield_demo PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:/W3 /permissive- /Zc:__cplusplus /MP>
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -lineinfo --allow-unsupported-compiler>
)

# Link only cudart here; device link is handled by the CUDA target + properties above
target_link_libraries(omnifield_demo PRIVATE CUDA::cudart)
