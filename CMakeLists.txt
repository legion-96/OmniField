cmake_minimum_required(VERSION 3.24)
project(OmniField LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
find_package(CUDAToolkit 12.0 REQUIRED)

# RTX 4060 (Ada)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Use static CUDA runtime AND static MSVC runtime to match (no LNK4098)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")  # /MT and /MTd

# ---- Single EXE using UNITY .cu (no device-link step) ----
add_executable(omnifield_demo
  app/example_demo.cpp
  app/unity_all.cu
  src/host/usd_loader/UsdDeviceSceneLoader.cpp
)

target_include_directories(omnifield_demo PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/app
)

# UNITY means no nvcc -dlink
set_target_properties(omnifield_demo PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF
)

target_compile_options(omnifield_demo PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:/W3 /permissive- /Zc:__cplusplus /MP>
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -lineinfo --allow-unsupported-compiler>
)

target_link_libraries(omnifield_demo PRIVATE CUDA::cudart_static)

