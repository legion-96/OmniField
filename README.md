# OmniField (Bootstrap)

Minimal, **meshless SDF CAD** bootstrap repo: CUDA sphere-tracing core (stubs),
clean C ABI, example app, and room to grow into Omniverse/OpenUSD + Hydra.

> This is a *starter* that compiles fast and lets you push to GitHub at `legion-96/OmniField`.
> Next steps: swap the stub marcher for your real kernels and enable USD/Hydra.

## Build (Linux, CUDA 12.4+, CMake 3.24+)
```bash
git clone https://github.com/legion-96/OmniField.git
cd OmniField
cmake -S . -B build -G Ninja -DWITH_CUDA=ON -DWITH_USD=OFF -DWITH_TESTS=OFF
cmake --build build -j
./build/omnifield_demo
```
If `Ninja` isn't installed, drop `-G Ninja`.

## Layout
```
include/sdfcad/march_core.h           # Stable C ABI (minimal)
src/core/marcher/cuda_marcher_core.h  # Scene structs, enums
src/core/marcher/cuda_marcher_impl.cu # Marcher stub (compiles, no GL needed)
src/core/marcher/sdf_primitives_impl.cu
src/core/marcher/debug_utils.h
src/host/usd_loader/UsdDeviceSceneLoader.h/.cpp  # USD-off fallback loader
app/example_demo.cpp
assets/stages/UsdMiniScene.usda
```

## Toggle USD later
When you have OpenUSD dev packages, configure with `-DWITH_USD=ON`. The code has a
fallback path so it also builds with USD off.
