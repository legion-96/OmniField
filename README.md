# OmniField (Phase‑2 Stubs)

Minimal **working** CMake/CUDA skeleton that matches the Phase‑2 demo API (USD loader, transforms, perf counters).
This compiles on **Windows 11 + VS2022 + CUDA 13** and runs a console demo.

> Real marcher/rendering code is **stubbed** — replace with your actual implementations later.

## Build (Ninja)
```powershell
# from repo root
Remove-Item -Recurse -Force .\build -ErrorAction SilentlyContinue
cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j 16

# Run
.\build\omnifield_demo.exe --usd assets\stages\UsdMiniScene.usda
```
If your GPU is Ampere (e.g., RTX 30xx), use `-DCMAKE_CUDA_ARCHITECTURES=86`.

## Layout
```
app/                      # example app using Phase‑2 interfaces
src/core/marcher/         # marcher API + CUDA stub
src/host/usd_loader/      # USD host-side loader stub
assets/stages/            # sample USD file
```

## Next
1. Tag this as `v0.1-bootstrap`.
2. Implement real marcher kernels in `cuda_marcher_impl.cu` and SDF primitives.
3. Flesh out the USD loader (or switch `WITH_USD=ON` later if you integrate OpenUSD).
