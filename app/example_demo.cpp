#include "sdfcad/march_core.h"
#include "host/usd_loader/UsdDeviceSceneLoader.h"
#include "core/marcher/cuda_marcher_core.h"
#include <iostream>

using namespace sdfcad;
using namespace omnifield;

int main() {
  // Configure marcher
  OmniMarcherConfig cfg{};
  cfg.baseEpsilon = 1e-4f;
  cfg.epsilonScale = 0.5f;
  cfg.maxSteps = 64;
  cfg.maxPolishSteps = 4;
  cfg.tileSize = 16;
  cfg.tilesX = 16; cfg.tilesY = 16;
  cfg.lipschitzSafety = 0.9f;
  cfg.minStepSize = 1e-5f;
  cfg.maxStepSize = 4.0f;
  cfg.seed = 42;
  cfg.deterministicMode = 1;

  OmniMarcherHandle marcher{};
  if (omni_march_create(&marcher, &cfg) != OMNI_MARCH_SUCCESS) {
    std::cerr << "Failed to create marcher\n";
    return 1;
  }

  // Load a tiny USD stage (fallback path)
  LoaderConfig lcfg{};
  UsdDeviceSceneLoader loader(lcfg);
  DeviceSceneHost hostScene;
  const char* stagePath = "assets/stages/UsdMiniScene.usda";
  if (!loader.loadFromFile(stagePath, hostScene)) {
    std::cerr << "Failed to load: " << stagePath << "\n";
    omni_march_destroy(marcher);
    return 2;
  }
  std::cout << "Loaded scene bytes: " << hostScene.paramBlob.size() << "\n";

  // Upload scene stub
  if (omni_march_update_scene(marcher, hostScene.paramBlob.data(),
                              (uint32_t)hostScene.paramBlob.size()) != OMNI_MARCH_SUCCESS) {
    std::cerr << "Failed to upload scene\n";
    omni_march_destroy(marcher);
    return 3;
  }

  // "Render" (noop)
  if (omni_march_render_noop(marcher) != OMNI_MARCH_SUCCESS) {
    std::cerr << "Render failed\n";
    omni_march_destroy(marcher);
    return 4;
  }

  std::cout << "OmniField bootstrap demo completed successfully.\n";
  omni_march_destroy(marcher);
  return 0;
}
