#pragma once
#include <string>
#include <vector>
#include "../../core/marcher/cuda_marcher_core.h"

namespace omnifield {
struct DeviceSceneHost;
}

namespace sdfcad {

struct LoaderConfig {
  bool verbose = false;
};

class UsdDeviceSceneLoader {
public:
  explicit UsdDeviceSceneLoader(const LoaderConfig& cfg) : cfg_(cfg) {}
  bool loadFromFile(const std::string& path, omnifield::DeviceSceneHost& outScene);

private:
  LoaderConfig cfg_;
};

} // namespace sdfcad
