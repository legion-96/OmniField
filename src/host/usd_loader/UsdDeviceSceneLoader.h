#pragma once
#include <string>
#include <iostream>
#include "../../core/marcher/cuda_marcher_core.h"

namespace sdfcad {
struct LoaderConfig{ bool verbose=false, enableTransforms=true, enableBounds=true, enableLipschitzComputation=true; };
struct LoaderStats{ uint32_t nodes{}; void print() const { std::cout<<"LoaderStats: nodes="<<nodes<<"\n"; } };
class UsdDeviceSceneLoader {
public: explicit UsdDeviceSceneLoader(const LoaderConfig& cfg): cfg_(cfg) {}
  bool loadFromFile(const std::string& path, omnifield::DeviceSceneHost& outScene, LoaderStats* stats);
private: LoaderConfig cfg_;
};
}
