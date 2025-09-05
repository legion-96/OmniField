#include "UsdDeviceSceneLoader.h"
#include <fstream>
#include <iostream>

using namespace omnifield;

namespace sdfcad {

bool UsdDeviceSceneLoader::loadFromFile(const std::string& path, DeviceSceneHost& outScene) {
  // Fallback loader: doesn't parse USD; just reads file to prove I/O.
  std::ifstream f(path);
  if (!f.good()) {
    std::cerr << "[UsdDeviceSceneLoader] File not found: " << path << std::endl;
    return false;
  }
  // Minimal "scene": stash file bytes into paramBlob (placeholder)
  outScene.clear();
  std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  outScene.paramBlob.assign(s.begin(), s.end());
  return true;
}

} // namespace sdfcad
