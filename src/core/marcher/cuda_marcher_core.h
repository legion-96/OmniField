#pragma once
#include <cstdint>
#include <vector>

// Tiny internal structures to keep the stub clean.
// Real project will replace with full DeviceScene + enums.

namespace omnifield {

struct Float3 { float x,y,z; };
struct AABB   { Float3 min, max; };

struct DeviceSceneHost {
  std::vector<uint8_t> paramBlob;
  uint32_t nodeCount() const { return 1; } // stub: one node
  void clear() { paramBlob.clear(); }
};

} // namespace omnifield
