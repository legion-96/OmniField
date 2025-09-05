#pragma once
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Minimal public C ABI so other languages/Kit can bind later.

typedef struct { float x,y,z; } OmniVec3;
typedef struct { OmniVec3 min,max; } OmniAABB;

typedef struct {
  OmniVec3 pos, dir, up;
  float fovY;
  float nearZ, farZ;
  float aspect;
} OmniCamera;

typedef struct {
  float baseEpsilon;
  float epsilonScale;
  uint32_t maxSteps;
  uint32_t maxPolishSteps;
  uint32_t tileSize;
  uint32_t tilesX, tilesY;
  float lipschitzSafety;
  float minStepSize;
  float maxStepSize;
  uint32_t seed;
  uint32_t deterministicMode; // bool
} OmniMarcherConfig;

typedef enum {
  OMNI_MARCH_SUCCESS = 0,
  OMNI_MARCH_ERROR_CUDA = -1,
  OMNI_MARCH_ERROR_INVALID = -2
} OmniMarchResult;

typedef struct OmniMarcher_* OmniMarcherHandle;

OmniMarchResult omni_march_create(OmniMarcherHandle* out, const OmniMarcherConfig* cfg);
OmniMarchResult omni_march_destroy(OmniMarcherHandle h);

// Super-minimal SoA upload (just enough for demo). Real project extends this.
OmniMarchResult omni_march_update_scene(
    OmniMarcherHandle h,
    const uint8_t* paramBlob, uint32_t paramBytes
);

// CPU-only placeholder (writes nothing yet) — compiles without GL.
OmniMarchResult omni_march_render_noop(OmniMarcherHandle h);

#ifdef __cplusplus
}
#endif
