#include "cuda_marcher_core.h"
#include "debug_utils.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstring>

// Public ABI
#include "../../../include/sdfcad/march_core.h"

using namespace omnifield;

struct OmniMarcher_ {
  OmniMarcherConfig cfg{};
  std::vector<uint8_t> sceneBlob;
};

extern "C" {

OmniMarchResult omni_march_create(OmniMarcherHandle* out, const OmniMarcherConfig* cfg) {
  if (!out || !cfg) return OMNI_MARCH_ERROR_INVALID;
  *out = new OmniMarcher_();
  (*out)->cfg = *cfg;
#ifdef __CUDACC__
  // Warm up CUDA context
  omni_cuda_check(cudaFree(0), "cudaFree(0)");
#endif
  return OMNI_MARCH_SUCCESS;
}

OmniMarchResult omni_march_destroy(OmniMarcherHandle h) {
  if (!h) return OMNI_MARCH_ERROR_INVALID;
  delete h;
  return OMNI_MARCH_SUCCESS;
}

OmniMarchResult omni_march_update_scene(
    OmniMarcherHandle h,
    const uint8_t* paramBlob, uint32_t paramBytes) {
  if (!h) return OMNI_MARCH_ERROR_INVALID;
  h->sceneBlob.assign(paramBlob, paramBlob + paramBytes);
  return OMNI_MARCH_SUCCESS;
}

OmniMarchResult omni_march_render_noop(OmniMarcherHandle h) {
  (void)h;
  // This is intentionally a no-op to keep the bootstrap portable.
  return OMNI_MARCH_SUCCESS;
}

} // extern "C"
