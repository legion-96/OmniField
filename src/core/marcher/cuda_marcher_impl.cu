#include "cuda_marcher_core.h"
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>   // for std::fill

namespace omnifield {
	struct Marcher_ {
		MarcherConfig cfg{};
		uint32_t nodeCount{};
		std::vector<uint32_t> perf;
	};
} // namespace omnifield

using namespace omnifield;

extern "C" {MarchResult march_create(MarcherHandle_t* out, const MarcherConfig* cfg){
  if(!out||!cfg) return MARCH_ERROR_INVALID;
  *out = new Marcher_(); (*out)->cfg = *cfg; (*out)->perf.assign(PERF_COUNTER_COUNT, 0);
  cudaFree(0); return MARCH_SUCCESS;
}
MarchResult march_destroy(MarcherHandle_t h){ if(!h) return MARCH_ERROR_INVALID; delete h; return MARCH_SUCCESS; }
MarchResult march_update_scene_with_transforms(MarcherHandle_t h,
  const NodeType*, const uint32_t*, const uint32_t*, const uint32_t*,
  const size_t*, const uint32_t*, const float*, const AABB*, const float*, const float*,
  const uint8_t*, const float*, const uint8_t*, size_t, uint32_t nodeCount){
  if(!h) return MARCH_ERROR_INVALID; h->nodeCount = nodeCount; return MARCH_SUCCESS;
}
MarchResult march_render_gl(MarcherHandle_t h, unsigned int, uint32_t, uint32_t, const Camera*, RenderStats* s){
  if(!h) return MARCH_ERROR_INVALID; if(s){ *s = {}; } return MARCH_SUCCESS;
}
MarchResult march_reset_perf_counters(MarcherHandle_t h){ if(!h) return MARCH_ERROR_INVALID; std::fill(h->perf.begin(), h->perf.end(), 0); return MARCH_SUCCESS; }
MarchResult march_get_perf_counters(MarcherHandle_t h, uint32_t* out, uint32_t n){
  if(!h||!out) return MARCH_ERROR_INVALID; for(uint32_t i=0;i<n && i<h->perf.size();++i) out[i]=h->perf[i]; return MARCH_SUCCESS;
}
MarchResult march_pick(MarcherHandle_t h, const Ray*, HitInfo* out){
  if(!h||!out) return MARCH_ERROR_INVALID; out->t=-1.0f; out->steps=1; out->flags=0; return MARCH_SUCCESS;
}
}
