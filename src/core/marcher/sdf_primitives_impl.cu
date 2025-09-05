#include "cuda_marcher_core.h"
#include <cuda_runtime.h>

namespace omnifield {

__device__ inline float length3(float x, float y, float z) {
  return sqrtf(x*x + y*y + z*z);
}

__device__ float sdfSphere(float x, float y, float z, float r) {
  return length3(x,y,z) - r;
}

} // namespace omnifield
