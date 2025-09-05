#pragma once
#include <cstdio>
#include <cuda_runtime.h>

inline void omni_cuda_check(cudaError_t e, const char* where) {
#ifdef __CUDACC__
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(e));
  }
#else
  (void)e; (void)where;
#endif
}
