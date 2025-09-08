#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <mma.h>
#include "cuda_marcher_core.h"
#include "sdf_eval.h"

namespace omnifield {
namespace cuda {

// CUDA kernel configurations for optimal GPU utilization
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;
constexpr int TILE_SIZE = 16;
constexpr int MAX_SDF_STACK_DEPTH = 16;
constexpr int SHARED_MEM_SDF_CACHE_SIZE = 1024;

// Advanced CUDA sphere tracing with warp-level optimizations
struct WarpRayState {
    float3 origin;
    float3 direction; 
    float t_current;
    float t_max;
    int steps;
    bool active;
};

struct SharedSdfCache {
    float3 positions[SHARED_MEM_SDF_CACHE_SIZE];
    float distances[SHARED_MEM_SDF_CACHE_SIZE];
    int count;
};

// Device constant memory for scene data (faster access)
__constant__ NodeType d_nodeTypes[4096];
__constant__ float d_nodeParams[16384];
__constant__ float4x4 d_transforms[1024];

// Warp-optimized SDF evaluation with divergence minimization
__device__ __forceinline__ float evaluate_sdf_warp_optimized(float3 pos, int warp_id);

// CUTLASS-accelerated matrix operations for transforms
__device__ __forceinline__ float3 apply_transform_cutlass(const float4x4& transform, float3 point);

// High-performance sphere tracing kernel with cooperative groups
__global__ void cuda_sphere_trace_optimized(
    float* __restrict__ image_buffer,
    const int width,
    const int height,
    const float3 camera_pos,
    const float3 camera_dir,
    const float3 camera_up,
    const float fov,
    const float aspect_ratio,
    const int max_steps,
    const float epsilon
);

// Warp-shuffle optimized ray generation
__device__ __forceinline__ float3 generate_ray_direction_warp(
    int pixel_x, int pixel_y, 
    int width, int height,
    const float3& camera_dir,
    const float3& camera_up,
    float fov,
    float aspect_ratio
);

// Coalesced memory access for SDF parameters
__device__ __forceinline__ void load_sdf_params_coalesced(
    int node_index, 
    float* params,
    int thread_id
);

// Advanced SDF primitives with analytical derivatives
__device__ __forceinline__ float sdf_sphere_optimized(float3 p, float radius);
__device__ __forceinline__ float sdf_box_optimized(float3 p, float3 dimensions);
__device__ __forceinline__ float sdf_torus_optimized(float3 p, float major_radius, float minor_radius);
__device__ __forceinline__ float sdf_capsule_optimized(float3 p, float3 a, float3 b, float radius);

// GPU-accelerated CSG operations with warp voting
__device__ __forceinline__ float csg_union_warp(float a, float b, unsigned int mask);
__device__ __forceinline__ float csg_smooth_union_optimized(float a, float b, float k);

// Normal estimation with finite differences and warp cooperation
__device__ __forceinline__ float3 compute_normal_warp_cooperative(
    float3 position, 
    int warp_id,
    float epsilon = 1e-4f
);

// Ambient occlusion and soft shadows
__device__ __forceinline__ float compute_ambient_occlusion(
    float3 position,
    float3 normal,
    int samples = 8
);

__device__ __forceinline__ float compute_soft_shadow(
    float3 position,
    float3 light_dir,
    float min_t = 0.001f,
    float max_t = 10.0f,
    float k = 32.0f
);

// Performance profiling structures
struct GpuProfilerData {
    unsigned long long cycles_per_pixel;
    int divergence_count;
    int cache_hits;
    int cache_misses;
    float occupancy;
};

} // namespace cuda
} // namespace omnifield