#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

namespace omnifield {
namespace cuda {

// Warp-level optimizations for GPU sphere tracing
using namespace cooperative_groups;

// Warp size constant
constexpr int WARP_SIZE = 32;

// Warp divergence minimization techniques
template<typename T>
__device__ __forceinline__ void warp_reduce_min(T& val) {
    thread_group warp = tiled_partition<WARP_SIZE>(this_thread_block());
    
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        T other = warp.shfl_down(val, offset);
        val = min(val, other);
    }
}

template<typename T>
__device__ __forceinline__ void warp_reduce_max(T& val) {
    thread_group warp = tiled_partition<WARP_SIZE>(this_thread_block());
    
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        T other = warp.shfl_down(val, offset);
        val = max(val, other);
    }
}

template<typename T>
__device__ __forceinline__ void warp_reduce_sum(T& val) {
    thread_group warp = tiled_partition<WARP_SIZE>(this_thread_block());
    
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += warp.shfl_down(val, offset);
    }
}

// Warp-level ballot operations for ray coherence
__device__ __forceinline__ unsigned int active_rays_mask() {
    thread_group warp = tiled_partition<WARP_SIZE>(this_thread_block());
    return warp.ballot(true);
}

__device__ __forceinline__ bool any_rays_active(bool ray_active) {
    thread_group warp = tiled_partition<WARP_SIZE>(this_thread_block());
    return warp.any(ray_active);
}

__device__ __forceinline__ bool all_rays_converged(bool ray_converged) {
    thread_group warp = tiled_partition<WARP_SIZE>(this_thread_block());
    return warp.all(ray_converged);
}

// Optimized warp shuffle for data sharing
template<typename T>
__device__ __forceinline__ T warp_broadcast(T value, int src_lane) {
    thread_group warp = tiled_partition<WARP_SIZE>(this_thread_block());
    return warp.shfl(value, src_lane);
}

// Coalesced memory access patterns
template<typename T>
__device__ __forceinline__ void coalesced_store(T* dst, T value, int stride = 1) {
    int lane_id = threadIdx.x % WARP_SIZE;
    dst[lane_id * stride] = value;
}

template<typename T>  
__device__ __forceinline__ T coalesced_load(const T* src, int stride = 1) {
    int lane_id = threadIdx.x % WARP_SIZE;
    return src[lane_id * stride];
}

// SDF evaluation with warp cooperation
struct WarpSdfEvaluator {
    __device__ __forceinline__ float evaluate_primitives_parallel(
        float3 position, 
        int primitive_start_idx,
        int num_primitives
    ) {
        thread_group warp = tiled_partition<WARP_SIZE>(this_thread_block());
        int lane_id = threadIdx.x % WARP_SIZE;
        
        float min_distance = FLT_MAX;
        
        // Each lane evaluates different primitives
        for (int i = lane_id; i < num_primitives; i += WARP_SIZE) {
            int primitive_idx = primitive_start_idx + i;
            float dist = evaluate_single_primitive(position, primitive_idx);
            min_distance = fminf(min_distance, dist);
        }
        
        // Reduce across warp to find minimum distance
        warp_reduce_min(min_distance);
        
        // Broadcast result to all lanes
        return warp.shfl(min_distance, 0);
    }
    
    __device__ __forceinline__ float evaluate_single_primitive(
        float3 position, 
        int primitive_idx
    ) {
        // This would be connected to the actual primitive evaluation
        // For now, return a placeholder
        return length(position) - 1.0f;
    }
};

// Warp-level CSG operations
struct WarpCsgProcessor {
    __device__ __forceinline__ float process_csg_tree(
        float3 position,
        int node_idx,
        int max_depth = 8
    ) {
        thread_group warp = tiled_partition<WARP_SIZE>(this_thread_block());
        
        // Use warp collaboration for tree traversal
        // Different lanes can evaluate different branches simultaneously
        
        // Simplified placeholder implementation
        return 0.0f;
    }
};

// Memory coalescing helpers for transform matrices
__device__ __forceinline__ void load_transform_coalesced(
    const float* transform_matrices,
    float* local_transform,
    int transform_idx
) {
    int lane_id = threadIdx.x % WARP_SIZE;
    
    // Load 16 floats (4x4 matrix) with coalesced access
    if (lane_id < 16) {
        local_transform[lane_id] = transform_matrices[transform_idx * 16 + lane_id];
    }
    
    __syncwarp();
}

// Warp-level normal computation using finite differences
__device__ __forceinline__ float3 compute_normal_warp_parallel(
    float3 position,
    float epsilon,
    WarpSdfEvaluator& evaluator
) {
    thread_group warp = tiled_partition<WARP_SIZE>(this_thread_block());
    int lane_id = threadIdx.x % WARP_SIZE;
    
    float3 offset[6] = {
        {epsilon, 0.0f, 0.0f}, {-epsilon, 0.0f, 0.0f},
        {0.0f, epsilon, 0.0f}, {0.0f, -epsilon, 0.0f},
        {0.0f, 0.0f, epsilon}, {0.0f, 0.0f, -epsilon}
    };
    
    float sample_dist = FLT_MAX;
    if (lane_id < 6) {
        float3 sample_pos = make_float3(
            position.x + offset[lane_id].x,
            position.y + offset[lane_id].y, 
            position.z + offset[lane_id].z
        );
        sample_dist = evaluator.evaluate_primitives_parallel(sample_pos, 0, 1);
    }
    
    // Gather samples using warp shuffle
    float dx_pos = warp.shfl(sample_dist, 0);
    float dx_neg = warp.shfl(sample_dist, 1);
    float dy_pos = warp.shfl(sample_dist, 2);
    float dy_neg = warp.shfl(sample_dist, 3);
    float dz_pos = warp.shfl(sample_dist, 4);
    float dz_neg = warp.shfl(sample_dist, 5);
    
    float3 gradient = make_float3(
        dx_pos - dx_neg,
        dy_pos - dy_neg,
        dz_pos - dz_neg
    );
    
    return normalize(gradient);
}

// Occupancy optimization hints
__device__ __forceinline__ void optimize_warp_occupancy() {
    // Use volatile to prevent compiler optimizations that might hurt occupancy
    volatile int dummy = threadIdx.x;
    
    // Ensure all warps are active before proceeding
    __syncwarp();
}

// Cache-friendly access patterns
template<int CACHE_SIZE>
struct WarpSharedCache {
    __shared__ float cache[CACHE_SIZE];
    __shared__ float3 position_cache[CACHE_SIZE/3];
    
    __device__ __forceinline__ bool lookup(float3 pos, float& result) {
        int lane_id = threadIdx.x % WARP_SIZE;
        int hash = int((pos.x + pos.y + pos.z) * 1000.0f) % CACHE_SIZE;
        
        // Check if this position is cached
        if (lane_id == 0) {
            // Simple cache lookup logic
            result = cache[hash];
        }
        
        bool hit = warp_broadcast(fabsf(result) < FLT_MAX, 0);
        return hit;
    }
    
    __device__ __forceinline__ void store(float3 pos, float value) {
        int lane_id = threadIdx.x % WARP_SIZE;
        int hash = int((pos.x + pos.y + pos.z) * 1000.0f) % CACHE_SIZE;
        
        if (lane_id == 0) {
            cache[hash] = value;
        }
    }
};

// Performance profiling utilities
struct WarpProfiler {
    __device__ __forceinline__ void record_divergence() {
        // Count divergent branches within warp
        thread_group warp = tiled_partition<WARP_SIZE>(this_thread_block());
        // Implementation would accumulate divergence statistics
    }
    
    __device__ __forceinline__ void record_cache_hit() {
        // Track cache hit rates
    }
    
    __device__ __forceinline__ void record_iterations(int count) {
        // Track iteration counts per warp
    }
};

} // namespace cuda
} // namespace omnifield