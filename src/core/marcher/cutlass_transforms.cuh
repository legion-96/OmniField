#pragma once

#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/kernel/default_gemm_universal.h>
#include <cutlass/gemm/kernel/gemm_universal.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>

namespace omnifield {
namespace cuda {

// CUTLASS configuration for fast 4x4 matrix operations
using ElementA = float;
using ElementB = float; 
using ElementC = float;
using ElementAccumulator = float;

// Use Tensor Cores when available (Ampere/Ada architecture)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
using ArchTag = cutlass::arch::Sm80;
using OpClass = cutlass::arch::OpClassTensorOp;
#else
using ArchTag = cutlass::arch::Sm75;
using OpClass = cutlass::arch::OpClassSimt;
#endif

// Thread block configuration for optimal occupancy
constexpr int ThreadblockShapeM = 32;
constexpr int ThreadblockShapeN = 32;
constexpr int ThreadblockShapeK = 32;
constexpr int WarpShapeM = 16;
constexpr int WarpShapeN = 16;
constexpr int WarpShapeK = 16;

// CUTLASS GEMM configuration
using CutlassGemm = cutlass::gemm::device::Gemm<
    ElementA,                           // Element type for A matrix
    cutlass::layout::RowMajor,          // Layout type for A matrix
    ElementB,                           // Element type for B matrix
    cutlass::layout::RowMajor,          // Layout type for B matrix  
    ElementC,                           // Element type for C matrix
    cutlass::layout::RowMajor,          // Layout type for C matrix
    ElementAccumulator,                 // Element type for internal accumulation
    OpClass,                           // Operator class tag
    ArchTag,                           // Target architecture
    cutlass::gemm::GemmShape<ThreadblockShapeM, ThreadblockShapeN, ThreadblockShapeK>,
    cutlass::gemm::GemmShape<WarpShapeM, WarpShapeN, WarpShapeK>,
    cutlass::gemm::GemmShape<1, 1, 1>, // Instruction shape
    cutlass::epilogue::thread::LinearCombination<
        ElementC,
        1,
        ElementAccumulator,
        ElementAccumulator
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2  // Stages
>;

// Optimized transform application using CUTLASS
class CutlassTransformEngine {
private:
    CutlassGemm::Arguments args_;
    std::unique_ptr<CutlassGemm> gemm_op_;
    
    // Device memory for batch matrix operations
    ElementA* d_transforms_;
    ElementB* d_positions_;
    ElementC* d_results_;
    
    int max_batch_size_;
    cudaStream_t stream_;
    
public:
    CutlassTransformEngine(int max_batch_size = 1024);
    ~CutlassTransformEngine();
    
    // Initialize CUTLASS engine
    cudaError_t initialize();
    
    // Batch transform multiple points using CUTLASS GEMM
    cudaError_t transform_batch(
        const float* transforms,        // 4x4 matrices [batch_size * 16]
        const float* positions,         // 3D positions [batch_size * 3] 
        float* results,                 // Output positions [batch_size * 3]
        int batch_size,
        cudaStream_t stream = nullptr
    );
    
    // Single transform optimized for small batches
    __device__ __forceinline__ static float3 transform_point_fast(
        const float* transform,  // 4x4 row-major matrix
        const float3& point
    );
    
    // Inverse transform using analytical inverse for rigid transforms
    __device__ __forceinline__ static float3 inverse_transform_point(
        const float* transform,  // 4x4 row-major matrix  
        const float3& point
    );
    
    // Decompose transform for optimized SDF evaluation
    __device__ __forceinline__ static void decompose_transform(
        const float* transform,
        float3* translation,
        float3* rotation,     // Euler angles
        float3* scale
    );
};

// Warp-cooperative matrix multiplication for small matrices
template<int M, int N, int K>
__device__ __forceinline__ void warp_matrix_multiply(
    const float* A,           // M x K matrix
    const float* B,           // K x N matrix  
    float* C,                 // M x N result matrix
    int warp_id
);

// Tensor core acceleration for 4x4 matrix operations (Ampere+)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
__device__ __forceinline__ void tensor_core_4x4_multiply(
    const float* A,
    const float* B, 
    float* C
);
#endif

// Fast 3x3 matrix operations for rotation-only transforms
__device__ __forceinline__ float3 rotate_point_3x3(
    const float* rotation_matrix,    // 3x3 row-major
    const float3& point
);

// Optimized inverse for 3x3 rotation matrices (transpose)
__device__ __forceinline__ void transpose_3x3(
    const float* matrix,
    float* result
);

// Quaternion-based rotation for better numerical stability
struct Quaternion {
    float w, x, y, z;
    
    __device__ __forceinline__ Quaternion(float W = 1.0f, float X = 0.0f, float Y = 0.0f, float Z = 0.0f)
        : w(W), x(X), y(Y), z(Z) {}
        
    __device__ __forceinline__ Quaternion normalize() const;
    __device__ __forceinline__ float3 rotate_point(const float3& point) const;
    __device__ __forceinline__ Quaternion conjugate() const;
};

// Convert 4x4 transform to optimized representation
__device__ __forceinline__ void extract_transform_components(
    const float* transform,
    float3* translation,
    Quaternion* rotation,
    float* uniform_scale
);

// Fast path for identity and uniform scale transforms
__device__ __forceinline__ bool is_identity_transform(const float* transform);
__device__ __forceinline__ bool is_uniform_scale_transform(const float* transform, float* scale);

} // namespace cuda
} // namespace omnifield