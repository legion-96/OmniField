#pragma once

#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include "cuda_marcher_core.h"

namespace omnifield {
namespace cuda {

// cuFFT-based procedural SDF generation for complex geometries
class ProceduralSdfGenerator {
private:
    cufftHandle fft_plan_3d_;
    cufftHandle ifft_plan_3d_;
    
    // Device memory for FFT operations
    cuComplex* d_frequency_domain_;
    cuComplex* d_spatial_domain_;
    float* d_noise_buffer_;
    float* d_sdf_output_;
    
    // Grid dimensions for 3D FFT
    int grid_size_;
    size_t memory_size_;
    
    // Noise generation parameters
    curandState* d_rand_states_;
    
public:
    ProceduralSdfGenerator(int grid_size = 128);
    ~ProceduralSdfGenerator();
    
    // Initialize cuFFT plans and allocate memory
    cudaError_t initialize();
    
    // Generate procedural SDF using frequency domain techniques
    cudaError_t generate_fractal_sdf(
        float* output_sdf,           // Output SDF values [grid_size^3]
        float frequency_scale = 1.0f, // Base frequency scale
        int octaves = 6,             // Number of fractal octaves
        float lacunarity = 2.0f,     // Frequency multiplier per octave
        float persistence = 0.5f,    // Amplitude multiplier per octave
        float amplitude = 1.0f,      // Base amplitude
        unsigned int seed = 42,      // Random seed
        cudaStream_t stream = nullptr
    );
    
    // Generate SDF from heightmap using FFT-based processing
    cudaError_t generate_heightmap_sdf(
        const float* heightmap,      // Input heightmap [grid_size x grid_size]
        float* output_sdf,           // Output SDF [grid_size^3]
        float height_scale = 1.0f,   // Height scaling factor
        float smoothing_radius = 2.0f, // Smoothing kernel radius
        cudaStream_t stream = nullptr
    );
    
    // Generate turbulence field for volumetric effects
    cudaError_t generate_turbulence_field(
        float* turbulence_field,     // Output turbulence [grid_size^3]
        float base_frequency = 1.0f,
        int octaves = 4,
        float lacunarity = 2.0f,
        float gain = 0.5f,
        unsigned int seed = 123,
        cudaStream_t stream = nullptr
    );
    
    // Apply frequency domain filtering for smooth SDF generation
    cudaError_t apply_spectral_filter(
        float* input_sdf,            // Input SDF values
        float* output_sdf,           // Filtered SDF output
        float cutoff_frequency = 0.5f, // Normalized cutoff frequency
        float filter_sharpness = 2.0f, // Filter rolloff sharpness
        cudaStream_t stream = nullptr
    );
    
    // Generate cellular/Voronoi patterns using FFT acceleration
    cudaError_t generate_cellular_sdf(
        float* output_sdf,           // Output SDF
        int num_cells = 64,          // Number of cell centers
        float jitter = 0.8f,         // Cell center randomization
        float cell_size = 0.1f,      // Average cell size
        unsigned int seed = 456,
        cudaStream_t stream = nullptr
    );
};

// CUDA kernels for FFT-based SDF generation
__global__ void generate_noise_spectrum_kernel(
    cuComplex* frequency_data,
    int grid_size,
    float frequency_scale,
    float amplitude,
    curandState* rand_states,
    unsigned int seed
);

__global__ void apply_fractal_combination_kernel(
    cuComplex* frequency_data,
    int grid_size,
    int octave,
    float lacunarity,
    float persistence,
    float base_amplitude
);

__global__ void heightmap_to_frequency_kernel(
    const float* heightmap,
    cuComplex* frequency_data,
    int grid_size,
    float height_scale
);

__global__ void apply_smoothing_filter_kernel(
    cuComplex* frequency_data,
    int grid_size,
    float smoothing_radius
);

__global__ void turbulence_noise_kernel(
    cuComplex* frequency_data,
    int grid_size,
    float frequency,
    float amplitude,
    curandState* rand_states,
    int octave
);

__global__ void spectral_lowpass_filter_kernel(
    cuComplex* frequency_data,
    int grid_size,
    float cutoff_frequency,
    float sharpness
);

__global__ void generate_voronoi_seeds_kernel(
    float3* cell_centers,
    int num_cells,
    float jitter,
    curandState* rand_states,
    unsigned int seed
);

__global__ void compute_voronoi_sdf_kernel(
    const float3* cell_centers,
    int num_cells,
    float* output_sdf,
    int grid_size,
    float cell_size
);

// Utility kernels for complex number operations
__global__ void complex_multiply_kernel(
    cuComplex* a,
    const cuComplex* b,
    int size
);

__global__ void complex_add_kernel(
    cuComplex* a,
    const cuComplex* b,
    int size,
    float scale = 1.0f
);

__global__ void real_to_complex_kernel(
    const float* real_data,
    cuComplex* complex_data,
    int size
);

__global__ void complex_to_real_kernel(
    const cuComplex* complex_data,
    float* real_data,
    int size
);

// Random state initialization for procedural generation
__global__ void init_curand_states_kernel(
    curandState* states,
    int size,
    unsigned int seed
);

// Advanced noise functions for procedural generation
__device__ __forceinline__ float noise3d_gradient(
    float x, float y, float z,
    curandState* local_state
);

__device__ __forceinline__ float noise3d_perlin(
    float x, float y, float z,
    int octaves,
    float lacunarity,
    float persistence,
    curandState* local_state
);

__device__ __forceinline__ float noise3d_simplex(
    float x, float y, float z,
    curandState* local_state
);

// Frequency domain utility functions
__device__ __forceinline__ float compute_frequency_magnitude(
    int i, int j, int k,
    int grid_size
);

__device__ __forceinline__ cuComplex create_complex_noise(
    float magnitude,
    float phase,
    curandState* local_state
);

__device__ __forceinline__ void apply_bandpass_filter(
    cuComplex* value,
    float frequency,
    float low_cutoff,
    float high_cutoff,
    float sharpness
);

// Optimized 3D indexing macros
#define IDX3D(x, y, z, size) ((z) * (size) * (size) + (y) * (size) + (x))
#define COORD3D(idx, size, x, y, z) \
    do { \
        (z) = (idx) / ((size) * (size)); \
        (y) = ((idx) % ((size) * (size))) / (size); \
        (x) = (idx) % (size); \
    } while(0)

} // namespace cuda  
} // namespace omnifield