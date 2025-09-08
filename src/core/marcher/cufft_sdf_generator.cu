#include "cufft_sdf_generator.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

namespace omnifield {
namespace cuda {

ProceduralSdfGenerator::ProceduralSdfGenerator(int grid_size) 
    : grid_size_(grid_size) {
    memory_size_ = grid_size_ * grid_size_ * grid_size_;
    
    d_frequency_domain_ = nullptr;
    d_spatial_domain_ = nullptr;
    d_noise_buffer_ = nullptr;
    d_sdf_output_ = nullptr;
    d_rand_states_ = nullptr;
}

ProceduralSdfGenerator::~ProceduralSdfGenerator() {
    if (fft_plan_3d_) cufftDestroy(fft_plan_3d_);
    if (ifft_plan_3d_) cufftDestroy(ifft_plan_3d_);
    
    if (d_frequency_domain_) cudaFree(d_frequency_domain_);
    if (d_spatial_domain_) cudaFree(d_spatial_domain_);
    if (d_noise_buffer_) cudaFree(d_noise_buffer_);
    if (d_sdf_output_) cudaFree(d_sdf_output_);
    if (d_rand_states_) cudaFree(d_rand_states_);
}

cudaError_t ProceduralSdfGenerator::initialize() {
    // Create 3D FFT plans
    cufftResult fft_result;
    
    fft_result = cufftPlan3d(&fft_plan_3d_, grid_size_, grid_size_, grid_size_, CUFFT_R2C);
    if (fft_result != CUFFT_SUCCESS) {
        return cudaErrorInvalidValue;
    }
    
    fft_result = cufftPlan3d(&ifft_plan_3d_, grid_size_, grid_size_, grid_size_, CUFFT_C2R);
    if (fft_result != CUFFT_SUCCESS) {
        return cudaErrorInvalidValue;
    }
    
    // Allocate device memory
    size_t complex_size = memory_size_ * sizeof(cuComplex);
    size_t float_size = memory_size_ * sizeof(float);
    size_t rand_size = memory_size_ * sizeof(curandState);
    
    cudaError_t result = cudaMalloc(&d_frequency_domain_, complex_size);
    if (result != cudaSuccess) return result;
    
    result = cudaMalloc(&d_spatial_domain_, complex_size);
    if (result != cudaSuccess) return result;
    
    result = cudaMalloc(&d_noise_buffer_, float_size);
    if (result != cudaSuccess) return result;
    
    result = cudaMalloc(&d_sdf_output_, float_size);
    if (result != cudaSuccess) return result;
    
    result = cudaMalloc(&d_rand_states_, rand_size);
    if (result != cudaSuccess) return result;
    
    // Initialize random states
    dim3 block(256);
    dim3 grid((memory_size_ + block.x - 1) / block.x);
    
    init_curand_states_kernel<<<grid, block>>>(d_rand_states_, memory_size_, 12345);
    result = cudaGetLastError();
    if (result != cudaSuccess) return result;
    
    result = cudaDeviceSynchronize();
    return result;
}

cudaError_t ProceduralSdfGenerator::generate_fractal_sdf(
    float* output_sdf,
    float frequency_scale,
    int octaves,
    float lacunarity,
    float persistence,
    float amplitude,
    unsigned int seed,
    cudaStream_t stream
) {
    // Clear frequency domain buffer
    cudaError_t result = cudaMemsetAsync(d_frequency_domain_, 0, 
        memory_size_ * sizeof(cuComplex), stream);
    if (result != cudaSuccess) return result;
    
    // Generate fractal noise in frequency domain
    dim3 block(8, 8, 8);
    dim3 grid(
        (grid_size_ + block.x - 1) / block.x,
        (grid_size_ + block.y - 1) / block.y,
        (grid_size_ + block.z - 1) / block.z
    );
    
    float current_frequency = frequency_scale;
    float current_amplitude = amplitude;
    
    for (int octave = 0; octave < octaves; ++octave) {
        // Generate noise spectrum for this octave
        generate_noise_spectrum_kernel<<<grid, block, 0, stream>>>(
            d_frequency_domain_,
            grid_size_,
            current_frequency,
            current_amplitude,
            d_rand_states_,
            seed + octave
        );
        
        // Apply fractal combination
        apply_fractal_combination_kernel<<<grid, block, 0, stream>>>(
            d_frequency_domain_,
            grid_size_,
            octave,
            lacunarity,
            persistence,
            amplitude
        );
        
        current_frequency *= lacunarity;
        current_amplitude *= persistence;
    }
    
    // Convert from frequency domain to spatial domain
    cufftResult fft_result = cufftExecC2R(ifft_plan_3d_, 
        d_frequency_domain_, d_sdf_output_);
    if (fft_result != CUFFT_SUCCESS) {
        return cudaErrorLaunchFailure;
    }
    
    // Copy result to output
    result = cudaMemcpyAsync(output_sdf, d_sdf_output_, 
        memory_size_ * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    
    return result;
}

cudaError_t ProceduralSdfGenerator::generate_heightmap_sdf(
    const float* heightmap,
    float* output_sdf,
    float height_scale,
    float smoothing_radius,
    cudaStream_t stream
) {
    // Convert heightmap to frequency domain
    dim3 block(16, 16);
    dim3 grid(
        (grid_size_ + block.x - 1) / block.x,
        (grid_size_ + block.y - 1) / block.y
    );
    
    heightmap_to_frequency_kernel<<<grid, block, 0, stream>>>(
        heightmap,
        d_frequency_domain_, 
        grid_size_,
        height_scale
    );
    
    // Apply smoothing filter
    dim3 block3d(8, 8, 8);
    dim3 grid3d(
        (grid_size_ + block3d.x - 1) / block3d.x,
        (grid_size_ + block3d.y - 1) / block3d.y,
        (grid_size_ + block3d.z - 1) / block3d.z
    );
    
    apply_smoothing_filter_kernel<<<grid3d, block3d, 0, stream>>>(
        d_frequency_domain_,
        grid_size_,
        smoothing_radius
    );
    
    // Convert back to spatial domain
    cufftResult fft_result = cufftExecC2R(ifft_plan_3d_, 
        d_frequency_domain_, d_sdf_output_);
    if (fft_result != CUFFT_SUCCESS) {
        return cudaErrorLaunchFailure;
    }
    
    // Copy result to output
    cudaError_t result = cudaMemcpyAsync(output_sdf, d_sdf_output_,
        memory_size_ * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    
    return result;
}

cudaError_t ProceduralSdfGenerator::generate_turbulence_field(
    float* turbulence_field,
    float base_frequency,
    int octaves,
    float lacunarity,
    float gain,
    unsigned int seed,
    cudaStream_t stream
) {
    // Initialize frequency domain buffer
    cudaError_t result = cudaMemsetAsync(d_frequency_domain_, 0,
        memory_size_ * sizeof(cuComplex), stream);
    if (result != cudaSuccess) return result;
    
    dim3 block(8, 8, 8);
    dim3 grid(
        (grid_size_ + block.x - 1) / block.x,
        (grid_size_ + block.y - 1) / block.y,
        (grid_size_ + block.z - 1) / block.z
    );
    
    float current_frequency = base_frequency;
    float current_amplitude = 1.0f;
    
    for (int octave = 0; octave < octaves; ++octave) {
        // Generate turbulence noise for this octave
        turbulence_noise_kernel<<<grid, block, 0, stream>>>(
            d_frequency_domain_,
            grid_size_,
            current_frequency,
            current_amplitude,
            d_rand_states_,
            octave
        );
        
        current_frequency *= lacunarity;
        current_amplitude *= gain;
    }
    
    // Transform to spatial domain
    cufftResult fft_result = cufftExecC2R(ifft_plan_3d_,
        d_frequency_domain_, d_sdf_output_);
    if (fft_result != CUFFT_SUCCESS) {
        return cudaErrorLaunchFailure;
    }
    
    // Copy result
    result = cudaMemcpyAsync(turbulence_field, d_sdf_output_,
        memory_size_ * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    
    return result;
}

// CUDA kernel implementations
__global__ void generate_noise_spectrum_kernel(
    cuComplex* frequency_data,
    int grid_size,
    float frequency_scale,
    float amplitude,
    curandState* rand_states,
    unsigned int seed
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= grid_size || y >= grid_size || z >= grid_size) return;
    
    int idx = IDX3D(x, y, z, grid_size);
    curandState* local_state = &rand_states[idx];
    
    // Compute frequency coordinates
    float fx = (x < grid_size/2) ? x : x - grid_size;
    float fy = (y < grid_size/2) ? y : y - grid_size;  
    float fz = (z < grid_size/2) ? z : z - grid_size;
    
    // Normalize frequencies
    fx /= grid_size;
    fy /= grid_size;
    fz /= grid_size;
    
    float freq_mag = sqrtf(fx*fx + fy*fy + fz*fz) * frequency_scale;
    
    // Generate noise with 1/f^β spectrum (pink noise)
    float beta = 1.0f; // Pink noise exponent
    float power = (freq_mag > 1e-6f) ? powf(freq_mag, -beta * 0.5f) : 0.0f;
    
    float phase = curand_uniform(local_state) * 2.0f * M_PI;
    float magnitude = power * amplitude;
    
    cuComplex noise_value = create_complex_noise(magnitude, phase, local_state);
    
    // Accumulate into frequency domain
    frequency_data[idx].x += noise_value.x;
    frequency_data[idx].y += noise_value.y;
}

__global__ void apply_fractal_combination_kernel(
    cuComplex* frequency_data,
    int grid_size,
    int octave,
    float lacunarity,
    float persistence,
    float base_amplitude
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= grid_size || y >= grid_size || z >= grid_size) return;
    
    int idx = IDX3D(x, y, z, grid_size);
    
    // Apply octave-specific scaling
    float octave_scale = powf(persistence, octave);
    frequency_data[idx].x *= octave_scale;
    frequency_data[idx].y *= octave_scale;
}

__global__ void heightmap_to_frequency_kernel(
    const float* heightmap,
    cuComplex* frequency_data,
    int grid_size,
    float height_scale
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= grid_size || y >= grid_size) return;
    
    // Extrude heightmap into 3D volume
    float height_value = heightmap[y * grid_size + x] * height_scale;
    
    for (int z = 0; z < grid_size; ++z) {
        int idx = IDX3D(x, y, z, grid_size);
        
        // Create SDF from heightmap (distance to surface)
        float world_z = (float(z) / grid_size - 0.5f) * 2.0f; // [-1, 1]
        float distance = world_z - height_value;
        
        // Store as real value in frequency domain (DC component)
        frequency_data[idx].x = distance;
        frequency_data[idx].y = 0.0f;
    }
}

__global__ void init_curand_states_kernel(
    curandState* states,
    int size,
    unsigned int seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    curand_init(seed, idx, 0, &states[idx]);
}

__device__ __forceinline__ cuComplex create_complex_noise(
    float magnitude,
    float phase,
    curandState* local_state
) {
    cuComplex result;
    result.x = magnitude * cosf(phase);
    result.y = magnitude * sinf(phase);
    return result;
}

__device__ __forceinline__ float compute_frequency_magnitude(
    int i, int j, int k,
    int grid_size
) {
    float fi = (i < grid_size/2) ? i : i - grid_size;
    float fj = (j < grid_size/2) ? j : j - grid_size;
    float fk = (k < grid_size/2) ? k : k - grid_size;
    
    fi /= grid_size;
    fj /= grid_size;  
    fk /= grid_size;
    
    return sqrtf(fi*fi + fj*fj + fk*fk);
}

__global__ void turbulence_noise_kernel(
    cuComplex* frequency_data,
    int grid_size,
    float frequency,
    float amplitude,
    curandState* rand_states,
    int octave
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= grid_size || y >= grid_size || z >= grid_size) return;
    
    int idx = IDX3D(x, y, z, grid_size);
    curandState* local_state = &rand_states[idx];
    
    float freq_mag = compute_frequency_magnitude(x, y, z, grid_size);
    
    // Apply turbulence-specific frequency response
    float target_freq = frequency;
    float freq_response = expf(-powf((freq_mag - target_freq) / (target_freq * 0.5f), 2.0f));
    
    float phase = curand_uniform(local_state) * 2.0f * M_PI;
    float magnitude = amplitude * freq_response;
    
    cuComplex turbulence = create_complex_noise(magnitude, phase, local_state);
    
    // Add to existing frequency data
    frequency_data[idx].x += turbulence.x;
    frequency_data[idx].y += turbulence.y;
}

__global__ void apply_smoothing_filter_kernel(
    cuComplex* frequency_data,
    int grid_size,
    float smoothing_radius
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= grid_size || y >= grid_size || z >= grid_size) return;
    
    int idx = IDX3D(x, y, z, grid_size);
    
    float freq_mag = compute_frequency_magnitude(x, y, z, grid_size);
    float cutoff = 1.0f / smoothing_radius;
    
    // Apply Gaussian lowpass filter
    float filter_response = expf(-0.5f * powf(freq_mag / cutoff, 2.0f));
    
    frequency_data[idx].x *= filter_response;
    frequency_data[idx].y *= filter_response;
}

__global__ void spectral_lowpass_filter_kernel(
    cuComplex* frequency_data,
    int grid_size,
    float cutoff_frequency,
    float sharpness
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= grid_size || y >= grid_size || z >= grid_size) return;
    
    int idx = IDX3D(x, y, z, grid_size);
    
    float freq_mag = compute_frequency_magnitude(x, y, z, grid_size);
    
    // Butterworth filter response
    float filter_response = 1.0f / (1.0f + powf(freq_mag / cutoff_frequency, 2.0f * sharpness));
    
    frequency_data[idx].x *= filter_response;
    frequency_data[idx].y *= filter_response;
}

} // namespace cuda
} // namespace omnifield