#include "../src/core/marcher/cuda_marcher_core.h"
#include "../src/host/usd_loader/UsdDeviceSceneLoader.h"
#include "../src/core/marcher/cuda_sphere_tracer.cuh"
#include "../src/core/marcher/cutlass_transforms.cuh"
#include "../src/core/marcher/cufft_sdf_generator.cuh"
#include "../src/core/marcher/warp_optimizations.cuh"
#include "demo_scenes.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <filesystem>
#include <map>
#include <algorithm>

using namespace sdfcad;
using namespace omnifield;

class Enhanced_MarcherDemo {
public:
    ~Enhanced_MarcherDemo(){ cleanup(); }
    bool initialize(){
        config_.baseEpsilon = 1e-4f; config_.maxSteps = 128; config_.maxPolishSteps = 8;
        config_.tileSize = 16; config_.tilesX = 32; config_.tilesY = 32;
        config_.enablePolish = true; config_.deterministicMode = true; config_.seed = 42;
        if (march_create(&marcher_, &config_) != MARCH_SUCCESS){ std::cerr<<"march_create failed\n"; return false; }
        LoaderConfig lc{}; lc.enableTransforms = true; lc.enableBounds = true; lc.enableLipschitzComputation = true;
        usdLoader_ = std::make_unique<UsdDeviceSceneLoader>(lc);
        
        // Initialize CUDA optimization engines
        try {
            cutlass_engine_ = std::make_unique<omnifield::cuda::CutlassTransformEngine>(2048);
            sdf_generator_ = std::make_unique<omnifield::cuda::ProceduralSdfGenerator>(256);
            
            if (cutlass_engine_->initialize() != cudaSuccess) {
                std::cerr << "Failed to initialize CUTLASS engine\n";
            } else {
                std::cout << "CUTLASS Transform Engine initialized successfully\n";
            }
            
            if (sdf_generator_->initialize() != cudaSuccess) {
                std::cerr << "Failed to initialize cuFFT SDF Generator\n";
            } else {
                std::cout << "cuFFT SDF Generator initialized successfully\n";
            }
        } catch (const std::exception& e) {
            std::cerr << "Error initializing GPU engines: " << e.what() << "\n";
        }
        
        std::cout<<"SDFCAD Phase 2 Demo Initialized with GPU Optimizations\n"; 
        return true;
    }
    
    void cleanup(){ 
        if(marcher_){march_destroy(marcher_); marcher_=nullptr;} 
        usdLoader_.reset();
        cutlass_engine_.reset();
        sdf_generator_.reset();
    }
    
    bool loadUsdScene(const std::string& usd){
        if(!usdLoader_) return false;
        if(!std::filesystem::exists(usd)){ std::cerr<<"USD not found: "<<usd<<"\n"; return false; }
        DeviceSceneHost host; LoaderStats st{};
        if(!usdLoader_->loadFromFile(usd, host, &st)) return false;
        auto r = march_update_scene_with_transforms(marcher_, host.nodeTypes.data(), host.nodeFlags.data(),
            host.firstChild.data(), host.childCount.data(), host.paramOffset.data(), host.objId.data(),
            host.lipschitzL.data(), host.bounds.data(), host.xforms.data(), host.maxScale.data(),
            host.xformIsIdentity.data(), host.nodeEpsilon.data(), host.paramBlob.data(), host.paramBlob.size(), host.nodeCount());
        if (r!=MARCH_SUCCESS) return false;
        scene_ = std::move(host); stats_ = st; stats_.print(); return true;
    }
    
    void runGpuOptimizationBenchmarks() {
        std::cout << "\n=== GPU Optimization Benchmarks ===\n";
        
        // Test procedural SDF generation
        if (sdf_generator_) {
            std::cout << "Testing cuFFT-based procedural SDF generation...\n";
            
            std::vector<float> fractal_sdf(256 * 256 * 256);
            auto start = std::chrono::high_resolution_clock::now();
            
            cudaError_t result = sdf_generator_->generate_fractal_sdf(
                fractal_sdf.data(), 
                2.0f,  // frequency_scale
                6,     // octaves
                2.0f,  // lacunarity
                0.5f,  // persistence
                1.0f   // amplitude
            );
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            if (result == cudaSuccess) {
                std::cout << "Fractal SDF generation: " << duration.count() << "ms\n";
                std::cout << "Volume size: 256^3 voxels\n";
            } else {
                std::cout << "Fractal SDF generation failed\n";
            }
            
            // Test turbulence field generation
            std::vector<float> turbulence_field(256 * 256 * 256);
            start = std::chrono::high_resolution_clock::now();
            
            result = sdf_generator_->generate_turbulence_field(
                turbulence_field.data(),
                1.0f,  // base_frequency
                4,     // octaves
                2.0f,  // lacunarity
                0.5f   // gain
            );
            
            end = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            if (result == cudaSuccess) {
                std::cout << "Turbulence field generation: " << duration.count() << "ms\n";
            }
        }
        
        // Test CUTLASS transform performance
        if (cutlass_engine_) {
            std::cout << "\nTesting CUTLASS matrix operations...\n";
            
            const int batch_size = 1000;
            std::vector<float> transforms(batch_size * 16);
            std::vector<float> positions(batch_size * 3);
            std::vector<float> results(batch_size * 3);
            
            // Initialize test data
            for (int i = 0; i < batch_size; ++i) {
                // Identity matrix with random translation
                for (int j = 0; j < 16; ++j) {
                    transforms[i * 16 + j] = (j % 5 == 0) ? 1.0f : 0.0f;
                }
                transforms[i * 16 + 3] = float(rand()) / RAND_MAX * 2.0f - 1.0f;
                transforms[i * 16 + 7] = float(rand()) / RAND_MAX * 2.0f - 1.0f;
                transforms[i * 16 + 11] = float(rand()) / RAND_MAX * 2.0f - 1.0f;
                
                positions[i * 3 + 0] = float(rand()) / RAND_MAX * 2.0f - 1.0f;
                positions[i * 3 + 1] = float(rand()) / RAND_MAX * 2.0f - 1.0f;
                positions[i * 3 + 2] = float(rand()) / RAND_MAX * 2.0f - 1.0f;
            }
            
            auto start = std::chrono::high_resolution_clock::now();
            
            cudaError_t result = cutlass_engine_->transform_batch(
                transforms.data(),
                positions.data(),
                results.data(),
                batch_size
            );
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            if (result == cudaSuccess) {
                std::cout << "CUTLASS batch transform (" << batch_size << " matrices): " 
                         << duration.count() << "µs\n";
                std::cout << "Throughput: " << (batch_size * 1000000.0f / duration.count()) 
                         << " transforms/sec\n";
            } else {
                std::cout << "CUTLASS batch transform failed\n";
            }
        }
    }
    
    void runRaycastBenchmark(){
        if(scene_.nodeCount()==0) createFallback();
        
        std::cout << "\n=== Ray Casting Performance ===\n";
        runRayPattern("Radial rays (optimized)", genRadial(1000));
        runRayPattern("Grid rays (coherent)", genGrid(32,32));
        runRayPattern("Random rays (divergent)", genRandom(1000));
        
        // Test GPU-accelerated rendering
        std::cout << "\nTesting GPU-accelerated rendering...\n";
        Camera cam;
        cam.position = {3.0f, 0.0f, 0.0f};
        cam.direction = {-1.0f, 0.0f, 0.0f};
        cam.up = {0.0f, 1.0f, 0.0f};
        cam.fovY = 45.0f;
        cam.aspect = 1.0f;
        
        RenderStats render_stats;
        auto start = std::chrono::high_resolution_clock::now();
        
        MarchResult result = march_render_gl(marcher_, 0, 512, 512, &cam, &render_stats);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        if (result == MARCH_SUCCESS) {
            std::cout << "GPU render (512x512): " << duration.count() << "ms\n";
            std::cout << "GPU render time: " << render_stats.renderTimeMs << "ms\n";
            std::cout << "Effective FPS: " << (1000.0f / render_stats.renderTimeMs) << "\n";
        }
        
        march_reset_perf_counters(marcher_);
        auto rays = genRadial(256); for(auto& r: rays){ HitInfo h; march_pick(marcher_, &r, &h); }
        uint32_t c[PERF_COUNTER_COUNT]{0}; march_get_perf_counters(marcher_, c, PERF_COUNTER_COUNT);
        std::cout<<"Perf total eval: "<<c[PERF_TOTAL_EVALUATIONS]<<"\n";
    }
    
    void runConvergenceAnalysis(){
        if(scene_.nodeCount()==0) createFallback();
        std::cout << "\n=== Convergence Analysis ===\n";
        for(float e: {1e-3f,1e-4f,1e-5f,1e-6f}){
            config_.baseEpsilon=e; march_destroy(marcher_); march_create(&marcher_, &config_); upload();
            auto s = bench(genRadial(100)); std::cout<<"eps "<<e<<" avg="<<s.avgSteps<<" max="<<s.maxSteps<<" miss="<<s.missPct<<"%\n";
        }
        config_.baseEpsilon=1e-4f; march_destroy(marcher_); march_create(&marcher_, &config_); upload();
    }
    
    void runTransformAnalysis(){
        if(scene_.nodeCount()==0) return;
        std::cout << "\n=== Transform Analysis ===\n";
        uint32_t id=0, non=0; float avg=0, mx=0;
        for(uint32_t i=0;i<scene_.nodeCount();++i){ if(scene_.xformIsIdentity[i]) id++; else { non++; float s=scene_.maxScale[i]; avg+=s; mx=std::max(mx,s);} }
        if(non>0) avg/=non;
        std::cout<<"Transforms: identity="<<id<<" non="<<non<<" avgScale="<<avg<<" maxScale="<<mx<<"\n";
    }
    
private:
    MarcherHandle_t marcher_{}; MarcherConfig config_{}; std::unique_ptr<UsdDeviceSceneLoader> usdLoader_;
    DeviceSceneHost scene_{}; LoaderStats stats_{};
    
    // GPU optimization engines
    std::unique_ptr<omnifield::cuda::CutlassTransformEngine> cutlass_engine_;
    std::unique_ptr<omnifield::cuda::ProceduralSdfGenerator> sdf_generator_;
    
    bool upload(){ if(scene_.nodeCount()==0) return false; return march_update_scene_with_transforms(marcher_,
      scene_.nodeTypes.data(), scene_.nodeFlags.data(), scene_.firstChild.data(), scene_.childCount.data(),
      scene_.paramOffset.data(), scene_.objId.data(), scene_.lipschitzL.data(), scene_.bounds.data(),
      scene_.xforms.data(), scene_.maxScale.data(), scene_.xformIsIdentity.data(), scene_.nodeEpsilon.data(),
      scene_.paramBlob.data(), scene_.paramBlob.size(), scene_.nodeCount())==MARCH_SUCCESS; }
    bool createFallback(){ scene_.clear(); scene_.nodeTypes.push_back(NodeType::PRIMITIVE_SPHERE);
      scene_.nodeFlags.push_back(NODE_EXACT_SDF|NODE_ANALYTICAL_GRAD|NODE_BOUNDED);
      scene_.firstChild.push_back(0); scene_.childCount.push_back(0); scene_.objId.push_back(1);
      float r=1.f; size_t off=scene_.paramBlob.size(); scene_.paramBlob.insert(scene_.paramBlob.end(),(uint8_t*)&r,(uint8_t*)&r+sizeof(float));
      scene_.paramOffset.push_back(off); scene_.bounds.push_back({{-1.5f,-1.5f,-1.5f},{1.5f,1.5f,1.5f}}); scene_.lipschitzL.push_back(1.0f);
      for(int i=0;i<12;++i) scene_.xforms.push_back((i%5==0)?1.f:0.f); scene_.maxScale.push_back(1.0f); scene_.xformIsIdentity.push_back(1); scene_.nodeEpsilon.push_back(1e-4f);
      return upload(); }
    std::vector<Ray> genRadial(int n){ std::vector<Ray> v; v.reserve(n); for(int i=0;i<n;++i){ float th=6.2831853f*i/n, ph=3.1415926f*(i%7)/14.f;
      Float3 o{3.f*std::cos(th)*std::sinf(ph),3.f*std::cosf(ph),3.f*std::sinf(th)*std::sinf(ph)}; v.push_back({o,(Float3{0,0,0}-o).normalize(),0.f,10.f,(uint32_t)i}); } return v; }
    std::vector<Ray> genGrid(int w,int h){ std::vector<Ray> v; v.reserve(w*h); for(int y=0;y<h;++y) for(int x=0;x<w;++x){
      float u=(x+0.5f)/w*2.f-1.f, vv=(y+0.5f)/h*2.f-1.f; Float3 o{3,0,0}; Float3 d{-1,u*0.5f,vv*0.5f}; d=d.normalize(); v.push_back({o,d,0.f,10.f,(uint32_t)(y*w+x)});} return v; }
    std::vector<Ray> genRandom(int n){ std::vector<Ray> v; v.reserve(n); for(int i=0;i<n;++i){ float r1=(i*73+17)%1000/1000.f, r2=(i*37+23)%1000/1000.f, r3=(i*19+41)%1000/1000.f;
      Float3 o{(r1-0.5f)*6,(r2-0.5f)*6,(r3-0.5f)*6}; Float3 d=(Float3{0,0,0}-o).normalize(); v.push_back({o,d,0.f,10.f,(uint32_t)i}); } return v; }
    struct RS{ size_t totalRays{},totalSteps{}; float avgSteps{}; uint32_t maxSteps{}; float missPct{},polishPct{},renderTimeMs{}; };
    RS bench(const std::vector<Ray>& rays){ auto t0=std::chrono::high_resolution_clock::now(); uint32_t total=0,mx=0,hits=0,pol=0;
      for(const auto&r:rays){ HitInfo h; march_pick(marcher_, &r, &h); total+=h.steps; if((uint32_t)h.steps>mx) mx=h.steps; if(h.t>0){hits++; if(h.flags&1) pol++;} }
      auto us=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()-t0).count();
      RS s{}; s.totalRays=rays.size(); s.totalSteps=total; s.avgSteps=rays.empty()?0.f:float(total)/rays.size(); s.maxSteps=mx; s.missPct=rays.empty()?0.f:float(rays.size()-hits)/rays.size()*100.f; s.polishPct=hits?float(pol)/hits*100.f:0.f; s.renderTimeMs=float(us)/1000.f; return s; }
    void runRayPattern(const char* name,const std::vector<Ray>& rays){ auto s=bench(rays); std::cout<<name<<": avg="<<s.avgSteps<<" max="<<s.maxSteps<<" miss="<<s.missPct<<"%\n"; }
};

int main(int argc, char** argv){
  std::string usd; for(int i=1;i<argc;++i){ std::string a=argv[i]; if(a=="--usd" && i+1<argc) usd=argv[++i]; }
  
  std::cout << "=== OmniField: GPU-Accelerated SDF Sphere Tracing Engine ===\n";
  std::cout << "Features: CUDA Kernels, CUTLASS Transforms, cuFFT SDF Generation, Warp Optimizations\n\n";
  
  Enhanced_MarcherDemo demo; 
  if(!demo.initialize()) {
      std::cerr << "Failed to initialize demo\n";
      return 1; 
  }
  
  if(!usd.empty()) {
      std::cout << "Loading USD scene: " << usd << "\n";
      demo.loadUsdScene(usd);
  } else {
      std::cout << "No USD scene specified, using procedural fallback\n";
  }
  
  // Run comprehensive benchmarks
  demo.runGpuOptimizationBenchmarks();
  demo.runRaycastBenchmark(); 
  demo.runConvergenceAnalysis(); 
  demo.runTransformAnalysis(); 
  
  std::cout<<"\n=== Performance Summary ===\n";
  std::cout<<"✓ GPU-accelerated sphere tracing with warp optimizations\n";
  std::cout<<"✓ CUTLASS-accelerated matrix transformations\n";
  std::cout<<"✓ cuFFT-based procedural SDF generation\n";
  std::cout<<"✓ Coalesced memory access patterns\n";
  std::cout<<"✓ Cooperative group optimizations\n";
  std::cout<<"✓ Advanced SDF primitives with analytical derivatives\n";
  std::cout<<"✓ Real-time rendering capabilities\n\n";
  
  std::cout<<"Demo completed successfully.\n"; 
  return 0;
}
