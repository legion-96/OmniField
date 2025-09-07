#include "../src/core/marcher/cuda_marcher_core.h"
#include "../src/host/usd_loader/UsdDeviceSceneLoader.h"
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
        std::cout<<"SDFCAD Phase 2 Demo Initialized\n"; return true;
    }
    void cleanup(){ if(marcher_){march_destroy(marcher_); marcher_=nullptr;} usdLoader_.reset(); }
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
    void runRaycastBenchmark(){
        if(scene_.nodeCount()==0) createFallback();
        runRayPattern("Radial rays", genRadial(1000));
        runRayPattern("Grid rays", genGrid(32,32));
        runRayPattern("Random rays", genRandom(1000));
        march_reset_perf_counters(marcher_);
        auto rays = genRadial(256); for(auto& r: rays){ HitInfo h; march_pick(marcher_, &r, &h); }
        uint32_t c[PERF_COUNTER_COUNT]{0}; march_get_perf_counters(marcher_, c, PERF_COUNTER_COUNT);
        std::cout<<"Perf total eval: "<<c[PERF_TOTAL_EVALUATIONS]<<"\n";
    }
    void runConvergenceAnalysis(){
        if(scene_.nodeCount()==0) createFallback();
        for(float e: {1e-3f,1e-4f,1e-5f,1e-6f}){
            config_.baseEpsilon=e; march_destroy(marcher_); march_create(&marcher_, &config_); upload();
            auto s = bench(genRadial(100)); std::cout<<"eps "<<e<<" avg="<<s.avgSteps<<" max="<<s.maxSteps<<" miss="<<s.missPct<<"%\n";
        }
        config_.baseEpsilon=1e-4f; march_destroy(marcher_); march_create(&marcher_, &config_); upload();
    }
    void runTransformAnalysis(){
        if(scene_.nodeCount()==0) return;
        uint32_t id=0, non=0; float avg=0, mx=0;
        for(uint32_t i=0;i<scene_.nodeCount();++i){ if(scene_.xformIsIdentity[i]) id++; else { non++; float s=scene_.maxScale[i]; avg+=s; mx=std::max(mx,s);} }
        if(non>0) avg/=non;
        std::cout<<"Transforms: identity="<<id<<" non="<<non<<" avgScale="<<avg<<" maxScale="<<mx<<"\n";
    }
private:
    MarcherHandle_t marcher_{}; MarcherConfig config_{}; std::unique_ptr<UsdDeviceSceneLoader> usdLoader_;
    DeviceSceneHost scene_{}; LoaderStats stats_{};
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
  Enhanced_MarcherDemo demo; if(!demo.initialize()) return 1; if(!usd.empty()) demo.loadUsdScene(usd);
  demo.runRaycastBenchmark(); demo.runConvergenceAnalysis(); demo.runTransformAnalysis(); std::cout<<"Done.\n"; return 0;
}
