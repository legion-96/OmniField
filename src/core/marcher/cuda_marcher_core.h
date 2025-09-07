#pragma once
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>

namespace omnifield {

struct Float3{ float x{},y{},z{}; Float3()=default; Float3(float X,float Y,float Z):x(X),y(Y),z(Z){}
  Float3 operator-(const Float3&o)const{return {x-o.x,y-o.y,z-o.z};}
  Float3 operator+(const Float3&o)const{return {x+o.x,y+o.y,z+o.z};}
  Float3 operator*(float s)const{return {x*s,y*s,z*s};}
  float length()const{return std::sqrt(x*x+y*y+z*z);} Float3 normalize()const{float L=length();return L>0?Float3{x/L,y/L,z/L}:*this;}
};
struct AABB{ Float3 min{},max{}; };

struct Transform3x4{ float m[12]{}; Transform3x4(){ for(int i=0;i<12;++i) m[i]=(i%5==0)?1.f:0.f; } explicit Transform3x4(const float* a){ for(int i=0;i<12;++i)m[i]=a[i]; }
  bool isOrthonormal() const { return true; }
  Transform3x4 fastInverse() const { Transform3x4 r; return r; } // stub
  Transform3x4 generalInverse() const { return fastInverse(); } // stub
  Float3 transformPoint(const Float3& p) const {
    return { m[0]*p.x+m[1]*p.y+m[2]*p.z+m[3],
             m[4]*p.x+m[5]*p.y+m[6]*p.z+m[7],
             m[8]*p.x+m[9]*p.y+m[10]*p.z+m[11] };
  }
};

enum class NodeType : uint32_t { PRIMITIVE_SPHERE=0, PRIMITIVE_BOX, PRIMITIVE_CAPSULE, PRIMITIVE_TORUS,
  CSG_UNION, CSG_INTERSECT, CSG_SUBTRACT, CSG_SMOOTH_UNION, EFUNC_PATCH, BRUSH_DELTA };
enum NodeFlags : uint32_t { NODE_NONE=0, NODE_EXACT_SDF=1<<0, NODE_ANALYTICAL_GRAD=1<<1, NODE_BOUNDED=1<<2 };

struct Ray{ Float3 origin{}, dir{}; float tmin{}, tmax{}; uint32_t id{}; };
struct HitInfo{ float t{-1.0f}; int steps{0}; uint32_t flags{0}; };
struct RenderStats{ size_t totalRays{},totalSteps{}; float avgSteps{}; uint32_t maxSteps{}; float missPct{},polishPct{},renderTimeMs{}; };
enum PerfCounters : uint32_t { PERF_TOTAL_EVALUATIONS=0, PERF_PRIMITIVE_EVALUATIONS, PERF_CSG_OPERATIONS, PERF_TRANSFORM_FAST_PATH, PERF_TRANSFORM_GENERAL_PATH, PERF_COUNTER_COUNT };
struct Camera{ Float3 position{}, direction{}, up{0,1,0}; float fovY{45.f}, aspect{1.f}, nearZ{0.1f}, farZ{100.f}; };

struct DeviceSceneHost{
  std::vector<NodeType>   nodeTypes;
  std::vector<uint32_t>   nodeFlags, firstChild, childCount;
  std::vector<size_t>     paramOffset;
  std::vector<uint32_t>   objId;
  std::vector<float>      lipschitzL;
  std::vector<AABB>       bounds;
  std::vector<float>      xforms, maxScale;     // 12 floats per node
  std::vector<uint8_t>    xformIsIdentity;
  std::vector<float>      nodeEpsilon;
  std::vector<uint8_t>    paramBlob;
  uint32_t nodeCount() const { return (uint32_t)nodeTypes.size(); }
  void clear(){ nodeTypes.clear(); nodeFlags.clear(); firstChild.clear(); childCount.clear(); paramOffset.clear(); objId.clear();
    lipschitzL.clear(); bounds.clear(); xforms.clear(); maxScale.clear(); xformIsIdentity.clear(); nodeEpsilon.clear(); paramBlob.clear(); }
};

using MarcherHandle_t = struct Marcher_*;
struct MarcherConfig{ float baseEpsilon{1e-4f}, epsilonScale{0.5f}; uint32_t maxSteps{128}, maxPolishSteps{8}, tileSize{16}, tilesX{32}, tilesY{32}; bool enablePolish{true}, deterministicMode{true}; uint32_t seed{42}; };
enum MarchResult{ MARCH_SUCCESS=0, MARCH_ERROR_INVALID=-1 };

extern "C"{
MarchResult march_create(MarcherHandle_t* out, const MarcherConfig* cfg);
MarchResult march_destroy(MarcherHandle_t h);
MarchResult march_update_scene_with_transforms(MarcherHandle_t h,
  const NodeType*, const uint32_t*, const uint32_t*, const uint32_t*, const size_t*, const uint32_t*,
  const float*, const AABB*, const float*, const float*, const uint8_t*, const float*, const uint8_t*, size_t, uint32_t);
MarchResult march_render_gl(MarcherHandle_t, unsigned int, uint32_t, uint32_t, const Camera*, RenderStats*);
MarchResult march_reset_perf_counters(MarcherHandle_t);
MarchResult march_get_perf_counters(MarcherHandle_t, uint32_t* out, uint32_t n);
MarchResult march_pick(MarcherHandle_t, const Ray*, HitInfo* out);
}

} // namespace omnifield
