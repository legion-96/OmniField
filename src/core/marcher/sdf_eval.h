#pragma once
// Minimal SDF utilities for host/device use.

#ifdef __CUDACC__
#define OF_HD __host__ __device__
#else
#define OF_HD
#endif

#include <cmath>

namespace omnifield {

struct Float3 {
  float x, y, z;
  OF_HD Float3() : x(0), y(0), z(0) {}
  OF_HD Float3(float X, float Y, float Z) : x(X), y(Y), z(Z) {}
};

OF_HD inline Float3 make3(float x,float y,float z){ return Float3{x,y,z}; }

OF_HD inline Float3 operator+(const Float3&a,const Float3&b){ return make3(a.x+b.x,a.y+b.y,a.z+b.z); }
OF_HD inline Float3 operator-(const Float3&a,const Float3&b){ return make3(a.x-b.x,a.y-b.y,a.z-b.z); }
OF_HD inline Float3 operator*(const Float3&a,float s){ return make3(a.x*s,a.y*s,a.z*s); }
OF_HD inline Float3 operator*(float s,const Float3&a){ return a*s; }
OF_HD inline Float3 operator/(const Float3&a,float s){ return make3(a.x/s,a.y/s,a.z/s); }

OF_HD inline float dot(const Float3&a,const Float3&b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
OF_HD inline float length(const Float3&a){ return std::sqrt(dot(a,a)); }
OF_HD inline Float3 normalize(const Float3&a){ float L=length(a); return (L>0.f)? a/L : a; }

// --- Primitive SDFs ---
OF_HD inline float sdf_sphere(const Float3&p, const Float3& c, float r){
  return length(p - c) - r;
}

// Axis-aligned box with half-extents b
OF_HD inline float sdf_box(const Float3&p, const Float3& b){
  Float3 q = make3(std::fabs(p.x), std::fabs(p.y), std::fabs(p.z)) - b;
  Float3 qpos = make3(fmaxf(q.x,0.f), fmaxf(q.y,0.f), fmaxf(q.z,0.f));
  float outside = length(qpos);
  float inside = fminf(fmaxf(q.x, fmaxf(q.y, q.z)), 0.0f);
  return outside + inside;
}

// --- CSG / blending ops ---
OF_HD inline float op_union(float a, float b){ return fminf(a,b); }
OF_HD inline float op_subtract(float a, float b){ return fmaxf(a,-b); }
OF_HD inline float op_intersect(float a, float b){ return fmaxf(a,b); }

// Polynomial smooth union (Inigo Quilez)
OF_HD inline float op_smooth_union(float a, float b, float k){
  // k > 0
  float h = fmaxf(0.f, k - fabsf(a - b)) / k;
  return fminf(a, b) - h*h*0.25f*k;
}

// Estimate normal by finite differences
OF_HD inline Float3 estimate_normal(const Float3& p, float (*eval)(const Float3&)){
  const float e = 1e-4f;
  const Float3 ex = make3(e,0,0), ey = make3(0,e,0), ez = make3(0,0,e);
  float dx = eval(p + ex) - eval(p - ex);
  float dy = eval(p + ey) - eval(p - ey);
  float dz = eval(p + ez) - eval(p - ez);
  return normalize(make3(dx,dy,dz));
}

} // namespace omnifield
