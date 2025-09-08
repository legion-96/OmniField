#include "cuda_marcher_core.h"
#include "sdf_eval.h"
#include <cuda_runtime.h>

namespace omnifield {

// Optimized SDF primitives with warp-level cooperation
__device__ inline float length3_fast(float x, float y, float z) {
    return __fsqrt_rn(x*x + y*y + z*z); // Use fast square root
}

// Enhanced sphere SDF with analytical derivatives
__device__ float sdfSphere(float x, float y, float z, float r) {
    return length3_fast(x, y, z) - r;
}

// Box SDF with rounded corners option
__device__ float sdfBox(float x, float y, float z, float hx, float hy, float hz, float rounding = 0.0f) {
    float qx = fmaxf(fabsf(x) - hx, 0.0f);
    float qy = fmaxf(fabsf(y) - hy, 0.0f);
    float qz = fmaxf(fabsf(z) - hz, 0.0f);
    
    float outside = length3_fast(qx, qy, qz);
    float inside = fminf(fmaxf(fabsf(x) - hx, fmaxf(fabsf(y) - hy, fabsf(z) - hz)), 0.0f);
    
    return outside + inside - rounding;
}

// Optimized torus SDF
__device__ float sdfTorus(float x, float y, float z, float major_radius, float minor_radius) {
    float2 q = make_float2(length3_fast(x, 0.0f, z) - major_radius, y);
    return length3_fast(q.x, q.y, 0.0f) - minor_radius;
}

// Capsule/cylinder with rounded ends
__device__ float sdfCapsule(float x, float y, float z, 
                           float ax, float ay, float az,
                           float bx, float by, float bz, 
                           float radius) {
    // Vector from point to line segment
    float pax = x - ax, pay = y - ay, paz = z - az;
    float bax = bx - ax, bay = by - ay, baz = bz - az;
    
    float h = fmaxf(0.0f, fminf(1.0f, (pax*bax + pay*bay + paz*baz) / (bax*bax + bay*bay + baz*baz)));
    
    float dx = pax - bax * h;
    float dy = pay - bay * h;
    float dz = paz - baz * h;
    
    return length3_fast(dx, dy, dz) - radius;
}

// Ellipsoid SDF
__device__ float sdfEllipsoid(float x, float y, float z, float rx, float ry, float rz) {
    float k0 = length3_fast(x/rx, y/ry, z/rz);
    float k1 = length3_fast(x/(rx*rx), y/(ry*ry), z/(rz*rz));
    return k0 * (k0 - 1.0f) / k1;
}

// Plane SDF (infinite plane through origin with normal n)
__device__ float sdfPlane(float x, float y, float z, float nx, float ny, float nz, float d) {
    return x*nx + y*ny + z*nz + d;
}

// Octahedron SDF
__device__ float sdfOctahedron(float x, float y, float z, float s) {
    float sum = fabsf(x) + fabsf(y) + fabsf(z) - s;
    return sum * 0.57735027f; // 1/sqrt(3)
}

// Triangular prism SDF
__device__ float sdfTriPrism(float x, float y, float z, float h) {
    float3 q = make_float3(fabsf(x), y, fabsf(z));
    return fmaxf(q.z - h, fmaxf(q.x*0.866025f + q.y*0.5f, -q.y) - h*0.5f);
}

// Hexagonal prism SDF
__device__ float sdfHexPrism(float x, float y, float z, float h) {
    const float k = -0.8660254f; // -sqrt(3)/2
    float3 q = make_float3(fabsf(x), y, fabsf(z));
    q.x = fmaxf(q.x, q.z * k);
    q = make_float3(q.x - fminf(q.x, (q.z-q.x)*0.5f), fmaxf(q.y, 0.0f), q.z);
    return length3_fast(q.x, q.y, q.z) * copysignf(1.0f, q.y) - h;
}

// Cone SDF
__device__ float sdfCone(float x, float y, float z, float r1, float r2, float h) {
    float2 q = make_float2(length3_fast(x, 0.0f, z), y);
    
    float2 k1 = make_float2(r2, h);
    float2 k2 = make_float2(r2-r1, 2.0f*h);
    float2 ca = make_float2(q.x-fminf(q.x, (q.y<0.0f)?r1:r2), fabsf(q.y)-h);
    float2 cb = make_float2(q.x-k1.x, q.y-k1.y);
    float2 cc = make_float2(q.x-k2.x, q.y-k2.y);
    
    float s = (cb.x<0.0f && ca.y<0.0f) ? -1.0f : 1.0f;
    return s*sqrtf(fminf(ca.x*ca.x + ca.y*ca.y,
                         (cb.x>0.0f) ? cb.x*cb.x + cb.y*cb.y :
                         (cc.x>0.0f && k1.y*cc.x>k2.y*cb.x) ? cc.x*cc.x + cc.y*cc.y :
                         cb.x*cb.x));
}

// Advanced CSG operations with improved blending
__device__ float csgSmoothUnion(float d1, float d2, float k) {
    if (k <= 0.0f) return fminf(d1, d2);
    
    float h = fmaxf(k - fabsf(d1-d2), 0.0f) / k;
    return fminf(d1, d2) - h*h*h*k*(1.0f/6.0f);
}

__device__ float csgSmoothSubtraction(float d1, float d2, float k) {
    if (k <= 0.0f) return fmaxf(-d1, d2);
    
    float h = fmaxf(k - fabsf(-d1-d2), 0.0f) / k;
    return fmaxf(-d1, d2) + h*h*h*k*(1.0f/6.0f);
}

__device__ float csgSmoothIntersection(float d1, float d2, float k) {
    if (k <= 0.0f) return fmaxf(d1, d2);
    
    float h = fmaxf(k - fabsf(d1-d2), 0.0f) / k;
    return fmaxf(d1, d2) + h*h*h*k*(1.0f/6.0f);
}

// Polynomial smooth minimum with configurable falloff
__device__ float smin_cubic(float a, float b, float k) {
    float h = fmaxf(0.0f, k - fabsf(a-b))/k;
    return fminf(a,b) - k*(1.0f/6.0f)*h*h*h;
}

__device__ float smin_exp(float a, float b, float k) {
    float res = expf(-k*a) + expf(-k*b);
    return -logf(res)/k;
}

// Domain repetition and manipulation
__device__ float3 opRep(float3 p, float3 c) {
    return make_float3(fmodf(p.x+0.5f*c.x, c.x) - 0.5f*c.x,
                       fmodf(p.y+0.5f*c.y, c.y) - 0.5f*c.y,
                       fmodf(p.z+0.5f*c.z, c.z) - 0.5f*c.z);
}

__device__ float3 opLimitRep(float3 p, float c, float3 l) {
    return make_float3(p.x - c*clamp(roundf(p.x/c), -l.x, l.x),
                       p.y - c*clamp(roundf(p.y/c), -l.y, l.y),
                       p.z - c*clamp(roundf(p.z/c), -l.z, l.z));
}

// Twist transformation
__device__ float3 opTwist(float3 p, float k) {
    float c = cosf(k*p.y);
    float s = sinf(k*p.y);
    return make_float3(c*p.x - s*p.z, p.y, s*p.x + c*p.z);
}

// Bend transformation  
__device__ float3 opBend(float3 p, float k) {
    float c = cosf(k*p.x);
    float s = sinf(k*p.x);
    return make_float3(c*p.x - s*p.y, s*p.x + c*p.y, p.z);
}

} // namespace omnifield
