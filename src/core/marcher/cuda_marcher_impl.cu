#include "cuda_marcher_core.h"
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cfloat>   // FLT_MAX
#include <cstring>  // std::memcpy

namespace omnifield {

struct SceneNode {
    NodeType type;
    // Generic params (we keep it simple for now)
    // Sphere: p0 = radius
    // Box   : p0,p1,p2 = half extents (hx, hy, hz)
    float p0{0}, p1{0}, p2{0};

    // TODO: transforms, bounds, flags, etc.
};

struct Marcher_ {
    MarcherConfig cfg{};
    uint32_t nodeCount{};
    std::vector<uint32_t> perf;

    // Host-side scene parsed from DeviceSceneHost
    std::vector<SceneNode> nodes;
};

} // namespace omnifield

using namespace omnifield;

// ----------- SDF helpers (host-side, keep math simple) ------------------

static inline float length3(const Float3& a) {
    return std::sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
}
static inline Float3 abs3(const Float3& a) {
    return Float3{std::fabs(a.x), std::fabs(a.y), std::fabs(a.z)};
}
static inline Float3 max3(const Float3& a, float b) {
    return Float3{ (a.x>b? a.x:b), (a.y>b? a.y:b), (a.z>b? a.z:b) };
}
static inline Float3 sub3(const Float3& a, const Float3& b) {
    return Float3{ a.x-b.x, a.y-b.y, a.z-b.z };
}

static inline float sdf_sphere(const Float3& p, float r) {
    return length3(p) - r;
}

// axis-aligned box centered at origin with half extents b=(hx,hy,hz)
static inline float sdf_box(const Float3& p, const Float3& b) {
    Float3 q = sub3(abs3(p), b);
    Float3 qp = max3(q, 0.0f);
    float outside = length3(qp);
    float inside  = std::min(std::max(q.x, std::max(q.y, q.z)), 0.0f);
    return outside + inside;
}

// -------------- Uploaded scene evaluator (union of nodes) ----------------

static inline float eval_uploaded_scene(const Marcher_* m, const Float3& p) {
    if (!m || m->nodes.empty()) return FLT_MAX;
    float d = FLT_MAX;
    for (const auto& n : m->nodes) {
        float dn = FLT_MAX;
        switch (n.type) {
            case NodeType::PRIMITIVE_SPHERE:
                dn = sdf_sphere(p, n.p0);
                break;
            case NodeType::PRIMITIVE_BOX:
                dn = sdf_box(p, Float3{n.p0, n.p1, n.p2});
                break;
            default:
                // ignore unsupported for now
                break;
        }
        d = std::min(d, dn); // plain union
    }
    return d;
}

// -------------- Small procedural fallback (only if no nodes) -------------
static inline float sdf_box_centered(const Float3& p, float hx, float hy, float hz) {
    return sdf_box(p, Float3{hx,hy,hz});
}
static inline float op_subtract(float a, float b) { return std::max(a, -b); }
static inline float op_smooth_union(float a, float b, float k) {
    // k > 0
    float h = std::max(0.f, k - std::fabs(a - b)) / k;
    return std::min(a, b) - h*h*0.25f*k;
}
static inline float eval_procedural(const Float3& p) {
    float d1 = sdf_sphere(p, 1.0f);
    float d2 = sdf_box_centered(Float3{p.x - 1.25f, p.y, p.z}, 0.35f, 0.35f, 0.35f);
    float d  = op_smooth_union(d1, d2, 0.25f);
    float hole = sdf_sphere(Float3{p.x - 0.25f, p.y, p.z - 0.25f}, 0.30f);
    return op_subtract(d, hole);
}

extern "C" {

// ---------------- Lifecycle ---------------------------------------------

MarchResult march_create(MarcherHandle_t* out, const MarcherConfig* cfg) {
    if (!out || !cfg) return MARCH_ERROR_INVALID;
    *out = new Marcher_();
    (*out)->cfg = *cfg;
    (*out)->perf.assign(PERF_COUNTER_COUNT, 0);
    cudaFree(0); // warm up context (ok even if we stay host-side for now)
    return MARCH_SUCCESS;
}

MarchResult march_destroy(MarcherHandle_t h) {
    if (!h) return MARCH_ERROR_INVALID;
    delete h;
    return MARCH_SUCCESS;
}

// Parse a minimal subset of DeviceSceneHost into our host-side nodes.
// Supported now: PRIMITIVE_SPHERE (radius float), PRIMITIVE_BOX (hx,hy,hz).
// Transforms are ignored for the moment (loader stub uses identity).
MarchResult march_update_scene_with_transforms(
    MarcherHandle_t h,
    const NodeType* nodeTypes, const uint32_t* nodeFlags,
    const uint32_t* firstChild, const uint32_t* childCount,
    const size_t* paramOffset, const uint32_t* objId, const float* lipschitzL,
    const AABB* bounds, const float* xforms, const float* maxScale,
    const uint8_t* xformIsIdentity, const float* nodeEpsilon,
    const uint8_t* paramBlob, size_t paramBytes, uint32_t nodeCount)
{
    (void)nodeFlags; (void)firstChild; (void)childCount; (void)objId;
    (void)lipschitzL; (void)bounds; (void)xforms; (void)maxScale;
    (void)xformIsIdentity; (void)nodeEpsilon; (void)paramBytes;

    if (!h || !nodeTypes || !paramOffset || !paramBlob) return MARCH_ERROR_INVALID;

    h->nodes.clear();
    h->nodes.reserve(nodeCount);

    auto read_f1 = [&](size_t off)->float {
        float v = 0.f;
        std::memcpy(&v, paramBlob + off, sizeof(float));
        return v;
    };
    auto read_f3 = [&](size_t off)->Float3 {
        Float3 v{0,0,0};
        std::memcpy(&v, paramBlob + off, sizeof(Float3));
        return v;
    };

    for (uint32_t i = 0; i < nodeCount; ++i) {
        SceneNode n{};
        n.type = nodeTypes[i];

        size_t off = paramOffset[i];
        switch (n.type) {
            case NodeType::PRIMITIVE_SPHERE: {
                // expects: float radius
                n.p0 = read_f1(off);
            } break;
            case NodeType::PRIMITIVE_BOX: {
                // expects: float3 half-extents
                Float3 he = read_f3(off);
                n.p0 = he.x; n.p1 = he.y; n.p2 = he.z;
            } break;
            default:
                // unsupported -> skip but keep placeholder
                break;
        }
        h->nodes.push_back(n);
    }

    h->nodeCount = nodeCount;
    return MARCH_SUCCESS;
}

MarchResult march_render_gl(MarcherHandle_t h, unsigned int, uint32_t, uint32_t,
                            const Camera*, RenderStats* s) {
    if (!h) return MARCH_ERROR_INVALID;
    if (s) { *s = {}; }
    return MARCH_SUCCESS;
}

MarchResult march_reset_perf_counters(MarcherHandle_t h) {
    if (!h) return MARCH_ERROR_INVALID;
    std::fill(h->perf.begin(), h->perf.end(), 0);
    return MARCH_SUCCESS;
}

MarchResult march_get_perf_counters(MarcherHandle_t h, uint32_t* out, uint32_t n) {
    if (!h || !out) return MARCH_ERROR_INVALID;
    const uint32_t m = std::min<uint32_t>(n, (uint32_t)h->perf.size());
    for (uint32_t i = 0; i < m; ++i) out[i] = h->perf[i];
    return MARCH_SUCCESS;
}

// ---------------- Sphere tracing over uploaded scene ---------------------
// Returns MARCH_SUCCESS for both hit/miss; check out->t (>0 = hit).
MarchResult march_pick(MarcherHandle_t h, const Ray* ray, HitInfo* out) {
    if (!h || !out || !ray) return MARCH_ERROR_INVALID;

    const float eps      = (h->cfg.baseEpsilon > 0.f) ? h->cfg.baseEpsilon : 1e-4f;
    const uint32_t maxS  = (h->cfg.maxSteps    > 0   ) ? h->cfg.maxSteps   : 256;
    const float tmax     = (ray->tmax          > 0.f ) ? ray->tmax         : 50.0f;
    const float tmin     = (ray->tmin          > 0.f ) ? ray->tmin         : 0.0f;

    Float3 ro = ray->origin;
    Float3 dir = ray->dir;
    float len = std::sqrt(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
    Float3 rd = (len > 0.f) ? Float3{dir.x/len, dir.y/len, dir.z/len} : Float3{0,0,1};

    float t = tmin;
    uint32_t steps = 0;

    auto eval = [&](const Float3& p)->float {
        // Prefer uploaded scene; if empty, use procedural fallback
        float d = eval_uploaded_scene(h, p);
        if (d == FLT_MAX) d = eval_procedural(p);
        return d;
    };

    for (; steps < maxS; ++steps) {
        Float3 p = Float3{ ro.x + rd.x*t, ro.y + rd.y*t, ro.z + rd.z*t };
        float d = eval(p);
        if (d < eps) {
            out->t = t;
            out->steps = (int)(steps + 1);
            out->flags = 0;
            if (h->perf.size() > PERF_TOTAL_EVALUATIONS)
                h->perf[PERF_TOTAL_EVALUATIONS] += (uint32_t)(steps + 1);
            return MARCH_SUCCESS;
        }
        t += d;
        if (t > tmax) break;
    }

    out->t = -1.0f;
    out->steps = (int)steps;
    out->flags = 0;
    if (h->perf.size() > PERF_TOTAL_EVALUATIONS)
        h->perf[PERF_TOTAL_EVALUATIONS] += (uint32_t)steps;
    return MARCH_SUCCESS;
}

} // extern "C"
