#include "cuda_marcher_core.h"
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstring>

namespace omnifield {

// Minimal per-node record stored host-side for evaluation
struct SceneNode {
    NodeType type{NodeType::PRIMITIVE_SPHERE};

    // Primitive params
    // Sphere: params: [float radius]
    // Box   : params: [float hx, float hy, float hz]
    float p0{0}, p1{0}, p2{0};

    // Operator params
    // CSG_SMOOTH_UNION: params: [float k] (blend radius)
    float k{0.0f};

    // Child range in SoA (we assume post-order or arbitrary DAG with child list)
    uint32_t firstChild{0};
    uint32_t childCount{0};

    // Transform (store inverse 3x4 row-major); applied only on leaves
    float inv[12]{ 1,0,0,0,
                   0,1,0,0,
                   0,0,1,0 };
    float maxScale{1.0f};
    uint8_t isIdentity{1};
};

struct Marcher_ {
    MarcherConfig cfg{};
    uint32_t nodeCount{};
    std::vector<uint32_t> perf;

    // Parsed host-side nodes
    std::vector<SceneNode> nodes;
};

} // namespace omnifield

using namespace omnifield;

// ---------------- math helpers ----------------
static inline float length3(const Float3& a){ return std::sqrt(a.x*a.x + a.y*a.y + a.z*a.z); }
static inline Float3 abs3(const Float3&a){ return {std::fabs(a.x),std::fabs(a.y),std::fabs(a.z)}; }
static inline Float3 sub3(const Float3&a,const Float3&b){ return {a.x-b.x,a.y-b.y,a.z-b.z}; }
static inline Float3 max3f(const Float3&a,float b){ return { (a.x>b?a.x:b), (a.y>b?a.y:b), (a.z>b?a.z:b) }; }

static bool invert3x4(const float m[12], float inv[12]){
    const float r00=m[0], r01=m[1], r02=m[2],  tx=m[3];
    const float r10=m[4], r11=m[5], r12=m[6],  ty=m[7];
    const float r20=m[8], r21=m[9], r22=m[10], tz=m[11];

    const float c00 =  (r11*r22 - r12*r21);
    const float c01 = -(r10*r22 - r12*r20);
    const float c02 =  (r10*r21 - r11*r20);
    const float det = r00*c00 + r01*c01 + r02*c02;
    if (std::fabs(det) < 1e-12f){
        inv[0]=1;inv[1]=0;inv[2]=0;inv[3]=0;
        inv[4]=0;inv[5]=1;inv[6]=0;inv[7]=0;
        inv[8]=0;inv[9]=0;inv[10]=1;inv[11]=0;
        return false;
    }
    const float id = 1.0f/det;

    const float i00 =  c00*id;
    const float i01 = -(r01*r22 - r02*r21)*id;
    const float i02 =  (r01*r12 - r02*r11)*id;

    const float i10 =  c01*id;
    const float i11 =  (r00*r22 - r02*r20)*id;
    const float i12 = -(r00*r12 - r02*r10)*id;

    const float i20 =  c02*id;
    const float i21 = -(r00*r21 - r01*r20)*id;
    const float i22 =  (r00*r11 - r01*r10)*id;

    const float itx = -(i00*tx + i01*ty + i02*tz);
    const float ity = -(i10*tx + i11*ty + i12*tz);
    const float itz = -(i20*tx + i21*ty + i22*tz);

    inv[0]=i00; inv[1]=i01; inv[2]=i02; inv[3]=itx;
    inv[4]=i10; inv[5]=i11; inv[6]=i12; inv[7]=ity;
    inv[8]=i20; inv[9]=i21; inv[10]=i22; inv[11]=itz;
    return true;
}

static inline Float3 xformPoint_inv(const float inv[12], const Float3& p){
    return {
        inv[0]*p.x + inv[1]*p.y + inv[2]*p.z + inv[3],
        inv[4]*p.x + inv[5]*p.y + inv[6]*p.z + inv[7],
        inv[8]*p.x + inv[9]*p.y + inv[10]*p.z + inv[11]
    };
}

// --------------- primitive SDFs (local space) ---------------
static inline float sdf_sphere_local(const Float3& p, float r){ return length3(p) - r; }
static inline float sdf_box_local(const Float3& p, const Float3& he){
    const Float3 q = sub3(abs3(p), he);
    const Float3 qp = max3f(q, 0.0f);
    const float outside = length3(qp);
    const float inside  = std::min(std::max(q.x, std::max(q.y, q.z)), 0.0f);
    return outside + inside;
}

// --------------- CSG ops (hard + smooth union) ---------------
static inline float op_union(float a, float b)      { return std::min(a,b); }
static inline float op_intersect(float a, float b)  { return std::max(a,b); }
static inline float op_subtract(float a, float b)   { return std::max(a, -b); }
// Quilez-style poly smooth union; k>0
static inline float op_smooth_union(float a, float b, float k){
    if (k <= 0.f) return std::min(a,b);
    const float h = std::max(0.f, k - std::fabs(a-b)) / k;
    return std::min(a,b) - h*h*0.25f*k;
}

// --------------- eval uploaded scene (recursive) ---------------
static float eval_node(const Marcher_* m, uint32_t i, const Float3& pW){
    const SceneNode& n = m->nodes[i];
    switch(n.type){
        // Leaves (apply world->local, then primitive; rescale by 1/maxScale for Lipschitz safety)
        case NodeType::PRIMITIVE_SPHERE: {
            const Float3 pL = n.isIdentity ? pW : xformPoint_inv(n.inv, pW);
            const float dL = sdf_sphere_local(pL, n.p0);
            const float L  = (n.maxScale>0.f)? n.maxScale : 1.f;
            return dL / L;
        }
        case NodeType::PRIMITIVE_BOX: {
            const Float3 pL = n.isIdentity ? pW : xformPoint_inv(n.inv, pW);
            const float dL = sdf_box_local(pL, {n.p0,n.p1,n.p2});
            const float L  = (n.maxScale>0.f)? n.maxScale : 1.f;
            return dL / L;
        }

        // Internal nodes (assume binary for now)
        case NodeType::CSG_UNION:
        case NodeType::CSG_INTERSECT:
        case NodeType::CSG_SUBTRACT:
        case NodeType::CSG_SMOOTH_UNION: {
            const uint32_t fc = n.firstChild;
            const uint32_t cc = n.childCount;
            if (cc == 0) return FLT_MAX;
            if (cc == 1)  return eval_node(m, fc, pW);
            // Use first two children for binary ops
            const float a = eval_node(m, fc+0, pW);
            const float b = eval_node(m, fc+1, pW);
            switch(n.type){
                case NodeType::CSG_UNION:         return op_union(a,b);
                case NodeType::CSG_INTERSECT:     return op_intersect(a,b);
                case NodeType::CSG_SUBTRACT:      return op_subtract(a,b);
                case NodeType::CSG_SMOOTH_UNION:  return op_smooth_union(a,b, n.k>0.f?n.k:0.0f);
                default: break;
            }
            return std::min(a,b);
        }

        default: return FLT_MAX;
    }
}

static inline float eval_uploaded_scene(const Marcher_* m, const Float3& pW){
    if (!m || m->nodes.empty()) return FLT_MAX;
    // Assume the last node is the root if there are operators; otherwise union all leaves.
    const SceneNode& root = m->nodes.back();
    if (root.childCount>0 || root.type==NodeType::CSG_UNION || root.type==NodeType::CSG_INTERSECT
        || root.type==NodeType::CSG_SUBTRACT || root.type==NodeType::CSG_SMOOTH_UNION){
        return eval_node(m, (uint32_t)(m->nodes.size()-1), pW);
    }
    // flat union fallback
    float d = FLT_MAX;
    for (uint32_t i=0;i<m->nodes.size();++i){
        const SceneNode& n = m->nodes[i];
        if (n.childCount!=0) continue;
        d = std::min(d, eval_node(m, i, pW));
    }
    return d;
}

// -------- procedural fallback if no uploaded nodes ----------
static inline float procedural_eval(const Float3& p){
    // sphere ∪ box with a hole
    float d1 = sdf_sphere_local(p, 1.0f);
    float d2 = sdf_box_local({p.x-1.25f,p.y,p.z}, {0.35f,0.35f,0.35f});
    float d  = op_smooth_union(d1,d2,0.25f);
    float hole = sdf_sphere_local({p.x-0.25f,p.y,p.z-0.25f}, 0.30f);
    return op_subtract(d, hole);
}

extern "C" {

// ---------------- lifecycle ----------------
MarchResult march_create(MarcherHandle_t* out, const MarcherConfig* cfg){
    if (!out || !cfg) return MARCH_ERROR_INVALID;
    *out = new Marcher_();
    (*out)->cfg = *cfg;
    (*out)->perf.assign(PERF_COUNTER_COUNT, 0);
    cudaFree(0);
    return MARCH_SUCCESS;
}

MarchResult march_destroy(MarcherHandle_t h){
    if (!h) return MARCH_ERROR_INVALID;
    delete h;
    return MARCH_SUCCESS;
}

// Parse your DeviceSceneHost SoA into our compact nodes vector.
// Layout matches your headers: nodeTypes/firstChild/childCount/paramOffset/xforms/maxScale/xformIsIdentity/paramBlob.
MarchResult march_update_scene_with_transforms(
    MarcherHandle_t h,
    const NodeType* nodeTypes, const uint32_t* nodeFlags,
    const uint32_t* firstChild, const uint32_t* childCount,
    const size_t* paramOffset, const uint32_t* objId, const float* lipschitzL,
    const AABB* bounds, const float* xforms, const float* maxScale,
    const uint8_t* xformIsIdentity, const float* nodeEpsilon,
    const uint8_t* paramBlob, size_t paramBytes, uint32_t nodeCount)
{
    (void)nodeFlags; (void)objId; (void)lipschitzL; (void)bounds; (void)nodeEpsilon;

    if (!h || !nodeTypes || !firstChild || !childCount || !paramOffset || !paramBlob) return MARCH_ERROR_INVALID;

    h->nodes.clear();
    h->nodes.resize(nodeCount);

    auto read_f1 = [&](size_t off)->float{
        if (off+sizeof(float) > paramBytes) return 0.f;
        float v=0; std::memcpy(&v, paramBlob+off, sizeof(float)); return v;
    };
    auto read_f3 = [&](size_t off)->Float3{
        Float3 v{0,0,0};
        if (off+3*sizeof(float) <= paramBytes){
            std::memcpy(&v.x, paramBlob+off+0*sizeof(float), sizeof(float));
            std::memcpy(&v.y, paramBlob+off+1*sizeof(float), sizeof(float));
            std::memcpy(&v.z, paramBlob+off+2*sizeof(float), sizeof(float));
        }
        return v;
    };

    for (uint32_t i=0;i<nodeCount;++i){
        SceneNode& n = h->nodes[i];
        n.type = nodeTypes[i];
        n.firstChild = firstChild[i];
        n.childCount = childCount[i];

        // Params by type
        const size_t off = paramOffset[i];
        switch(n.type){
            case NodeType::PRIMITIVE_SPHERE: {
                n.p0 = read_f1(off); // radius
            } break;
            case NodeType::PRIMITIVE_BOX: {
                const Float3 he = read_f3(off); // half-extents
                n.p0 = he.x; n.p1 = he.y; n.p2 = he.z;
            } break;
            case NodeType::CSG_SMOOTH_UNION: {
                n.k = read_f1(off); // blend radius (optional)
                if (n.k <= 0.f) n.k = 0.25f;
            } break;
            default: break;
        }

        // Transforms (leaves): inverse + maxScale; internal nodes are identity
        n.isIdentity = xformIsIdentity ? xformIsIdentity[i] : 1;
        n.maxScale   = maxScale ? maxScale[i] : 1.0f;
        if (!n.isIdentity && xforms){
            const float* m = xforms + i*12;
            invert3x4(m, n.inv); // safe even if singular (falls back to identity)
        } else {
            n.inv[0]=1; n.inv[1]=0; n.inv[2]=0; n.inv[3]=0;
            n.inv[4]=0; n.inv[5]=1; n.inv[6]=0; n.inv[7]=0;
            n.inv[8]=0; n.inv[9]=0; n.inv[10]=1; n.inv[11]=0;
        }
    }

    h->nodeCount = nodeCount;
    return MARCH_SUCCESS;
}

MarchResult march_render_gl(MarcherHandle_t h, unsigned int, uint32_t, uint32_t,
                            const Camera*, RenderStats* s){
    if (!h) return MARCH_ERROR_INVALID;
    if (s) *s = {};
    return MARCH_SUCCESS;
}

MarchResult march_reset_perf_counters(MarcherHandle_t h){
    if (!h) return MARCH_ERROR_INVALID;
    std::fill(h->perf.begin(), h->perf.end(), 0);
    return MARCH_SUCCESS;
}

MarchResult march_get_perf_counters(MarcherHandle_t h, uint32_t* out, uint32_t n){
    if (!h || !out) return MARCH_ERROR_INVALID;
    const uint32_t m = std::min<uint32_t>(n, (uint32_t)h->perf.size());
    for (uint32_t i=0;i<m;++i) out[i] = h->perf[i];
    return MARCH_SUCCESS;
}

// ---------------- sphere tracing ----------------
MarchResult march_pick(MarcherHandle_t h, const Ray* ray, HitInfo* out){
    if (!h || !out || !ray) return MARCH_ERROR_INVALID;

    const float eps      = (h->cfg.baseEpsilon > 0.f) ? h->cfg.baseEpsilon : 1e-4f;
    const uint32_t maxS  = (h->cfg.maxSteps    > 0   ) ? h->cfg.maxSteps   : 256;
    const float tmax     = (ray->tmax          > 0.f ) ? ray->tmax         : 50.0f;
    const float tmin     = (ray->tmin          > 0.f ) ? ray->tmin         : 0.0f;

    Float3 ro = ray->origin;
    Float3 d  = ray->dir;
    float Ld = std::sqrt(d.x*d.x + d.y*d.y + d.z*d.z);
    Float3 rd = (Ld>0.f)? Float3{d.x/Ld, d.y/Ld, d.z/Ld} : Float3{0,0,1};

    float t = tmin;
    uint32_t steps = 0;

    auto eval = [&](const Float3& pw)->float{
        float du = eval_uploaded_scene(h, pw);
        if (du == FLT_MAX) du = procedural_eval(pw);
        return du;
    };

    for (; steps<maxS; ++steps){
        const Float3 p = { ro.x + rd.x*t, ro.y + rd.y*t, ro.z + rd.z*t };
        const float dist = eval(p);
        if (dist < eps){
            out->t = t;
            out->steps = (int)(steps+1);
            out->flags = 0;
            if (h->perf.size() > PERF_TOTAL_EVALUATIONS)
                h->perf[PERF_TOTAL_EVALUATIONS] += (uint32_t)(steps+1);
            return MARCH_SUCCESS;
        }
        t += dist;
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