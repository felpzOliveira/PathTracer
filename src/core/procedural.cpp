#include <procedural.h>

#define MAX_STEPS 256
#define WARN_NORMAL 10

__bidevice__ Float Dot2(const vec3f &v){
    return Dot(v, v);
}

__bidevice__ Float NegDot(const vec2f &a, const vec2f &b){
    return (a.x * b.x - a.y * b.y);
}

__bidevice__ Float Sign(Float x){
    if(IsZero(x)) return 0;
    if(x > 0) return 1;
    return -1;
}

__bidevice__ vec3f ZRot(vec3f p, Float angle){
    Float a = Radians(angle);
    Float co = std::cos(a);
    Float si = std::sin(a);
    return vec3f(co * p.x - si * p.y, si * p.x + co * p.y, p.z);
}

__bidevice__ Float SmoothMax(Float a, Float b, Float k){
    Float h = Clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return Mix(a, b, h) + k * h * (1.0 - h);
}

__bidevice__ Float SmoothMin(Float a, Float b, Float k){
    float h = Clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return Mix(b, a, h) - k * h * (1.0 - h);
}

__bidevice__ Float SDF::Sphere(vec3f p, Float radius){
    return p.Length() - radius;
}

__bidevice__ Float SDF::Box(vec3f p, vec3f boxLength){
    vec3f d = Abs(p) - boxLength;
    return Min(Max(d.x,Max(d.y,d.z)),0.0) + (Max(d,vec3f(0.0)).Length());
}

__bidevice__ Float SDF::BoxRound(vec3f p, vec3f boxLength, Float radius){
    return SDF::Box(p, boxLength) - radius;
}

__bidevice__ Float SDF::Ellipse(vec3f p, vec3f center, vec3f axis){
    vec3f ir(1.f / axis.x, 1.f / axis.y, 1.f / axis.z);
    vec3f a = (p - center) * ir;
    Float h = a.Length() - 1.f;
    return h * Min(Min(axis.x, axis.y), axis.z);
}

__bidevice__ Float SDF::Capsule(vec3f p, vec3f a, vec3f b, Float r){
    vec3f pa = p - a, ba = b - a;
    Float h  = Clamp(Dot(pa, ba) / Dot(ba, ba), 0.0f, 1.0f);
    return (pa - ba * h).Length() - r;
}

__bidevice__ Float SDF::Rhombus(vec3f p, vec2f axis, Float height, Float corner){
    vec2f b(axis);
    Float ra = corner;
    Float h = height;
    p = Abs(p);
    
    Float f = Clamp((NegDot(b, b - 2.0 * vec2f(p.x, p.z))) / Dot(b, b), -1.0, 1.0);
    vec2f t = vec2f(p.x, p.z) - 0.5 * b * vec2f(1.0 - f, 1.0 + f);
    vec2f q = vec2f(t.Length() * Sign(p.x * b.y + p.z * b.x - b.x * b.y) - ra, p.y - h);
    return Min(Max(q.x, q.y), 0.0) + (Max(q, vec2f(0.0))).Length();
}

__bidevice__ Float SDF::Triangle(vec3f p, vec3f a, vec3f b, vec3f c){
    vec3f ba = b - a; vec3f pa = p - a;
    vec3f cb = c - b; vec3f pb = p - b;
    vec3f ac = a - c; vec3f pc = p - c;
    vec3f nor = Cross(ba, ac);
    
    Float sa = Sign(Dot(Cross(ba, nor), pa));
    Float sb = Sign(Dot(Cross(cb, nor), pb));
    Float sc = Sign(Dot(Cross(ac, nor), pc));
    Float ss = sa + sb + sc;
    if(ss < 2){
        Float a2 = Dot2(ba * Clamp(Dot(ba, pa) / Dot2(ba), 0.0, 1.0) - pa);
        Float b2 = Dot2(cb * Clamp(Dot(cb, pb) / Dot2(cb), 0.0, 1.0) - pb);
        Float c2 = Dot2(ac * Clamp(Dot(ac, pc) / Dot2(ac), 0.0, 1.0) - pc);
        return sqrt(Min(a2, Min(b2, c2)));
    }else{
        return sqrt(Dot(nor,pa) * Dot(nor, pa) / Dot2(nor));
    }
}

__bidevice__ vec2f OpRepeatLimited(const vec2f &p, const Float &s, const vec2f &lim){
    AssertA(!IsZero(s), "Zero repeate interval given");
    Float invS = 1 / s;
    return p - s * Clamp(Round(p * invS), -lim, lim);
}


__bidevice__ Float ProceduralShape::MapP(Point3f p) const{
    Point2f uv;
    int id;
    return Map(p, &uv, &id);
}

__bidevice__ bool ProceduralShape::RefineHit(Point3f *hit, Point2f *uv, 
                                             int *shapeId, Normal3f *n) const
{
    (void)uv; (void)n; (void)shapeId;
    *uv = Point2f(0, 0);
    return false;
}

/*
* NOTE: Must be very carefull when computing normals as this length can easily become zero.
*       perhaps there is a better way to calculate normal vector but I'm guessing this
*       relies on how far things are so currently our best guess is to simply iterate
*       until its lenght is no longer zero.
*/
__bidevice__ Normal3f ProceduralShape::DerivativeNormal(Point3f p, Float t) const{
    (void)t;
    bool found = false;
    float scale = 0.001;
    Normal3f n;
    int it = 0;
    while(!found){
        Float e = scale * ToVec3(p).Length();
        Float x = MapP(Point3f(p.x + e, p.y, p.z)) - MapP(Point3f(p.x - e, p.y, p.z));
        Float y = MapP(Point3f(p.x, p.y + e, p.z)) - MapP(Point3f(p.x, p.y - e, p.z));
        Float z = MapP(Point3f(p.x, p.y, p.z + e)) - MapP(Point3f(p.x, p.y, p.z - e));
        n = Normal3f(x, y, z);
        found = !IsZero(n.Length());
        if(!found){
            if(it++ > WARN_NORMAL){
                printf("Warning: Unstable mapping, normal length is zero with scale (%g)\n", 
                       scale);
                n = Normal3f(0,1,0);
                break;
            }
            
            scale *= 2;
        }
    }
    
    return Normalize(n);
}

__bidevice__ bool ProceduralShape::Intersect(const Ray &r, Float *tHit,
                                             SurfaceInteraction *isect) const
{
    Point3f pHit;
    vec3f oErr, dErr;
    bool hit = false, fromInside = false;
    Float tMax;
    Float t = 0;
    
    Point2f uv;
    int id;
    
    Ray ur(r.o, Normalize(r.d));
    Bounds3f localBound = GetLocalBounds();
    Ray ray = WorldToObject(ur, &oErr, &dErr);
    
    ur.tMax = r.tMax * r.d.Length();
    tMax = ur.tMax;
    
    fromInside = Inside(ray.o, localBound);
    if(!fromInside){
        Float t1, t2;
        if(!localBound.IntersectP(ray, &t1, &t2)) return false;
        
        t = Min(t1, t2);
    }
    
    for(int i = 0; i < MAX_STEPS && t < tMax; i++){
        Point3f p = ray(t);
        
        Float d = Map(p, &uv, &id);
        if(Absf(d) < (0.001*t)){
            hit = true;
            break;
        }
        
        t += Absf(d);
        
        if(!Inside(ray(t), localBound)) break;
    }
    
    if(hit){
        vec3f dpdu, dpdv;
        Normal3f n;
        Normal3f dndu(0), dndv(0);
        pHit = ray(t);
        if(!RefineHit(&pHit, &uv, &id, &n)){
            n = DerivativeNormal(pHit, t);
        }else if(fromInside){
            //n = -n;
        }
        
        //TODO: Compute dndu, dndv?
        
        CoordinateSystem(ToVec3(n), &dpdu, &dpdv);
        vec3f pError = gamma(5) * Abs((vec3f)pHit);
        *isect = ObjectToWorld(SurfaceInteraction(pHit, pError, uv,
                                                  -ray.d, dpdu, dpdv, dndu, dndv,
                                                  ray.time, this, id));
        *tHit = t;
    }
    
    return hit;
}

//////////////////////////////////////
// PROCEDURAL SPHERE
/////////////////////////////////////
__bidevice__ ProceduralSphere::ProceduralSphere(const Transform &toWorld, Float radius)
: ProceduralShape(toWorld), radius(radius){type = SPHERE_PROCEDURAL;}

__bidevice__ Float ProceduralSphere::Map(Point3f p, Point2f *uv, int *shapeId) const{
    (void)uv; (void)shapeId;
    return SDF::Sphere(ToVec3(p), radius);
}

__bidevice__ bool ProceduralSphere::RefineHit(Point3f *pHit, Point2f *uv, 
                                              int *shapeId, Normal3f *n) const
{
    *shapeId = 0;
    *pHit *= radius / Distance(*pHit, Point3f(0,0,0));
    Float phi = std::atan2(pHit->y, pHit->x);
    if(phi < 0) phi += 2 * Pi;
    Float u = phi / (2 * Pi);
    Float theta = std::acos(Clamp(pHit->z / radius, -1, 1));
    Float v = (Pi - theta) / Pi;
    *uv = Point2f(u, v);
    *n = Normal3f(pHit->x, pHit->y, pHit->z) / radius;
    return true;
}

__bidevice__ Bounds3f ProceduralSphere::GetLocalBounds() const{
    return Bounds3f(Point3f(-radius, -radius, -radius),
                    Point3f(radius, radius, radius));
}

__bidevice__ Bounds3f ProceduralSphere::GetBounds() const{
    return ObjectToWorld(Bounds3f(Point3f(-radius, -radius, -radius),
                                  Point3f(radius, radius, radius)));
}

//////////////////////////////////////
// PROCEDURAL BOX
/////////////////////////////////////
__bidevice__ ProceduralBox::ProceduralBox(const Transform &toWorld, const vec3f &length)
: ProceduralShape(toWorld), length(length){type = BOX_PROCEDURAL;}

__bidevice__ Float ProceduralBox::Map(Point3f p, Point2f *uv, int *shapeId) const{
    (void)uv; (void)shapeId;
    return SDF::Box(ToVec3(p), length * 0.5);
}

__bidevice__ bool ProceduralBox::RefineHit(Point3f *pHit, Point2f *uv, 
                                           int *shapeId, Normal3f *n) const
{
    vec3f half = length * 0.5;
    vec3f chosen(Infinity, -1, -1);
    vec3f quasi[6] = {
        vec3f(Absf(pHit->x - half.x), 1, +half.x), // id 1 +X
        vec3f(Absf(pHit->x + half.x), 3, -half.x), // id 3 -X
        vec3f(Absf(pHit->y - half.y), 4, +half.y), // id 4 +Y
        vec3f(Absf(pHit->y + half.y), 5, -half.y), // id 5 -Y
        vec3f(Absf(pHit->z - half.z), 2, +half.z), // id 2 +Z
        vec3f(Absf(pHit->z + half.z), 0, -half.z), // id 0 -Z
    };
    
    for(int i = 0; i < 6; i++){
        if(chosen.x > quasi[i].x){
            chosen = quasi[i];
        }
    }
    
    AssertA(chosen.y > -1, "Could not find refine axis");
    int target = (int)chosen.y;
    Float u = 0, v = 0;
    switch(target){
        case 1:
        case 3:{
            pHit->x = chosen.z;
            *n = chosen.z > 0 ? Normal3f(1, 0, 0) : Normal3f(-1, 0, 0);
            u = (pHit->z + half.z) / length.z;
            v = 1.f - (pHit->y + half.y) / length.y;
        } break;
        case 4:
        case 5:{
            pHit->y = chosen.z;
            *n = chosen.z > 0 ? Normal3f(0, 1, 0) : Normal3f(0, -1, 0);
            u = (pHit->x + half.x) / length.x;
            v = (pHit->z + half.z) / length.z;
        } break;
        case 0:
        case 2:{
            pHit->z = chosen.z;
            *n = chosen.z > 0 ? Normal3f(0, 0, 1) : Normal3f(0, 0, -1);
            u = (pHit->x + half.x) / length.x;
            v = 1.f - (pHit->y + half.y) / length.z;
        } break;
        
        default:{
            AssertA(0, "Unknown refine axis");
        }
    }
    
    *uv = Point2f(u, v);
    return true;
}

__bidevice__ Bounds3f ProceduralBox::GetLocalBounds() const{
    vec3f half = length * 0.5;
    return Bounds3f(Point3f(-half.x, -half.y, -half.z),
                    Point3f(half.x, half.y, half.z));
}

__bidevice__ Bounds3f ProceduralBox::GetBounds() const{
    vec3f half = length * 0.5;
    return ObjectToWorld(Bounds3f(Point3f(-half.x, -half.y, -half.z),
                                  Point3f(half.x, half.y, half.z)));
}

//////////////////////////////////////
// PROCEDURAL TOYS
/////////////////////////////////////
__bidevice__ ProceduralComponent::ProceduralComponent(const Transform &toWorld, 
                                                      const Bounds3f &bounds, int id): 
ProceduralShape(toWorld), id(id), bounds(bounds){ type = COMPONENT_PROCEDURAL; }

__bidevice__ Bounds3f ProceduralComponent::GetBounds() const{
    return ObjectToWorld(bounds);
}
__bidevice__ Bounds3f ProceduralComponent::GetLocalBounds() const{
    return bounds;
}

__bidevice__ Float ProceduralComponent::Map(Point3f p, Point2f *uv, int *shapeId) const{
    vec3f length(bounds.ExtentOn(0), bounds.ExtentOn(1), bounds.ExtentOn(2));
    vec3f half = length * 0.5;
    switch(id){
        case 0: return SDF::OrigamiBoat(ToVec3(p), half, shapeId);
        case 1: return SDF::OrigamiDragon(ToVec3(p), half, shapeId);
        case 2: return SDF::OrigamiWhale(ToVec3(p), half, shapeId);
        case 3: return SDF::OrigamiBird(ToVec3(p), half, shapeId);
        default:{
            printf("Unknown component id\n");
        }
    }
    
    return false;
}

////////////////////// MOVE THIS TO A SPECIFIC FILE
__bidevice__ Float SDF::OrigamiBird(vec3f p, vec3f half, int *shapeId){
    vec3f q(p.x, p.y, Absf(p.z));
    vec3f Q0(-half.x * 0.25,  half.y * 0.88, half.z * 0.30);
    vec3f Q1(+half.x * 0.05,  half.y * 0.50, half.z * 0.13);
    vec3f Q2(-half.x * 0.15,  half.y * 0.05, half.z * 0.13);
    vec3f Q3(+half.x * 0.60, -half.y * 0.10, half.z * 0.08);
    vec3f Q4(+half.x * 0.50,  half.y * 0.02, half.z * 0.03);
    vec3f Q5(+half.x * 0.75,  half.y * 0.20, half.z * 0.00);
    vec3f Q6(+half.x * 0.85,  half.y * 0.00, half.z * 0.00);
    vec3f Q7(-half.x * 0.20, -half.y * 0.50, half.z * 0.03);
    vec3f Q8(-half.x * 0.85, -half.y * 0.90, half.z * 0.00);
    vec3f Q9(-half.x * 0.25, -half.y * 0.80, half.z * 0.00);
    
    vec2f tris[] = {
        vec2f(SDF::Triangle(q, Q0, Q1, Q2), 0),
        vec2f(SDF::Triangle(q, Q1, Q2, Q3), 1),
        vec2f(SDF::Triangle(q, Q3, Q4, Q5), 2),
        vec2f(SDF::Triangle(q, Q6, Q5, Q3), 1),
        vec2f(SDF::Triangle(q, Q2, Q3, Q7), 1),
        vec2f(SDF::Triangle(q, Q2, Q8, Q9), 2)
    };
    
    vec2f sdf(Infinity, -1);
    for(int i = 0; i < 6; i++){
        if(sdf.x > tris[i].x){
            sdf = tris[i];
        }
    }
    
    AssertA(sdf.y > -1, "Invalid triangle mapping");
    *shapeId = sdf.y;
    return sdf.x;
}

__bidevice__ Float SDF::OrigamiBoat(vec3f p, vec3f half, int *shapeId){
    vec3f q = vec3f(Absf(p.x), p.y, Absf(p.z));
    
    vec3f Poo(0.f, -half.y * 0.9, 0.f);
    vec3f P0(half.x*0.1, -half.y * 0.9, half.z * 0.6);
    vec3f P1(half.x*0.1, -half.y * 0.9, half.z * 0.01 + 0.03f);
    vec3f P2(0.01f, half.y * 0.95, 0.01f  + 0.03f);
    vec3f Pk(half.x * 0.95, 0.f, half.z * 0.01 + 0.03f);
    vec3f Pn(0.01f, half.y * 0.1, half.z * 0.9);
    
    Float tri1 = SDF::Triangle(q, P0, P1, P2);
    Float tri2 = SDF::Triangle(q, P0, Pk, Poo);
    Float tri3 = SDF::Triangle(q, P0, Pk, Pn);
    vec2f tris[3] = { vec2f(tri1, 0), vec2f(tri2, 1), vec2f(tri3, 2) };
    vec2f sdf(Infinity, -1);
    for(int i = 0; i < 3; i++){
        if(sdf.x > tris[i].x){
            sdf = tris[i];
        }
    }
    
    AssertA(sdf.y > -1, "Invalid triangle mapping");
    *shapeId = sdf.y;
    return sdf.x;
}

__bidevice__ Float SDF::OrigamiWhale(vec3f p, vec3f half, int *shapeId){
    vec3f q(p.x, p.y, Absf(p.z));
    
    vec3f Q0 (-half.x * 0.15,  half.y * 0.99, half.z * 0.20);
    vec3f Q1 (-half.x * 0.65, -half.y * 0.05, half.z * 0.80);
    vec3f Q2 (-half.x * 0.10, -half.y * 0.80, half.z * 0.99);
    
    vec3f Q3 (-half.x * 0.98,  half.y * 0.82, half.z * 0.40);
    vec3f Q4 (+half.x * 0.98, -half.y * 0.15, half.z * 0.03);
    vec3f Q5 (+half.x * 0.90, -half.y * 0.80, half.z * 0.70);
    vec3f Q6 (-half.x * 0.98,  half.y * 0.15, half.z * 0.50);
    vec3f Q7 (+half.x * 0.85, -half.y * 0.99, half.z * 0.80);
    vec3f Q8 (-half.x * 0.80, -half.y * 0.99, half.z * 0.80);
    vec3f Q9 (-half.x * 0.96, -half.y * 0.76, half.z * 0.70);
    vec3f Q10(+half.x * 0.95,  half.y * 0.85, half.z * 0.00);
    vec3f Q11(+half.x * 0.65,  half.y * 0.10, half.z * 0.00);
    vec3f Q12(+half.x * 0.75,  half.y * 0.05, half.z * 0.03);
    
    vec2f tris[] = {
        vec2f(SDF::Triangle(q, Q0, Q1, Q2), 0),
        vec2f(SDF::Triangle(q, Q0, Q3, Q4), 1),
        vec2f(SDF::Triangle(q, Q3, Q4, Q5), 1),
        vec2f(SDF::Triangle(q, Q3, Q5, Q6), 1),
        vec2f(SDF::Triangle(q, Q7, Q8, Q9), 2),
        vec2f(SDF::Triangle(q, Q7, Q6, Q9), 2),
        vec2f(SDF::Triangle(q, Q7, Q6, Q5), 2),
        vec2f(SDF::Triangle(q, Q4, Q12, Q10), 1),
        vec2f(SDF::Triangle(q, Q12, Q10, Q11), 0)
    };
    
    vec2f sdf(Infinity, -1);
    for(int i = 0; i < 9; i++){
        if(sdf.x > tris[i].x){
            sdf = tris[i];
        }
    }
    
    AssertA(sdf.y > -1, "Invalid mapping");
    
    *shapeId = sdf.y;
    return sdf.x;
}

__bidevice__ Float SDF::OrigamiDragon(vec3f p, vec3f half, int *shapeId){
    vec3f q(p.x, p.y, Absf(p.z));
    vec3f P0   (-half.x * 0.70,  half.y * 0.85 ,half.z * 0.05);
    vec3f P1   (-half.x * 0.25,  half.y * 0.40, half.z * 0.06);
    vec3f P2   (-half.x * 0.55, -half.y * 0.15, half.z * 0.05);
    vec3f P3   (+half.x * 0.10, -half.y * 0.60, half.z * 0.09);
    vec3f P4   (+half.x * 0.20,  half.y * 0.15, half.z * 0.08);
    vec3f P5Zp (+half.x * 0.05,  half.y * 0.25, half.z * 0.00);
    vec3f P6Zp (+half.x * 0.35,  half.y * 0.10, half.z * 0.02);
    vec3f P7Zp (+half.x * 0.45, -half.y * 0.60, half.z * 0.05);
    vec3f P8Zp (-half.x * 0.35, -half.y * 0.60, half.z * 0.05);
    vec3f P2Sh (-half.x * 0.55, -half.y * 0.15, half.z * 0.05);
    vec3f P9pp (+half.x * 0.40,  half.y * 0.20, half.z * 0.00);
    vec3f P10pp(+half.x * 0.70, -half.y * 0.75, half.z * 0.10);
    vec3f P11pp(+half.x * 0.43, -half.y * 0.96, half.z * 0.15);
    vec3f P12m (-half.x * 0.65, -half.y * 0.96, half.z * 0.15);
    vec3f P13m (-half.x * 0.45, -half.y * 0.05, half.z * 0.05);
    vec3f P14p (+half.x * 0.80,  half.y * 0.10, half.z * 0.01);
    vec3f P15p (+half.x * 0.50, -half.y * 0.60, half.z * 0.01);
    vec3f Q3   (+half.x * 0.60,  half.y * 0.10, half.z * 0.01);
    vec3f Q2   (+half.x * 0.55,  half.y * 0.85, half.z * 0.03);
    vec3f Q0   (+half.x * 0.70,  half.y * 0.90, half.z * 0.03);
    vec3f Q6   (+half.x * 0.71,  half.y * 0.70, half.z * 0.10);
    vec3f Q4   (+half.x * 0.90,  half.y * 0.90, half.z * 0.00);
    vec3f Q5   (+half.x * 0.88,  half.y * 0.80, half.z * 0.00);
    vec3f Q7   (+half.x * 0.65,  half.y * 1.00, half.z * 0.00);
    vec3f Q8   (+half.x * 0.45,  half.y * 1.00, half.z * 0.00);
    vec3f Q9   (+half.x * 0.64,  half.y * 0.88, half.z * 0.03);
    vec3f T0   (-half.x * 0.85, -half.y * 0.25, half.z * 0.03);
    vec3f T1   (-half.x * 0.80, -half.y * 0.50, half.z * 0.03);
    vec3f T2   (-half.x * 0.83,  half.y * 0.16, half.z * 0.01);
    vec3f T3   (-half.x * 1.00,  half.y * 0.19, half.z * 0.01);
    vec3f T4   (-half.x * 0.53, -half.y * 0.23, half.z * 0.01);
    vec3f T5   (-half.x * 0.90,  half.y * 0.75, half.z * 0.00);
    
    Float tri1  = SDF::Triangle(q, P0, P1, P2);
    Float tri2  = SDF::Triangle(q, P1, P2, P3);
    Float tri3  = SDF::Triangle(q, P1, P3, P4);
    Float tri4  = SDF::Triangle(q, P5Zp, P6Zp, P7Zp);
    Float tri5  = SDF::Triangle(q, P5Zp, P7Zp, P8Zp);
    Float tri6  = SDF::Triangle(q, P5Zp, P8Zp, P2Sh);
    Float tri7  = SDF::Triangle(q, P9pp, P6Zp, P10pp);
    Float tri8  = SDF::Triangle(q, P6Zp, P11pp, P10pp);
    Float tri9  = SDF::Triangle(q, P8Zp, P12m, P13m);
    Float tri10 = SDF::Triangle(q, P15p, P14p, P6Zp);
    Float tri11 = SDF::Triangle(q, Q3, Q2, P14p);
    Float tri12 = SDF::Triangle(q, Q0, P14p, Q2);
    Float tri13 = SDF::Triangle(q, Q6, Q5, Q0);
    Float tri14 = SDF::Triangle(q, Q5, Q4, Q0);
    Float tri15 = SDF::Triangle(q, Q4, Q7, Q9);
    Float tri16 = SDF::Triangle(q, Q7, Q9, Q8);
    Float tri17 = SDF::Triangle(q, P9pp, T0, T1);
    Float tri18 = SDF::Triangle(q, T1, P9pp, P7Zp);
    Float tri19 = SDF::Triangle(q, T0, T2, T3);
    Float tri20 = SDF::Triangle(q, T0, T2, T4);
    Float tri21 = SDF::Triangle(q, T3, T2, T5);
    
    Float sdf = Min(Min(tri1, tri2), tri3);
    sdf = Min(Min(sdf, tri4), tri5);
    sdf = Min(Min(sdf, tri6), tri7);
    sdf = Min(Min(sdf, tri8), tri9);
    sdf = Min(Min(sdf, tri10), tri11);
    sdf = Min(Min(sdf, tri12), tri13);
    sdf = Min(Min(sdf, tri14), tri15);
    sdf = Min(Min(sdf, tri16), tri17);
    sdf = Min(Min(sdf, tri18), tri19);
    sdf = Min(Min(sdf, tri20), tri21);
    *shapeId = 5;
    return sdf;
}
