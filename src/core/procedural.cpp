#include <procedural.h>

#define MAX_STEPS 256
#define WARN_NORMAL 10

__bidevice__ Float NegDot(const vec2f &a, const vec2f &b){
    return (a.x * b.x - a.y * b.y);
}

__bidevice__ Float Sign(Float x){
    if(IsZero(x)) return 0;
    if(x > 0) return 1;
    return -1;
}

__bidevice__ Float SDF::Sphere(vec3f p, Float radius){
    return p.Length() - radius;
}

__bidevice__ Float SDF::Box(vec3f p, vec3f boxLength){
    vec3f d = Abs(p) - boxLength;
    return Min(Max(d.x,Max(d.y,d.z)),0.0) + (Max(d,vec3f(0.0)).Length());
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

__bidevice__ vec2f OpRepeatLimited(const vec2f &p, const Float &s, const vec2f &lim){
    AssertA(!IsZero(s), "Zero repeate interval given");
    Float invS = 1 / s;
    return p - s * Clamp(Round(p * invS), -lim, lim);
}


/*
* NOTE: Must be very carefull when computing normals as this length can easily become zero.
*       perhaps there is a better way to calculate normal vector but I'm guessing this
*       relies on how far things are so currently our best guess is to simply iterate
*       until its lenght is no longer zero.
*/
__bidevice__ Normal3f ProceduralShape::ComputeNormal(Point3f p, Float t) const{
    (void)t;
    bool found = false;
    float scale = 0.001;
    Normal3f n;
    int it = 0;
    while(!found){
        Float e = scale * ToVec3(p).Length();
        Float x = Map(Point3f(p.x + e, p.y, p.z)) - Map(Point3f(p.x - e, p.y, p.z));
        Float y = Map(Point3f(p.x, p.y + e, p.z)) - Map(Point3f(p.x, p.y - e, p.z));
        Float z = Map(Point3f(p.x, p.y, p.z + e)) - Map(Point3f(p.x, p.y, p.z - e));
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
    bool hit = false;
    Float tMax;
    Float t = 0;
    
    Ray ur(r.o, Normalize(r.d));
    Bounds3f localBound = GetLocalBounds();
    Ray ray = WorldToObject(ur, &oErr, &dErr);
    
    ur.tMax = r.tMax * r.d.Length();
    tMax = ur.tMax;
    
    if(!Inside(ray.o, localBound)){
        Float t1, t2;
        if(!localBound.IntersectP(ray, &t1, &t2)) return false;
        
        t = Min(t1, t2);
    }
    
    for(int i = 0; i < MAX_STEPS && t < tMax; i++){
        Point3f p = ray(t);
        
        Float d = Map(p);
        
        if(Absf(d) < (0.001*t)){
            hit = true;
            break;
        }
        
        t += Absf(d);
        
        if(!Inside(ray(t), localBound)) break;
    }
    
    if(hit){
        Float u = 0, v = 0; //TODO: How do we define this
        pHit = ray(t);
        
        Normal3f n = ComputeNormal(pHit, t);
        vec3f dpdu, dpdv;
        Normal3f dndu(0), dndv(0);
        CoordinateSystem(ToVec3(n), &dpdu, &dpdv);
        
        vec3f pError = gamma(5) * Abs((vec3f)pHit);
        *isect = ObjectToWorld(SurfaceInteraction(pHit, pError, Point2f(u, v),
                                                  -ray.d, dpdu, dpdv, dndu, dndv,
                                                  ray.time, this));
        *tHit = t;
    }
    
    return hit;
}

__bidevice__ ProceduralSphere::ProceduralSphere(const Transform &toWorld, Float radius)
: ProceduralShape(toWorld), radius(radius){type = SPHERE_PROCEDURAL;}

__bidevice__ Float ProceduralSphere::Map(Point3f po) const{
    Float rRadius = 5;
    Float maxHeight = rRadius;
    vec3f p = ToVec3(po);
    vec3f q(p.x, p.y-0.1, p.z);
    
    vec2f u2 = OpRepeatLimited(vec2f(q.x, q.z), 4.0, vec2f(4.0, 2.0));
    q.x = u2.x;
    q.z = u2.y;
    
    vec2f q2(q.x, q.z);
    // columns
    Float columnHeight = 0.4 * maxHeight;
    Float boxL = 0.25 * rRadius;
    Float rad = 0.1 * rRadius; // base size
    Float d = 1e20;
    Float tileLen = 0;
    Float extension = 2 * boxL;
    
    rad -= 0.05 * q.y; // make smaller at top
    rad -= 0.1*pow(0.5 + 0.5*Sin(16.0*std::atan2(q.x, q.z)), 2.0); // oscilate the circle
    rad += 0.15*pow(0.5 + 0.5*Sin(q.y * 5.0), 0.12) - 0.15; // create the edges
    
    d = q2.Length() - rad; // circle in 2D (stretch in +-y, minimal cylinder)
    d = Max(d, 0);
    d = Max(d, q.y - columnHeight * 0.7);
    
    
    vec3f qq = vec3f(q.x, Absf(q.y + 0.45 * rRadius) - columnHeight * 1.8, q.z);
    d = Min(d, SDF::Box(qq, vec3f(boxL, 0.12 * boxL, boxL)));
    
    
    // remove center columns
    
    d = Max(d, -SDF::Box(p, vec3f(extension, 2 * radius, extension)));
    
    // floor
    tileLen = extension/6;
    u2 = OpRepeatLimited(vec2f(p.x, p.z), 0.8, vec2f(8.0, 8.0));
    q.x = u2.x; q.z = u2.y; q.y = p.y;
    d = Min(d, SDF::Box(vec3f(q.x, q.y  + 1.15 * rRadius, q.z), 
                        vec3f(tileLen, 0.1, tileLen) - vec3f(0.09)));
    
    // roof
    tileLen = extension/3;
    u2 = OpRepeatLimited(vec2f(p.x, p.z), 1.8, vec2f(3.0, 3.0));
    q.x = u2.x; q.z = u2.y; q.y = p.y;
    d = Min(d, SDF::Box(vec3f(q.x, q.y - 0.41 * rRadius, q.z), 
                        vec3f(tileLen, 0.3, tileLen) - vec3f(0.09)));
    
    d *= 0.1; // Do we need this?
    
#if 0
    vec3f q = p;
    Float sq = 0.2;
    
    Float scale = sq * rRadius;
    Float rad = rRadius + scale * Sin(2 * q.x) * Sin(2 * q.z) * Sin(2 * q.y);
    Float d = SDF::Sphere(p, rad);
    d *= sq;
#endif
    return d;
}

__bidevice__ Bounds3f ProceduralSphere::GetLocalBounds() const{
    return Bounds3f(Point3f(-radius, -radius, -radius),
                    Point3f(radius, radius, radius));
}

__bidevice__ Bounds3f ProceduralSphere::GetBounds() const{
    return ObjectToWorld(Bounds3f(Point3f(-radius, -radius, -radius),
                                  Point3f(radius, radius, radius)));
}
