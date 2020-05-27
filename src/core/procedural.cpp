#include <procedural.h>

#define MAX_STEPS 256

__bidevice__ Normal3f ProceduralShape::ComputeNormal(Point3f p, Float t) const{
    (void)t;
    Float e = 0.01 * ToVec3(p).Length();
    Float x = Map(Point3f(p.x + e, p.y, p.z)) - Map(Point3f(p.x - e, p.y, p.z));
    Float y = Map(Point3f(p.x, p.y + e, p.z)) - Map(Point3f(p.x, p.y - e, p.z));
    Float z = Map(Point3f(p.x, p.y, p.z + e)) - Map(Point3f(p.x, p.y, p.z - e));
    return Normalize(Normal3f(x, y, z));
}

__bidevice__ bool ProceduralShape::Intersect(const Ray &r, Float *tHit,
                                             SurfaceInteraction *isect) const
{
    Point3f pHit;
    vec3f oErr, dErr;
    
    Ray ur(r.o, Normalize(r.d));
    ur.tMax = r.tMax * r.d.Length();
    
    Ray ray = WorldToObject(ur, &oErr, &dErr);
    Float t = 0;
    bool hit = false;
    Float tMax = ur.tMax;
    for(int i = 0; i < MAX_STEPS && t < tMax; i++){
        Float d = Map(ray(t));
        if(Absf(d) < (0.0001*t)){
            hit = true;
            break;
        }
        
        t += Absf(d);
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

__bidevice__ Float ProceduralSphere::Map(Point3f p) const{
    vec3f pp = ToVec3(p);
    vec3f d = Abs(pp) - vec3f(radius);
    return Min(Max(d.x, Max(d.y,d.z)), 0.0) + (Max(d, vec3f(0)).Length());
    //return ToVec3(p).Length() - radius;
}

__bidevice__ Bounds3f ProceduralSphere::GetBounds() const{
    return ObjectToWorld(Bounds3f(Point3f(-radius, -radius, -radius),
                                  Point3f(radius, radius, radius)));
}
