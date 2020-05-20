#include <shape.h>

__bidevice__ Bounds3f Disk::GetBounds() const{
    return ObjectToWorld(Bounds3f(Point3f(-radius, -radius, height), 
                                  Point3f(radius, radius, height)));
}

__bidevice__ bool Disk::Intersect(const Ray &r, Float *tHit, 
                                  SurfaceInteraction *isect) const
{
    vec3f oErr, dErr;
    Ray ray = WorldToObject(r, &oErr, &dErr);
    
    if(IsZero(ray.d.z)) return false;
    Float tShapeHit = (height - ray.o.z) / ray.d.z;
    if(tShapeHit <= 0 || tShapeHit >= ray.tMax) return false;
    
    Point3f pHit = ray(tShapeHit);
    Float dist2 = pHit.x * pHit.x + pHit.y * pHit.y;
    if(dist2 > radius * radius || dist2 < innerRadius * innerRadius)
        return false;
    
    Float phi = std::atan2(pHit.y, pHit.x);
    if(phi < 0) phi += 2 * Pi;
    if(phi > phiMax) return false;
    
    Float u = phi / phiMax;
    Float rHit = std::sqrt(dist2);
    Float v = (radius - rHit) / (radius - innerRadius);
    vec3f dpdu(-phiMax * pHit.y, phiMax * pHit.x, 0);
    vec3f dpdv = vec3f(pHit.x, pHit.y, 0.) * (innerRadius - radius) / rHit;
    Normal3f dndu(0, 0, 0), dndv(0, 0, 0);
    
    pHit.z = height;
    
    vec3f pError(0, 0, 0);
    
    *isect = ObjectToWorld(SurfaceInteraction(pHit, pError, Point2f(u, v),
                                              -ray.d, dpdu, dpdv, dndu, dndv,
                                              ray.time, this));
    
    *tHit = (Float)tShapeHit;
    return true;
}

__bidevice__ Float Disk::Area() const{
    return phiMax * 0.5 * (radius * radius - innerRadius * innerRadius);
}

__bidevice__ Interaction Disk::Sample(const Point2f &u, Float *pdf) const{
    Point2f pd = ConcentricSampleDisk(u);
    Point3f pObj(pd.x * radius, pd.y * radius, height);
    Interaction it;
    it.n = Normalize(ObjectToWorld(Normal3f(0, 0, 1)));
    if(reverseOrientation) it.n *= -1;
    it.p = ObjectToWorld(pObj, vec3f(0, 0, 0), &it.pError);
    *pdf = 1 / Area();
    return it;
}

__bidevice__ Interaction Disk::Sample(const Interaction &ref, const Point2f &u,
                                      Float *pdf) const
{
    Interaction intr = Sample(u, pdf);
    vec3f wi = intr.p - ref.p;
    wi = Normalize(wi);
    if(IsZero(wi.LengthSquared())){
        *pdf = 0;
    }else{
        wi = Normalize(wi);
        *pdf *= DistanceSquared(ref.p, intr.p) / AbsDot(intr.n, -wi);
    }
    
    if (std::isinf(*pdf)) *pdf = 0.f;
    return intr;
}

__bidevice__ Float Disk::Pdf(const Interaction &ref, const vec3f &wi) const{
    return Shape::Pdf(ref, wi);
}