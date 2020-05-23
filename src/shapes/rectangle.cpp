#include <shape.h>

__bidevice__ bool Rectangle::Intersect(const Ray &r, Float *tHit,
                                       SurfaceInteraction *isect) const
{
    Point3f pHit;
    vec3f oErr, dErr;
    Ray ray = WorldToObject(r, &oErr, &dErr);
    
    Float t = (-ray.o.z) / ray.d.z;
    if(t < 0 || t > ray.tMax) return false;
    
    Float hx = sizex/2, hy = sizey/2;
    pHit = ray(t);
    if(pHit.x < -hx || pHit.x > hx || pHit.y < -hy || pHit.y > hy) return false;
    
    Float u = Absf((pHit.x - hx)/sizex);
    Float v = Absf((pHit.y - hy)/sizey);
    //printf("UV: %g %g\n", u, v);
    vec3f dpdu(1, 0, 0);
    vec3f dpdv(0, 1, 0);
    Normal3f dndu(0), dndv(0);
    
    vec3f pError = gamma(5) * Abs((vec3f)pHit);
    *isect = ObjectToWorld(SurfaceInteraction(pHit, pError, Point2f(u, v),
                                              -ray.d, dpdu, dpdv, dndu, dndv,
                                              ray.time, this));
    *tHit = t;
    return true;
}

__bidevice__ Bounds3f Rectangle::GetBounds() const{
    Float hx = sizex/2, hy = sizey/2;
    return ObjectToWorld(Bounds3f(Point3f(-hx,-hy,-0.0001), Point3f(hx,hy,0.0001)));
}

__bidevice__ Float Rectangle::Area() const{
    return sizex * sizey;
}

__bidevice__ Float Rectangle::Pdf(const Interaction &ref, const vec3f &wi) const{
    return Shape::Pdf(ref, wi);
}

__bidevice__ Interaction Rectangle::Sample(const Point2f &u, Float *pdf) const{
    Float hx = sizex/2;
    Float hy = sizey/2;
    Point3f pObj(-hx + u[0] * sizex, -hy + u[1] * sizey, 0);
    Interaction it;
    it.n = Normalize(ObjectToWorld(Normal3f(0,0,1)));
    if(reverseOrientation) it.n *= -1;
    it.p = ObjectToWorld(pObj, vec3f(0,0,0), &it.pError);
    *pdf = 1 / Area();
    return it;
}

__bidevice__ Interaction Rectangle::Sample(const Interaction &ref, const Point2f &u,
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