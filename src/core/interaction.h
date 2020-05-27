#pragma once
#include <geometry.h>
#include <medium.h>

class Shape;
class Primitive;
class BSDF;

class Interaction{
    public:
    Point3f p;
    Float time;
    vec3f pError;
    vec3f wo;
    Normal3f n;
    MediumInterface mediumInterface;
    
    __bidevice__ Interaction(){}
    
    __bidevice__ Ray SpawnRay(const vec3f &d) const{
        Point3f o = OffsetRayOrigin(p, pError, n, d);
        return Ray(o, d, Infinity, time, GetMedium(d));
    }
    
    __bidevice__ Ray SpawnRayTo(const Interaction &it) const;
    
    __bidevice__ Interaction(const Point3f &p, const Normal3f &n, const vec3f &pError,
                             const vec3f &wo, Float time) :
    p(p), time(time), pError(pError), wo(Normalize(wo)), n(n), mediumInterface(nullptr){}
    
    __bidevice__ Interaction(const Point3f &p, const vec3f &wo, Float time,
                             const MediumInterface &mediumInterface)
        : p(p), time(time), wo(wo), mediumInterface(mediumInterface) {}
    
    __bidevice__ const Medium *GetMedium(const vec3f &w) const{
        return Dot(w, ToVec3(n)) > 0 ? mediumInterface.outside : mediumInterface.inside;
    }
    
    __bidevice__ const Medium *GetMedium() const{
        return mediumInterface.inside;
    }
    
    __bidevice__ bool IsSurfaceInteraction() const { return n != Normal3f(); }
    
    __bidevice__ bool IsMediumInteraction() const { return !IsSurfaceInteraction(); }
    
};

class SurfaceInteraction : public Interaction{
    public:
    Point2f uv;
    vec3f dpdu, dpdv;
    Normal3f dndu, dndv;
    const Shape *shape = nullptr;
    const Primitive *primitive = nullptr;
    int faceIndex;
    
    __bidevice__ SurfaceInteraction(){}
    __bidevice__ SurfaceInteraction(const Point3f &p, const vec3f &pError,
                                    const Point2f &uv, const vec3f &wo,
                                    const vec3f &dpdu, const vec3f &dpdv,
                                    const Normal3f &dndu, const Normal3f &dndv, Float time,
                                    const Shape *sh, int faceIndex = 0);
    
    __bidevice__ Spectrum Le(const vec3f &w) const;
    
    __bidevice__ void ComputeDifferentials(const RayDifferential &r) const{}
    
    __bidevice__ void ComputeScatteringFunctions(BSDF *bsdf, const RayDifferential &r, 
                                                 TransportMode mode, bool mLobes);
};

class MediumInteraction : public Interaction{
    public:
    PhaseFunction phase;
    __bidevice__ MediumInteraction(){}
    
    __bidevice__ MediumInteraction(const Point3f &p, const vec3f &wo, Float time,
                                   const Medium *medium, Float g)
        : Interaction(p, wo, time, medium) { phase.SetG(g); }
    
    __bidevice__ bool IsValid() const { return phase.is_initialized == 1; }
};