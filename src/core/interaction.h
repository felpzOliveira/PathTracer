#pragma once
#include <geometry.h>

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
    
    __bidevice__ Interaction(){}
    __bidevice__ Interaction(const Point3f &p, const Normal3f &n, const vec3f &pError,
                             const vec3f &wo, Float time) :
    p(p), time(time), pError(pError), wo(Normalize(wo)), n(n){}
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
    
    __bidevice__ void ComputeDifferentials(const RayDifferential &r) const{}
    
    __bidevice__ void ComputeScatteringFunctions(BSDF *bsdf, const RayDifferential &r, 
                                                 TransportMode mode, bool mLobes);
};