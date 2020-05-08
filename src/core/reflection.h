#pragma once

#include <geometry.h>
#include <cutil.h>
#include <interaction.h>

enum BxDFType {
    BSDF_REFLECTION = 1 << 0,
    BSDF_TRANSMISSION = 1 << 1,
    BSDF_DIFFUSE = 1 << 2,
    BSDF_GLOSSY = 1 << 3,
    BSDF_SPECULAR = 1 << 4,
    BSDF_ALL = BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_SPECULAR | BSDF_REFLECTION |
        BSDF_TRANSMISSION,
};

class LambertianReflection{
    public:
    Spectrum R;
    const BxDFType type;
    const Normal3f ns, ng;
    const vec3f ss, ts;
    
    
    __bidevice__ LambertianReflection(const Spectrum &R, 
                                      const SurfaceInteraction &si)
        : type(BxDFType(BSDF_REFLECTION | BSDF_DIFFUSE)), R(R),
    ns(si.n), ng(si.n), ss(Normalize(si.dpdu)), ts(Cross(ToVec3(ns), ss)) {}
    
    __bidevice__ Spectrum f(const vec3f &wo, const vec3f &wi) const;
    
    __bidevice__ virtual Spectrum Sample_f(const vec3f &wo, vec3f *wi,
                                           const Point2f &sample, Float *pdf,
                                           BxDFType *sampledType = nullptr) const;
    
    __bidevice__ virtual Float Pdf(const vec3f &wo, const vec3f &wi) const;
    
    __bidevice__ Spectrum rho(const vec3f &, int, const Point2f *) const { return R; }
    __bidevice__ Spectrum rho(int, const Point2f *, const Point2f *) const { return R; }
    
    
    __bidevice__ vec3f LocalToWorld(const vec3f &v) const{
        return vec3f(ss.x * v.x + ts.x * v.y + ns.x * v.z,
                     ss.y * v.x + ts.y * v.y + ns.y * v.z,
                     ss.z * v.x + ts.z * v.y + ns.z * v.z);
    }
    
    __bidevice__ vec3f WorldToLocal(const vec3f &v) const {
        return vec3f(Dot(v, ss), Dot(v, ts), Dot(v, ToVec3(ns)));
    }
};
