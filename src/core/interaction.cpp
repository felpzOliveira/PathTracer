#include <interaction.h>
#include <primitive.h>
#include <light.h>

__bidevice__ 
SurfaceInteraction::SurfaceInteraction(const Point3f &p, const vec3f &pError,
                                       const Point2f &uv, const vec3f &wo,
                                       const vec3f &dpdu, const vec3f &dpdv,
                                       const Normal3f &dndu, const Normal3f &dndv, 
                                       Float time, const Shape *sh, int faceIndex)
: Interaction(p, Normal3f(Normalize(Cross(dpdu, dpdv))), pError, wo, time),
uv(uv), dpdu(dpdu), dpdv(dpdv), dndu(dndu), dndv(dndv), 
shape(shape), faceIndex(faceIndex){}

__bidevice__ void SurfaceInteraction::ComputeScatteringFunctions(BSDF *bsdf, 
                                                                 const RayDifferential &r, 
                                                                 TransportMode mode, 
                                                                 bool mLobes)
{
    ComputeDifferentials(r);
    primitive->ComputeScatteringFunctions(bsdf, this, mode, mLobes);
}

__bidevice__ Spectrum SurfaceInteraction::Le(const vec3f &w) const{
    DiffuseAreaLight *light = primitive->GetLight();
    if(light){
        Spectrum s = light->L(*this, w);
        if(s.IsBlack()){
            //printf("Black Spectrum\n");
        }
    }
    
    return light ? light->L(*this, w) : Spectrum(0.f);
}