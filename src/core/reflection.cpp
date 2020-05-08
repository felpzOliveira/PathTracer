#include <reflection.h>

__bidevice__ Float LambertianReflection::Pdf(const vec3f &wo, const vec3f &wi) const{
    return SameHemisphere(wo, wi) ? AbsCosTheta(wi) * InvPi : 0;
}

__bidevice__ Spectrum LambertianReflection::Sample_f(const vec3f &woW, vec3f *wiW, 
                                                     const Point2f &u,Float *pdf, 
                                                     BxDFType *sampledType) const 
{
    vec3f wo = WorldToLocal(woW);
    vec3f wi;
    wi = CosineSampleHemisphere(u);
    if (wo.z < 0) wi.z *= -1;
    *pdf = Pdf(wo, wi);
    Spectrum e = f(wo, wi);
    
    *wiW = LocalToWorld(wi);
    return e;
}

__bidevice__ Spectrum LambertianReflection::f(const vec3f &wo, const vec3f &wi) const{
    return R * InvPi;
}
