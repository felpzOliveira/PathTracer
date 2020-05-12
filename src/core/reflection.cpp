#include <reflection.h>

__bidevice__ Float FrDieletric(Float cosThetaI, Float etaI, Float etaT);

__bidevice__ Spectrum FrConductor(Float cosThetaI, const Spectrum &etai,
                                  const Spectrum &etat, const Spectrum &k);

__bidevice__ Float BxDF::LambertianReflection_Pdf(const vec3f &wo, const vec3f &wi) const{
    return SameHemisphere(wo, wi) ? AbsCosTheta(wi) * InvPi : 0;
}

__bidevice__ Float BxDF::Pdf(const vec3f &wo, const vec3f &wi) const{
    switch(impl){
        case BxDFImpl::LambertianReflection:{
            return LambertianReflection_Pdf(wo, wi);
        } break;
        
        case BxDFImpl::SpecularReflection:{
            return SpecularReflection_Pdf(wo, wi);
        } break;
        
        case BxDFImpl::SpecularTransmission:{
            return SpecularTransmission_Pdf(wo, wi);
        } break;
        
        case BxDFImpl::FresnelSpecular:{
            return FresnelSpecular_Pdf(wo, wi);
        } break;
        
        case BxDFImpl::OrenNayar:{
            return OrenNayar_Pdf(wo, wi);
        } break;
        
        default:{
            return 0;
        }
    }
}

__bidevice__ Spectrum BxDF::LambertianReflection_f(const vec3f &wo, const vec3f &wi) const{
    return S * InvPi;
}

__bidevice__ Spectrum BxDF::OrenNayar_f(const vec3f &wo, const vec3f &wi) const{
    Float sinThetaI = SinTheta(wi);
    Float sinThetaO = SinTheta(wo);
    // Compute cosine term of Oren-Nayar model
    Float maxCos = 0;
    if(sinThetaI > 1e-4 && sinThetaO > 1e-4){
        Float sinPhiI = SinPhi(wi), cosPhiI = CosPhi(wi);
        Float sinPhiO = SinPhi(wo), cosPhiO = CosPhi(wo);
        Float dCos = cosPhiI * cosPhiO + sinPhiI * sinPhiO;
        maxCos = Max((Float)0, dCos);
    }
    
    // Compute sine and tangent terms of Oren-Nayar model
    Float sinAlpha, tanBeta;
    if(AbsCosTheta(wi) > AbsCosTheta(wo)){
        sinAlpha = sinThetaO;
        tanBeta = sinThetaI / AbsCosTheta(wi);
    }else{
        sinAlpha = sinThetaI;
        tanBeta = sinThetaO / AbsCosTheta(wo);
    }
    
    return S * InvPi * (A + B * maxCos * sinAlpha * tanBeta);
}

__bidevice__ Spectrum BxDF::f(const vec3f &wo, const vec3f &wi) const{
    switch(impl){
        case BxDFImpl::LambertianReflection:{
            return LambertianReflection_f(wo, wi);
        } break;
        
        case BxDFImpl::SpecularReflection:{
            return SpecularReflection_f(wo, wi);
        } break;
        
        case BxDFImpl::SpecularTransmission:{
            return SpecularTransmission_f(wo, wi);
        } break;
        
        case BxDFImpl::FresnelSpecular:{
            return FresnelSpecular_f(wo, wi);
        } break;
        
        case BxDFImpl::OrenNayar:{
            return OrenNayar_f(wo, wi);
        } break;
        
        default:{
            return Spectrum(0.f);
        }
    }
}


__bidevice__ Spectrum BxDF::LambertianReflection_Sample_f(const vec3f &wo, vec3f *wi, 
                                                          const Point2f &u,Float *pdf, 
                                                          BxDFType *sampledType) const 
{
    *wi = CosineSampleHemisphere(u);
    if (wo.z < 0) wi->z *= -1;
    *pdf = Pdf(wo, *wi);
    return f(wo, *wi);
}

__bidevice__ Spectrum BxDF::SpecularReflection_Sample_f(const vec3f &wo, vec3f *wi, 
                                                        const Point2f &u,Float *pdf, 
                                                        BxDFType *sampledType) const
{
    *wi = vec3f(-wo.x, -wo.y, wo.z);
    *pdf = 1;
    return fresnel.Evaluate(CosTheta(*wi)) * S / AbsCosTheta(*wi);
}

__bidevice__ Spectrum BxDF::SpecularTransmission_Sample_f(const vec3f &wo, vec3f *wi,
                                                          const Point2f &u, Float *pdf,
                                                          BxDFType *sampledType) const
{
    bool entering = CosTheta(wo) > 0;
    Float etaI = entering ? A : B;
    Float etaT = entering ? B : A;
    
    if(!Refract(wo, Faceforward(Normal3f(0, 0, 1), wo), etaI / etaT, wi))
        return 0;
    *pdf = 1;
    Spectrum ft = S * (Spectrum(1.) - fresnel.Evaluate(CosTheta(*wi)));
    
    if(mode == TransportMode::Radiance ) ft *= (etaI * etaI) / (etaT * etaT);
    return ft / AbsCosTheta(*wi);
}

__bidevice__ Spectrum BxDF::FresnelSpecular_Sample_f(const vec3f &wo, vec3f *wi,
                                                     const Point2f &u, Float *pdf,
                                                     BxDFType *sampledType) const
{
    Float F = FrDieletric(CosTheta(wo), A, B);
    if(u[0] < F) {
        *wi = vec3f(-wo.x, -wo.y, wo.z);
        if(sampledType)
            *sampledType = BxDFType(BSDF_SPECULAR | BSDF_REFLECTION);
        *pdf = F;
        
        return F * S / AbsCosTheta(*wi);
    }else{
        bool entering = CosTheta(wo) > 0;
        Float etaI = entering ? A : B;
        Float etaT = entering ? B : A;
        
        if(!Refract(wo, Faceforward(Normal3f(0, 0, 1), wo), etaI / etaT, wi)) return 0;
        Spectrum ft = T * (1 - F);
        
        if(mode == TransportMode::Radiance) ft *= (etaI * etaI) / (etaT * etaT);
        if(sampledType)
            *sampledType = BxDFType(BSDF_SPECULAR | BSDF_TRANSMISSION);
        
        *pdf = 1 - F;
        return ft / AbsCosTheta(*wi);
    }
}

__bidevice__ Spectrum BxDF::Sample_f(const vec3f &wo, vec3f *wi,
                                     const Point2f &sample, Float *pdf,
                                     BxDFType *sampledType) const
{
    switch(impl){
        case BxDFImpl::LambertianReflection:{
            return LambertianReflection_Sample_f(wo, wi, sample, pdf, sampledType);
        } break;
        
        case BxDFImpl::SpecularReflection:{
            return SpecularReflection_Sample_f(wo, wi, sample, pdf, sampledType);
        } break;
        
        case BxDFImpl::SpecularTransmission:{
            return SpecularTransmission_Sample_f(wo, wi, sample, pdf, sampledType);
        } break;
        
        case BxDFImpl::FresnelSpecular:{
            return FresnelSpecular_Sample_f(wo, wi, sample, pdf, sampledType);
        } break;
        
        case BxDFImpl::OrenNayar:{
            return OrenNayar_Sample_f(wo, wi, sample, pdf, sampledType);
        } break;
        
        default:{
            *pdf = 0;
            return Spectrum(0.f);
        }
    }
}

__bidevice__ Spectrum BSDF::Sample_f(const vec3f &woW, vec3f *wiW, const Point2f &u,
                                     Float *pdf, BxDFType *sampledType) const
{
    if(nBxDfs > 0){
        vec3f wo = WorldToLocal(woW);
        vec3f wi;
        Spectrum e = bxdfs[0].Sample_f(wo, &wi, u, pdf, sampledType);
        *wiW = LocalToWorld(wi);
        return e;
    }
    
    *pdf = 0;
    return Spectrum(0.f);
}
