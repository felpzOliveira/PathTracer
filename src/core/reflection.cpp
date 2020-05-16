#include <reflection.h>

__bidevice__ Float pow5(Float v){
    return v*v*v*v*v;
}

__bidevice__ Spectrum SchlickFresnel(Float cosTheta, Spectrum Rs){
    return Rs + pow5(1 - cosTheta) * (Spectrum(1.) - Rs);
}

__bidevice__ Float FrDieletric(Float cosThetaI, Float etaI, Float etaT);

__bidevice__ Spectrum FrConductor(Float cosThetaI, const Spectrum &etai,
                                  const Spectrum &etat, const Spectrum &k);

__bidevice__ Float BxDF::LambertianReflection_Pdf(const vec3f &wo, const vec3f &wi) const{
    return SameHemisphere(wo, wi) ? AbsCosTheta(wi) * InvPi : 0;
}

__bidevice__ Float BxDF::LambertianTransmission_Pdf(const vec3f &wo, const vec3f &wi) const{
    return !SameHemisphere(wo, wi) ? AbsCosTheta(wi) * InvPi : 0;
}

__bidevice__ Float BxDF::MicrofacetTransmission_Pdf(const vec3f &wo, const vec3f &wi) const{
    if(SameHemisphere(wo, wi)) return 0;
    Float eta = CosTheta(wo) > 0 ? (B / A) : (A / B);
    vec3f h = wo + wi * eta;
    vec3f wh = Normalize(h);
    
    Float sqrtDenom = Dot(wo, wh) + eta * Dot(wi, wh);
    Float dwh_dwi = Absf((eta * eta * Dot(wi, wh)) / (sqrtDenom * sqrtDenom));
    return mDist.Pdf(wo, wh) * dwh_dwi;
}

__bidevice__ Float BxDF::MicrofacetReflection_Pdf(const vec3f &wo, const vec3f &wi) const{
    if(!SameHemisphere(wo, wi)) return 0;
    vec3f wh = Normalize(wo + wi);
    return mDist.Pdf(wo, wh) / (4 * Dot(wo, wh));
}

__bidevice__ Float BxDF::FresnelBlend_Pdf(const vec3f &wo, const vec3f &wi) const{
    if(!SameHemisphere(wo, wi)) return 0;
    vec3f wh = Normalize(wo + wi);
    Float pdf_wh = mDist.Pdf(wo, wh);
    return .5f * (AbsCosTheta(wi) * InvPi + pdf_wh / (4 * Dot(wo, wh)));
}

__bidevice__ Float BxDF::Pdf(const vec3f &wo, const vec3f &wi) const{
    switch(impl){
        case BxDFImpl::LambertianReflection:{
            return LambertianReflection_Pdf(wo, wi);
        } break;
        
        case BxDFImpl::LambertianTransmission:{
            return LambertianTransmission_Pdf(wo, wi);
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
        
        case BxDFImpl::MicrofacetTransmission:{
            return MicrofacetTransmission_Pdf(wo, wi);
        } break;
        
        case BxDFImpl::MicrofacetReflection:{
            return MicrofacetReflection_Pdf(wo, wi);
        } break;
        
        case BxDFImpl::FresnelBlend:{
            return FresnelBlend_Pdf(wo, wi);
        } break;
        
        default:{
            return 0;
        }
    }
}

__bidevice__ Spectrum BxDF::LambertianReflection_f(const vec3f &wo, const vec3f &wi) const{
    return S * InvPi;
}

__bidevice__ Spectrum BxDF::LambertianTransmission_f(const vec3f &wo, const vec3f &wi) const{
    return S * InvPi;
}

__bidevice__ Spectrum BxDF::OrenNayar_f(const vec3f &wo, const vec3f &wi) const{
    Float sinThetaI = SinTheta(wi);
    Float sinThetaO = SinTheta(wo);
    Float maxCos = 0;
    if(sinThetaI > 1e-4 && sinThetaO > 1e-4){
        Float sinPhiI = SinPhi(wi), cosPhiI = CosPhi(wi);
        Float sinPhiO = SinPhi(wo), cosPhiO = CosPhi(wo);
        Float dCos = cosPhiI * cosPhiO + sinPhiI * sinPhiO;
        maxCos = Max((Float)0, dCos);
    }
    
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

__bidevice__ Spectrum BxDF::MicrofacetTransmission_f(const vec3f &wo, const vec3f &wi) const{
    if(SameHemisphere(wo, wi)) return 0;
    
    Float cosThetaO = CosTheta(wo);
    Float cosThetaI = CosTheta(wi);
    if(IsZero(cosThetaI) || IsZero(cosThetaO)) return Spectrum(0);
    
    Float eta = CosTheta(wo) > 0 ? (B / A) : (A / B);
    vec3f wh = Normalize(wo + wi * eta);
    if (wh.z < 0) wh = -wh;
    
    Spectrum F = fresnel.Evaluate(Dot(wo, wh));
    
    Float sqrtDenom = Dot(wo, wh) + eta * Dot(wi, wh);
    Float factor = (mode == TransportMode::Radiance) ? (1 / eta) : 1;
    
    return (Spectrum(1.f) - F) * T * Absf(mDist.D(wh) * mDist.G(wo, wi) * eta * eta *
                                          AbsDot(wi, wh) * AbsDot(wo, wh) * factor * factor /
                                          (cosThetaI * cosThetaO * sqrtDenom * sqrtDenom));
}

__bidevice__ Spectrum BxDF::MicrofacetReflection_f(const vec3f &wo, const vec3f &wi) const{
    Float cosThetaO = AbsCosTheta(wo), cosThetaI = AbsCosTheta(wi);
    vec3f wh = wi + wo;
    if(IsZero(cosThetaI) || IsZero(cosThetaO)) return Spectrum(0.);
    if(IsZero(wh.x) && IsZero(wh.y) && IsZero(wh.z)) return Spectrum(0.);
    wh = Normalize(wh);
    Spectrum F = fresnel.Evaluate(Dot(wi, Faceforward(wh, vec3f(0,0,1))));
    return S * mDist.D(wh) * mDist.G(wo, wi) * F / (4 * cosThetaI * cosThetaO);
}

__bidevice__ Spectrum BxDF::FresnelBlend_f(const vec3f &wo, const vec3f &wi) const{
    Spectrum diffuse = (28.f / (23.f * Pi)) * S * (Spectrum(1.f) - T) *
        (1 - pow5(1 - .5f * AbsCosTheta(wi))) * (1 - pow5(1 - .5f * AbsCosTheta(wo)));
    vec3f wh = wi + wo;
    if(IsZero(wh.x) && IsZero(wh.y) && IsZero(wh.z)) return Spectrum(0);
    
    wh = Normalize(wh);
    Spectrum specular = mDist.D(wh) / (4 * AbsDot(wi, wh) * Max(AbsCosTheta(wi), 
                                                                AbsCosTheta(wo))) *
        SchlickFresnel(Dot(wi, wh), T);
    
    return diffuse + specular;
}

__bidevice__ Spectrum BxDF::f(const vec3f &wo, const vec3f &wi) const{
    switch(impl){
        case BxDFImpl::LambertianReflection:{
            return LambertianReflection_f(wo, wi);
        } break;
        
        case BxDFImpl::LambertianTransmission:{
            return LambertianTransmission_f(wo, wi);
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
        
        case BxDFImpl::MicrofacetTransmission:{
            return MicrofacetTransmission_f(wo, wi);
        } break;
        
        case BxDFImpl::MicrofacetReflection:{
            return MicrofacetReflection_f(wo, wi);
        } break;
        
        case BxDFImpl::FresnelBlend:{
            return FresnelBlend_f(wo, wi);
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

__bidevice__ Spectrum BxDF::LambertianTransmission_Sample_f(const vec3f &wo, vec3f *wi, 
                                                            const Point2f &u,Float *pdf, 
                                                            BxDFType *sampledType) const 
{
    *wi = CosineSampleHemisphere(u);
    if (wo.z > 0) wi->z *= -1;
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

__bidevice__ Spectrum BxDF::MicrofacetTransmission_Sample_f(const vec3f &wo, vec3f *wi,
                                                            const Point2f &sample, Float *pdf,
                                                            BxDFType *sampledType) const
{
    if(IsZero(wo.z)) return 0.;
    vec3f wh = mDist.Sample_wh(wo, sample);
    if(Dot(wo, wh) < 0) return 0.;
    
    Float eta = CosTheta(wo) > 0 ? (A / B) : (B / A);
    if(!Refract(wo, (Normal3f)wh, eta, wi)) return 0;
    *pdf = Pdf(wo, *wi);
    return f(wo, *wi);
}

__bidevice__ Spectrum BxDF::MicrofacetReflection_Sample_f(const vec3f &wo, vec3f *wi,
                                                          const Point2f &sample, Float *pdf,
                                                          BxDFType *sampledType) const
{
    if (IsZero(wo.z)) return 0.;
    vec3f wh = mDist.Sample_wh(wo, sample);
    if(Dot(wo, wh) < 0) return 0.;
    *wi = Reflect(wo, wh);
    if(!SameHemisphere(wo, *wi)) return Spectrum(0.f);
    
    *pdf = mDist.Pdf(wo, wh) / (4 * Dot(wo, wh));
    return f(wo, *wi);
}

__bidevice__ Spectrum BxDF::FresnelBlend_Sample_f(const vec3f &wo, vec3f *wi,
                                                  const Point2f &sample, Float *pdf,
                                                  BxDFType *sampledType) const
{
    Point2f u = sample;
    if(u[0] < .5){
        u[0] = Min(2 * u[0], OneMinusEpsilon);
        *wi = CosineSampleHemisphere(u);
        if(wo.z < 0) wi->z *= -1;
    }else{
        u[0] = Min(2 * (u[0] - .5f), OneMinusEpsilon);
        vec3f wh = mDist.Sample_wh(wo, u);
        *wi = Reflect(wo, wh);
        if(!SameHemisphere(wo, *wi)) return Spectrum(0.f);
    }
    
    *pdf = Pdf(wo, *wi);
    return f(wo, *wi);
}

__bidevice__ Spectrum BxDF::Sample_f(const vec3f &wo, vec3f *wi,
                                     const Point2f &sample, Float *pdf,
                                     BxDFType *sampledType) const
{
    switch(impl){
        case BxDFImpl::LambertianReflection:{
            return LambertianReflection_Sample_f(wo, wi, sample, pdf, sampledType);
        } break;
        
        case BxDFImpl::LambertianTransmission:{
            return LambertianTransmission_Sample_f(wo, wi, sample, pdf, sampledType);
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
        
        case BxDFImpl::MicrofacetTransmission:{
            return MicrofacetTransmission_Sample_f(wo, wi, sample, pdf, sampledType);
        } break;
        
        case BxDFImpl::MicrofacetReflection:{
            return MicrofacetReflection_Sample_f(wo, wi, sample, pdf, sampledType);
        } break;
        
        case BxDFImpl::FresnelBlend:{
            return FresnelBlend_Sample_f(wo, wi, sample, pdf, sampledType);
        } break;
        
        default:{
            *pdf = 0;
            return Spectrum(0.f);
        }
    }
}

__bidevice__ int BSDF::NumComponents(BxDFType flags) const{
    int num = 0;
    for(int i = 0; i < nBxDFs; ++i)
        if(bxdfs[i].MatchesFlags(flags)) ++num;
    return num;
}

__bidevice__ Spectrum BSDF::Sample_f(const vec3f &woWorld, vec3f *wiWorld, const Point2f &u,
                                     Float *pdf, BxDFType type, BxDFType *sampledType) const
{
    int matchingComps = NumComponents(type);
    if (matchingComps == 0) {
        *pdf = 0;
        if (sampledType) *sampledType = BxDFType(0);
        return Spectrum(0);
    }
    int comp = Min((int)std::floor(u[0] * matchingComps), matchingComps - 1);
    
    BxDF *bxdf = nullptr;
    int count = comp;
    int picked = -1;
    for(int i = 0; i < nBxDFs; ++i){
        if(bxdfs[i].MatchesFlags(type) && count-- == 0){
            bxdf = (BxDF *)&bxdfs[i];
            picked = i;
            break;
        }
    }
    
    Point2f uRemapped(Min(u[0] * matchingComps - comp, OneMinusEpsilon), u[1]);
    
    vec3f wi, wo = WorldToLocal(woWorld);
    if(IsZero(wo.z)) return 0.;
    
    *pdf = 0;
    
    if(sampledType) *sampledType = bxdf->type;
    Spectrum f = bxdf->Sample_f(wo, &wi, uRemapped, pdf, sampledType);
    
    if(IsZero(*pdf)){
        if(sampledType) *sampledType = BxDFType(0);
        return 0;
    }
    
    *wiWorld = LocalToWorld(wi);
    
    if(!(bxdf->type & BSDF_SPECULAR) && matchingComps > 1){
        for(int i = 0; i < nBxDFs; ++i){
            if(i != picked && bxdfs[i].MatchesFlags(type)){
                *pdf += bxdfs[i].Pdf(wo, wi);
            }
        }
    }
    
    if(matchingComps > 1) *pdf /= matchingComps;
    
    if(!(bxdf->type & BSDF_SPECULAR)){
        bool reflect = Dot(*wiWorld, ToVec3(ng)) * Dot(woWorld, ToVec3(ng)) > 0;
        f = 0.;
        
        for(int i = 0; i < nBxDFs; ++i){
            if(bxdfs[i].MatchesFlags(type) &&
               ((reflect && (bxdfs[i].type & BSDF_REFLECTION)) ||
                (!reflect && (bxdfs[i].type & BSDF_TRANSMISSION))))
            {
                f += bxdfs[i].f(wo, wi);
            }
        }
    }
    
    return f;
}
