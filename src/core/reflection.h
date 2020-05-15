#pragma once

#include <geometry.h>
#include <cutil.h>
#include <interaction.h>
#include <microfacet.h>

#define MAX_BxDFS 8

enum BxDFType {
    BSDF_REFLECTION = 1 << 0,
    BSDF_TRANSMISSION = 1 << 1,
    BSDF_DIFFUSE = 1 << 2,
    BSDF_GLOSSY = 1 << 3,
    BSDF_SPECULAR = 1 << 4,
    BSDF_ALL = BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_SPECULAR | BSDF_REFLECTION |
        BSDF_TRANSMISSION,
};

enum FresnelType{
    NoOp, Dieletric, Conductor
};

class Fresnel{
    public:
    FresnelType type;
    Float etaI, etaT;
    Spectrum setaI, setaT, sk;
    
    __bidevice__ Fresnel(): type(FresnelType::NoOp){}
    __bidevice__ void Init_Dieletric(Float etaI, Float etaT);
    __bidevice__ void Init_Conductor(Spectrum etaI, Spectrum etaT, Spectrum k);
    
    __bidevice__ Spectrum Evaluate(Float cosI) const;
    
    private:
    __bidevice__ Spectrum EvaluateDieletric(Float cosI) const;
    __bidevice__ Spectrum EvaluateConductor(Float cosI) const;
};

enum BxDFImpl{
    LambertianReflection = 0,
    LambertianTransmission,
    SpecularReflection,
    SpecularTransmission,
    FresnelSpecular,
    OrenNayar,
    MicrofacetTransmission,
    MicrofacetReflection,
    FresnelBlend
};

class BxDF{
    public:
    BxDFType type;
    BxDFImpl impl;
    
    Spectrum S, T;
    Fresnel fresnel;
    
    TransportMode mode;
    Float A, B;
    
    MicrofacetDistribution mDist;
    
    __bidevice__ BxDF(){}
    __bidevice__ BxDF(BxDFImpl impl) : impl(impl), type(BxDFType(0)){}
    
    __bidevice__ void Init_LambertianReflection(const Spectrum &R){
        type = BxDFType(BSDF_REFLECTION | BSDF_DIFFUSE);
        impl = BxDFImpl::LambertianReflection;
        S = R;
    }
    
    __bidevice__ void Init_LambertianTransmission(const Spectrum &_T){
        type = BxDFType(BSDF_TRANSMISSION | BSDF_DIFFUSE);
        impl = BxDFImpl::LambertianTransmission;
        S = _T;
    }
    
    __bidevice__ void Init_SpecularReflection(const Spectrum &R, Fresnel *_fresnel){
        type = BxDFType(BSDF_REFLECTION | BSDF_SPECULAR);
        impl = BxDFImpl::SpecularReflection;
        S = R;
        fresnel = *_fresnel;
    }
    
    __bidevice__ void Init_SpecularTransmission(const Spectrum &T, Float _etaA, 
                                                Float _etaB, TransportMode _mode)
    {
        type = BxDFType(BSDF_TRANSMISSION | BSDF_SPECULAR);
        impl = BxDFImpl::SpecularTransmission;
        S = T;
        A = _etaA;
        B = _etaB;
        fresnel.Init_Dieletric(A, B);
        mode = _mode;
    }
    
    __bidevice__ void Init_FresnelSpecular(const Spectrum &R, const Spectrum &_T,
                                           Float eA, Float eB, TransportMode _mode)
    {
        type = BxDFType(BSDF_REFLECTION | BSDF_TRANSMISSION | BSDF_SPECULAR);
        impl = BxDFImpl::FresnelSpecular;
        S = R;
        T = _T;
        A = eA;
        B = eB;
        mode = _mode;
    }
    
    __bidevice__ void Init_OrenNayar(const Spectrum &R, Float sigma){
        type = BxDFType(BSDF_REFLECTION | BSDF_DIFFUSE);
        impl = BxDFImpl::OrenNayar;
        S = R;
        Float sigma2 = sigma * sigma;
        A = 1.f - (sigma2 / (2.f * (sigma2 + 0.33f)));
        B = 0.45f * sigma2 / (sigma2 + 0.09f);
    }
    
    __bidevice__ void Init_MicrofacetTransmission(const Spectrum &_T, Float eA, Float eB,
                                                  Float alphax, Float alphay,
                                                  TransportMode _mode)
    {
        type = BxDFType(BSDF_TRANSMISSION | BSDF_GLOSSY);
        impl = BxDFImpl::MicrofacetTransmission;
        A = eA;
        B = eB;
        T = _T;
        fresnel.Init_Dieletric(A, B);
        mDist.Set(alphax, alphay);
        mode = _mode;
    }
    
    __bidevice__ void Init_MicrofacetReflection(const Spectrum &R, Float alphax,
                                                Float alphay, Fresnel *_fresnel,
                                                TransportMode _mode)
    {
        type = BxDFType(BSDF_REFLECTION | BSDF_GLOSSY);
        impl = BxDFImpl::MicrofacetReflection;
        S = R;
        fresnel = *_fresnel;
        mDist.Set(alphax, alphay);
        mode = _mode;
    }
    
    __bidevice__ void Init_MicrofacetReflection(const Spectrum &R, Float eA, Float eB,
                                                Float alphax, Float alphay,
                                                TransportMode _mode)
    {
        type = BxDFType(BSDF_REFLECTION | BSDF_GLOSSY);
        impl = BxDFImpl::MicrofacetReflection;
        A = eA;
        B = eB;
        S = R;
        fresnel.Init_Dieletric(A, B);
        mDist.Set(alphax, alphay);
        mode = _mode;
    }
    
    __bidevice__ void Init_FresnelBlend(const Spectrum &R, const Spectrum &_T,
                                        Float alphax, Float alphay)
    {
        type = BxDFType(BSDF_REFLECTION | BSDF_GLOSSY);
        impl = BxDFImpl::FresnelBlend;
        S = R;
        T = _T;
        mDist.Set(alphax, alphay);
    }
    
    __bidevice__ Spectrum f(const vec3f &wo, const vec3f &wi) const;
    
    __bidevice__ Spectrum Sample_f(const vec3f &wo, vec3f *wi,
                                   const Point2f &sample, Float *pdf,
                                   BxDFType *sampledType = nullptr) const;
    
    __bidevice__ Float Pdf(const vec3f &wo, const vec3f &wi) const;
    
    
    __bidevice__ bool MatchesFlags(BxDFType t) const { return (type & t) == type; }
    
    private:
    __bidevice__ Spectrum LambertianReflection_Sample_f(const vec3f &wo, vec3f *wi,
                                                        const Point2f &sample, Float *pdf,
                                                        BxDFType *sampledType = nullptr) const;
    __bidevice__ Spectrum LambertianReflection_f(const vec3f &wo, const vec3f &wi) const;
    __bidevice__ Float LambertianReflection_Pdf(const vec3f &wo, const vec3f &wi) const;
    
    __bidevice__ Spectrum LambertianTransmission_Sample_f(const vec3f &wo, vec3f *wi,
                                                          const Point2f &sample, Float *pdf,
                                                          BxDFType *sampledType = nullptr) const;
    __bidevice__ Spectrum LambertianTransmission_f(const vec3f &wo, const vec3f &wi) const;
    __bidevice__ Float LambertianTransmission_Pdf(const vec3f &wo, const vec3f &wi) const;
    
    
    __bidevice__ Spectrum SpecularReflection_f(const vec3f &wo, const vec3f &wi) const{
        return Spectrum(0.f);
    }
    
    __bidevice__ Float SpecularReflection_Pdf(const vec3f &wo, const vec3f &wi) const{
        return 0;
    }
    
    __bidevice__ Spectrum SpecularReflection_Sample_f(const vec3f &wo, vec3f *wi,
                                                      const Point2f &sample, Float *pdf,
                                                      BxDFType *sampledType = nullptr) const;
    
    __bidevice__ Spectrum SpecularTransmission_f(const vec3f &wo, const vec3f &wi) const{
        return Spectrum(0.f);
    }
    
    __bidevice__ Float SpecularTransmission_Pdf(const vec3f &wo, const vec3f &wi) const{
        return 0;
    }
    
    __bidevice__ Spectrum SpecularTransmission_Sample_f(const vec3f &wo, vec3f *wi,
                                                        const Point2f &sample, Float *pdf,
                                                        BxDFType *sampledType = nullptr) const;
    
    __bidevice__ Spectrum FresnelSpecular_f(const vec3f &wo, const vec3f &wi) const{
        return Spectrum(0.f);
    }
    
    __bidevice__ Float FresnelSpecular_Pdf(const vec3f &wo, const vec3f &wi) const{
        return 0;
    }
    
    __bidevice__ Spectrum FresnelSpecular_Sample_f(const vec3f &wo, vec3f *wi,
                                                   const Point2f &sample, Float *pdf,
                                                   BxDFType *sampledType = nullptr) const;
    
    __bidevice__ Spectrum OrenNayar_f(const vec3f &wo, const vec3f &wi) const;
    __bidevice__ Float OrenNayar_Pdf(const vec3f &wo, const vec3f &wi) const{
        return LambertianReflection_Pdf(wo, wi);
    }
    
    __bidevice__ Spectrum OrenNayar_Sample_f(const vec3f &wo, vec3f *wi,
                                             const Point2f &sample, Float *pdf,
                                             BxDFType *sampledType = nullptr) const
    {
        return LambertianReflection_Sample_f(wo, wi, sample, pdf, sampledType);
    }
    
    __bidevice__ Spectrum MicrofacetTransmission_f(const vec3f &wo, const vec3f &wi) const;
    __bidevice__ Float MicrofacetTransmission_Pdf(const vec3f &wo, const vec3f &wi) const;
    __bidevice__ Spectrum MicrofacetTransmission_Sample_f(const vec3f &wo, vec3f *wi,
                                                          const Point2f &sample, Float *pdf,
                                                          BxDFType *sampledType = nullptr) const;
    
    __bidevice__ Spectrum MicrofacetReflection_f(const vec3f &wo, const vec3f &wi) const;
    __bidevice__ Float MicrofacetReflection_Pdf(const vec3f &wo, const vec3f &wi) const;
    __bidevice__ Spectrum MicrofacetReflection_Sample_f(const vec3f &wo, vec3f *wi,
                                                        const Point2f &sample, Float *pdf,
                                                        BxDFType *sampledType = nullptr) const;
    
    __bidevice__ Spectrum FresnelBlend_f(const vec3f &wo, const vec3f &wi) const;
    __bidevice__ Float FresnelBlend_Pdf(const vec3f &wo, const vec3f &wi) const;
    __bidevice__ Spectrum FresnelBlend_Sample_f(const vec3f &wo, vec3f *wi,
                                                const Point2f &sample, Float *pdf,
                                                BxDFType *sampledType = nullptr) const;
};

class BSDF{
    public:
    BxDF bxdfs[MAX_BxDFS];
    int nBxDFs;
    const Normal3f ns, ng;
    const vec3f ss, ts;
    
    __bidevice__ BSDF(const SurfaceInteraction &si)
        : ns(si.n), ng(si.n), ss(Normalize(si.dpdu)), ts(Cross(ToVec3(ns), ss)) 
    {
        nBxDFs = 0;
    }
    
    __bidevice__ vec3f LocalToWorld(const vec3f &v) const{
        return vec3f(ss.x * v.x + ts.x * v.y + ns.x * v.z,
                     ss.y * v.x + ts.y * v.y + ns.y * v.z,
                     ss.z * v.x + ts.z * v.y + ns.z * v.z);
    }
    
    __bidevice__ vec3f WorldToLocal(const vec3f &v) const{
        return vec3f(Dot(v, ss), Dot(v, ts), Dot(v, ToVec3(ns)));
    }
    
    __bidevice__ Spectrum Sample_f(const vec3f &woW, vec3f *wiW, const Point2f &u,
                                   Float *pdf, BxDFType type,
                                   BxDFType *sampledType = nullptr) const;
    
    __bidevice__ void Push(BxDF *bxdf){
        if(nBxDFs < MAX_BxDFS){
            bxdfs[nBxDFs++] = *bxdf;
        }
    }
    
    __bidevice__ int NumComponents(BxDFType flags) const;
};