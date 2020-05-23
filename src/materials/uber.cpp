#include <material.h>

__bidevice__ UberMaterial::UberMaterial(Texture<Spectrum> *kd, Texture<Spectrum> *ks,
                                        Texture<Spectrum> *kr, Texture<Spectrum> *kt,
                                        Texture<Float> *urough, Texture<Float> *vrough,
                                        Texture<Spectrum> *op, Texture<Float> *eta,
                                        Texture<Float> *bump)
: bumpMap(bump), Kd(kd), Ks(ks), Kr(kr), Kt(kt), uRough(urough), 
vRough(vrough), opacity(op), eta(eta){ has_bump = 1; }

__bidevice__ UberMaterial::UberMaterial(Texture<Spectrum> *kd, Texture<Spectrum> *ks,
                                        Texture<Spectrum> *kr, Texture<Spectrum> *kt,
                                        Texture<Float> *urough, Texture<Float> *vrough,
                                        Texture<Spectrum> *op, Texture<Float> *eta)
: Kd(kd), Ks(ks), Kr(kr), Kt(kt), uRough(urough), bumpMap(nullptr),
vRough(vrough), opacity(op), eta(eta){ has_bump = 0; }

__bidevice__ void UberMaterial::ComputeScatteringFunctions(BSDF *bsdf, SurfaceInteraction *si, 
                                                           TransportMode mode, 
                                                           bool mLobes)  const
{
    Float e = eta->Evaluate(si);
    Spectrum op = opacity->Evaluate(si);
    Spectrum t = (-op + Spectrum(1.f));
    if(!t.IsBlack()){
        BxDF bxdf(BxDFImpl::SpecularTransmission);
        bxdf.Init_SpecularTransmission(t, 1.f, 1.f, mode);
        bsdf->Push(&bxdf);
    }
    
    Spectrum kd = op * Kd->Evaluate(si);
    if(!kd.IsBlack()){
        BxDF bxdf(BxDFImpl::LambertianReflection);
        bxdf.Init_LambertianReflection(kd);
        bsdf->Push(&bxdf);
    }
    
    Spectrum ks = op * Ks->Evaluate(si);
    if(!ks.IsBlack()){
        Fresnel fresnel;
        fresnel.Init_Dieletric(1.f, e);
        Float roughu = uRough->Evaluate(si);
        Float roughv = vRough->Evaluate(si);
        
        BxDF bxdf(BxDFImpl::MicrofacetReflection);
        bxdf.Init_MicrofacetReflection(ks, roughu, roughv, &fresnel, mode);
        bsdf->Push(&bxdf);
    }
    
    Spectrum kr = op * Kr->Evaluate(si);
    if(!kr.IsBlack()){
        Fresnel fresnel;
        fresnel.Init_Dieletric(1.f, e);
        BxDF bxdf(BxDFImpl::SpecularReflection);
        bxdf.Init_SpecularReflection(kr, &fresnel);
        bsdf->Push(&bxdf);
    }
    
    Spectrum kt = op * Kt->Evaluate(si);
    if(!kt.IsBlack()){
        BxDF bxdf(BxDFImpl::SpecularTransmission);
        bxdf.Init_SpecularTransmission(kt, 1.f, e, mode);
        bsdf->Push(&bxdf);
    }
}
