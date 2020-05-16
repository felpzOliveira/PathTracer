#include <material.h>

__bidevice__ void Material::Init_Uber(Texture kd, Texture ks, Texture kr, Texture kt,
                                      Texture roughu, Texture roughv, Texture op, Texture eta)
{
    K = kd;
    I = ks;
    Kr = kr;
    Kt = kt;
    uRough = roughu;
    vRough = roughv;
    index = eta;
    sigma = op;
    type = MaterialType::Uber;
}

__bidevice__ void Material::Init_Uber(Spectrum kd, Spectrum ks, Spectrum kr, Spectrum kt,
                                      Float roughu, Float roughv, Spectrum op, Float eta)
{
    K.Init_ConstantTexture(kd);
    I.Init_ConstantTexture(ks);
    Kr.Init_ConstantTexture(kr);
    Kt.Init_ConstantTexture(kt);
    uRough.Init_ConstantTexture(Spectrum(roughu));
    vRough.Init_ConstantTexture(Spectrum(roughv));
    index.Init_ConstantTexture(Spectrum(eta));
    sigma.Init_ConstantTexture(op);
    type = MaterialType::Uber;
}

__bidevice__ void Material::ComputeScatteringFunctionsUber(BSDF *bsdf,
                                                           SurfaceInteraction *si, 
                                                           TransportMode mode, 
                                                           bool mLobes) const
{
    Float e = index.Evaluate(si)[0];
    Spectrum op = sigma.Evaluate(si);
    
    Spectrum t = (-op + Spectrum(1.f));
    if(!t.IsBlack()){
        BxDF bxdf(BxDFImpl::SpecularTransmission);
        bxdf.Init_SpecularTransmission(t, 1.f, 1.f, mode);
        bsdf->Push(&bxdf);
    }
    
    Spectrum kd = op * K.Evaluate(si);
    if(!kd.IsBlack()){
        BxDF bxdf(BxDFImpl::LambertianReflection);
        bxdf.Init_LambertianReflection(kd);
        bsdf->Push(&bxdf);
    }
    
    Spectrum ks = op * I.Evaluate(si);
    if(!ks.IsBlack()){
        Fresnel fresnel;
        fresnel.Init_Dieletric(1.f, e);
        Float roughu = uRough.Evaluate(si)[0];
        Float roughv = vRough.Evaluate(si)[0];
        
        BxDF bxdf(BxDFImpl::MicrofacetReflection);
        bxdf.Init_MicrofacetReflection(ks, roughu, roughv, &fresnel, mode);
        bsdf->Push(&bxdf);
    }
    
    Spectrum kr = op * Kr.Evaluate(si);
    if(!kr.IsBlack()){
        Fresnel fresnel;
        fresnel.Init_Dieletric(1.f, e);
        BxDF bxdf(BxDFImpl::SpecularReflection);
        bxdf.Init_SpecularReflection(kr, &fresnel);
        bsdf->Push(&bxdf);
    }
    
    Spectrum kt = op * Kt.Evaluate(si);
    if(!kt.IsBlack()){
        BxDF bxdf(BxDFImpl::SpecularTransmission);
        bxdf.Init_SpecularTransmission(kt, 1.f, e, mode);
        bsdf->Push(&bxdf);
    }
}