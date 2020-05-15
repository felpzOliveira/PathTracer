#include <material.h>

__bidevice__ void Material::Init_Plastic(Spectrum kd, Spectrum ks, Float rough){
    K.Init_ConstantTexture(kd);
    Kt.Init_ConstantTexture(ks);
    uRough.Init_ConstantTexture(rough);
    type = MaterialType::Plastic;
}

__bidevice__ void Material::Init_Plastic(Texture kd, Texture ks, Texture rough){
    K = kd;
    Kt = ks;
    uRough = rough;
    type = MaterialType::Plastic;
}

__bidevice__ void Material::ComputeScatteringFunctionsPlastic(BSDF *bsdf, 
                                                              SurfaceInteraction *si, 
                                                              TransportMode mode, 
                                                              bool mLobes) const
{
    Spectrum kd = K.Evaluate(si);
    if(!kd.IsBlack()){
        BxDF bxdf(BxDFImpl::LambertianReflection);
        bxdf.Init_LambertianReflection(kd);
        bsdf->Push(&bxdf);
    }
    
    Spectrum ks = Kt.Evaluate(si);
    if(!ks.IsBlack()){
        Fresnel fresnel;
        fresnel.Init_Dieletric(1.5f, 1.f);
        Float rough = uRough.Evaluate(si)[0];
        BxDF bxdf(BxDFImpl::MicrofacetReflection);
        bxdf.Init_MicrofacetReflection(ks, rough, rough, &fresnel, mode);
        bsdf->Push(&bxdf);
    }
}