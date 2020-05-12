#include <material.h>

__bidevice__ void Material::Init_Matte(Texture _Kd, Texture _sigma){
    K = _Kd;
    sigma = _sigma;
    type = MaterialType::Matte;
}

__bidevice__ void Material::ComputeScatteringFunctionsMatte(BSDF *bsdf, 
                                                            SurfaceInteraction *si, 
                                                            TransportMode mode, 
                                                            bool mLobes) const
{
    Spectrum kd = K.Evaluate(si);
    Float sig = sigma.Evaluate(si)[0];
    if(!kd.IsBlack()){
        if(IsZero(sig)){
            BxDF bxdf(BxDFImpl::LambertianReflection);
            bxdf.Init_LambertianReflection(kd);
            bsdf->Push(&bxdf);
        }else{
            BxDF bxdf(BxDFImpl::OrenNayar);
            bxdf.Init_OrenNayar(kd, sig);
            bsdf->Push(&bxdf);
        }
    }
}