#include <material.h>

__bidevice__ MatteMaterial::MatteMaterial(Texture<Spectrum> *kd, Texture<Float> *sig, 
                                          Texture<Float> *bump)
: bumpMap(bump), Kd(kd), sigma(sig){ has_bump = 1; }

__bidevice__ MatteMaterial::MatteMaterial(Texture<Spectrum> *kd, Texture<Float> *sig)
: Kd(kd), bumpMap(nullptr), sigma(sig){ has_bump = 0; }

__bidevice__ 
void MatteMaterial::ComputeScatteringFunctions(BSDF *bsdf, SurfaceInteraction *si, 
                                               TransportMode mode, bool mLobes) const
{
    Spectrum kd = Kd->Evaluate(si);
    Float sig = sigma->Evaluate(si);
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