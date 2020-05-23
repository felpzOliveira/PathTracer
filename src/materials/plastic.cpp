#include <material.h>

__bidevice__ PlasticMaterial::PlasticMaterial(Texture<Spectrum> *kd, Texture<Spectrum> *ks, 
                                              Texture<Float> *rough, Texture<Float> *bump)
: bumpMap(bump), Kd(kd), Ks(ks), roughness(rough){ has_bump = 1; }

__bidevice__ PlasticMaterial::PlasticMaterial(Texture<Spectrum> *kd, Texture<Spectrum> *ks, 
                                              Texture<Float> *rough)
: Kd(kd), Ks(ks), bumpMap(nullptr), roughness(rough){ has_bump = 0; }


__bidevice__ void PlasticMaterial::ComputeScatteringFunctions(BSDF *bsdf, SurfaceInteraction *si, 
                                                              TransportMode mode, 
                                                              bool mLobes)  const
{
    Spectrum kd = Kd->Evaluate(si);
    if(!kd.IsBlack()){
        BxDF bxdf(BxDFImpl::LambertianReflection);
        bxdf.Init_LambertianReflection(kd);
        bsdf->Push(&bxdf);
    }
    
    Spectrum ks = Ks->Evaluate(si);
    if(!ks.IsBlack()){
        Fresnel fresnel;
        fresnel.Init_Dieletric(1.5f, 1.f);
        Float rough = roughness->Evaluate(si);
        BxDF bxdf(BxDFImpl::MicrofacetReflection);
        bxdf.Init_MicrofacetReflection(ks, rough, rough, &fresnel, mode);
        bsdf->Push(&bxdf);
    }
}
