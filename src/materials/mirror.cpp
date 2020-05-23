#include <material.h>

__bidevice__ MirrorMaterial::MirrorMaterial(Texture<Spectrum> *kr, Texture<Float> *bump)
: bumpMap(bump), Kr(kr){ has_bump = 1; }

__bidevice__ MirrorMaterial::MirrorMaterial(Texture<Spectrum> *kr) : 
Kr(kr), bumpMap(nullptr){ has_bump = 0; }

__bidevice__ void MirrorMaterial::ComputeScatteringFunctions(BSDF *bsdf, SurfaceInteraction *si, 
                                                             TransportMode mode, 
                                                             bool mLobes)  const
{
    Spectrum kr = Kr->Evaluate(si);
    if(!kr.IsBlack()){
        BxDF bxdf(BxDFImpl::SpecularReflection);
        Fresnel fr; //no op by default
        bxdf.Init_SpecularReflection(kr, &fr);
        bsdf->Push(&bxdf);
    }
}
