#include <material.h>

__bidevice__ void Material::Init_Mirror(Texture Kr){
    K = Kr;
    type = MaterialType::Mirror;
}

__bidevice__ void Material::ComputeScatteringFunctionsMirror(BSDF *bsdf, 
                                                             SurfaceInteraction *si, 
                                                             TransportMode mode, 
                                                             bool mLobes) const
{
    Spectrum Kr = K.Evaluate(si);
    if(!Kr.IsBlack()){
        BxDF bxdf(BxDFImpl::SpecularReflection);
        Fresnel fr; //no op by default
        bxdf.Init_SpecularReflection(Kr, &fr);
        bsdf->Push(&bxdf);
    }
}