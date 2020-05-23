#include <material.h>

__bidevice__ MetalMaterial::MetalMaterial(Texture<Spectrum> *eta, Texture<Spectrum> *k, 
                                          Texture<Float> *urough, Texture<Float> *vrough,
                                          Texture<Float> *bump)
: bumpMap(bump), eta(eta), k(k), uRough(urough), vRough(vrough) { has_bump = 1; }

__bidevice__ MetalMaterial::MetalMaterial(Texture<Spectrum> *eta, Texture<Spectrum> *k, 
                                          Texture<Float> *urough, Texture<Float> *vrough)
: eta(eta), k(k), uRough(urough), bumpMap(nullptr), vRough(vrough) { has_bump = 0; }

__bidevice__ void MetalMaterial::ComputeScatteringFunctions(BSDF *bsdf, SurfaceInteraction *si, 
                                                            TransportMode mode, 
                                                            bool mLobes)  const
{
    Float urough = uRough->Evaluate(si);
    Float vrough = vRough->Evaluate(si);
    
    Fresnel fresnel;
    fresnel.Init_Conductor(1.f, eta->Evaluate(si), k->Evaluate(si));
    
    BxDF bxdf(BxDFImpl::MicrofacetReflection);
    bxdf.Init_MicrofacetReflection(1.f, urough, vrough, &fresnel, mode);
    bsdf->Push(&bxdf);
}
