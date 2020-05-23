#include <material.h>

__bidevice__ GlassMaterial::GlassMaterial(Texture<Spectrum> *kr, Texture<Spectrum> *kt,
                                          Texture<Float> *urough, Texture<Float> *vrough,
                                          Texture<Float> *index, Texture<Float> *bump)
: bumpMap(bump), Kr(kr), Kt(kt), uRough(urough), vRough(vrough), index(index){ has_bump = 1; }

__bidevice__ GlassMaterial::GlassMaterial(Texture<Spectrum> *kr, Texture<Spectrum> *kt,
                                          Texture<Float> *urough, Texture<Float> *vrough,
                                          Texture<Float> *index)
: Kr(kr), Kt(kt), uRough(urough), vRough(vrough), bumpMap(nullptr), 
index(index){ has_bump = 0; }

__bidevice__ void GlassMaterial::ComputeScatteringFunctions(BSDF *bsdf, SurfaceInteraction *si, 
                                                            TransportMode mode, 
                                                            bool mLobes)  const
{
    Float eta = index->Evaluate(si);
    Float urough = uRough->Evaluate(si);
    Float vrough = vRough->Evaluate(si);
    Spectrum R = Kr->Evaluate(si);
    Spectrum T = Kt->Evaluate(si);
    
    if(R.IsBlack() && T.IsBlack()) return;
    
    bool isSpecular = IsZero(urough) && IsZero(vrough);
    if(mLobes && isSpecular){
        BxDF bxdf(BxDFImpl::FresnelSpecular);
        bxdf.Init_FresnelSpecular(R, T, 1.f, eta, mode);
        bsdf->Push(&bxdf);
    }else{
        if(!T.IsBlack()){
            if(isSpecular){
                BxDF bxdf(BxDFImpl::SpecularTransmission);
                bxdf.Init_SpecularTransmission(T, 1.f, eta, mode);
                bsdf->Push(&bxdf);
            }else{
                BxDF bxdf(BxDFImpl::MicrofacetTransmission);
                bxdf.Init_MicrofacetTransmission(T, 1.f, eta, urough, vrough, mode);
                bsdf->Push(&bxdf);
            }
        }
        
        if(!R.IsBlack()){
            Fresnel fresnel;
            fresnel.Init_Dieletric(1.f, eta);
            if(isSpecular){
                BxDF bxdf(BxDFImpl::SpecularReflection);
                bxdf.Init_SpecularReflection(R, &fresnel);
                bsdf->Push(&bxdf);
            }else{
                BxDF bxdf(BxDFImpl::MicrofacetReflection);
                bxdf.Init_MicrofacetReflection(R, urough, vrough, &fresnel, mode);
                bsdf->Push(&bxdf);
            }
        }
    }
}
