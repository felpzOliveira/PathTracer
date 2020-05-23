#include <material.h>

__bidevice__ TranslucentMaterial::TranslucentMaterial(Texture<Spectrum> *kd, 
                                                      Texture<Spectrum> *ks,
                                                      Texture<Spectrum> *refl, 
                                                      Texture<Spectrum> *trans,
                                                      Texture<Float> *rough, 
                                                      Texture<Float> *bump)
: bumpMap(bump), Kd(kd), Ks(ks), reflect(refl), transmit(trans), roughness(rough)
{ has_bump = 1; }

__bidevice__ TranslucentMaterial::TranslucentMaterial(Texture<Spectrum> *kd, 
                                                      Texture<Spectrum> *ks,
                                                      Texture<Spectrum> *refl, 
                                                      Texture<Spectrum> *trans,
                                                      Texture<Float> *rough)
: Kd(kd), Ks(ks), reflect(refl), transmit(trans), bumpMap(nullptr), 
roughness(rough){ has_bump = 0; }

__bidevice__ void TranslucentMaterial::ComputeScatteringFunctions(BSDF *bsdf, 
                                                                  SurfaceInteraction *si, 
                                                                  TransportMode mode, 
                                                                  bool mLobes)  const
{
    Float eta = 1.5f;
    
    Spectrum r = reflect->Evaluate(si);
    Spectrum t = transmit->Evaluate(si);
    if(r.IsBlack() && t.IsBlack()) return;
    
    Spectrum kd = Kd->Evaluate(si);
    if(!kd.IsBlack()){
        if(!r.IsBlack()){
            BxDF lRefl(BxDFImpl::LambertianReflection);
            lRefl.Init_LambertianReflection(r * kd);
            bsdf->Push(&lRefl);
        }
        
        if(!t.IsBlack()){
            BxDF lTran(BxDFImpl::LambertianTransmission);
            Spectrum f = t*kd;
            lTran.Init_LambertianTransmission(f);
            bsdf->Push(&lTran);
        }
    }
    
    Spectrum ks = Ks->Evaluate(si);
    if(!ks.IsBlack() && (!r.IsBlack() || !t.IsBlack())){
        Float rough = roughness->Evaluate(si);
        if(!r.IsBlack()){
            Fresnel fresnel;
            fresnel.Init_Dieletric(1.f, eta);
            BxDF mRefl(BxDFImpl::MicrofacetReflection);
            mRefl.Init_MicrofacetReflection(r * ks, rough, rough, &fresnel, mode);
            bsdf->Push(&mRefl);
        }
        
        if (!t.IsBlack()){
            BxDF mTran(BxDFImpl::MicrofacetTransmission);
            mTran.Init_MicrofacetTransmission(t * ks, 1.f, eta, rough, rough, mode);
            bsdf->Push(&mTran);
        }
    }
}
