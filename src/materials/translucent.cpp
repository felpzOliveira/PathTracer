#include <material.h>

__bidevice__ void Material::Init_Translucent(Spectrum kd, Spectrum ks, Float rough,
                                             Spectrum refl, Spectrum trans)
{
    K.Init_ConstantTexture(kd);
    Kt.Init_ConstantTexture(refl);
    I.Init_ConstantTexture(ks);
    T.Init_ConstantTexture(trans);
    uRough.Init_ConstantTexture(rough);
    type = MaterialType::Translucent;
}


__bidevice__ void Material::ComputeScatteringFunctionsTranslucent(BSDF *bsdf, 
                                                                  SurfaceInteraction *si, 
                                                                  TransportMode mode, 
                                                                  bool mLobes) const
{
    Spectrum r = Kt.Evaluate(si);
    Spectrum t = T.Evaluate(si);
    Float eta = 1.5f;
    if(r.IsBlack() && t.IsBlack()) return;
    
    Spectrum kd = K.Evaluate(si);
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
    
    Spectrum ks = I.Evaluate(si);
    if(!ks.IsBlack() && (!r.IsBlack() || !t.IsBlack())){
        Float rough = uRough.Evaluate(si)[0];
        if(!r.IsBlack()){
            Fresnel fresnel;
            fresnel.Init_Dieletric(1.f, eta);
            BxDF mRefl(BxDFImpl::MicrofacetReflection);
            mRefl.Init_MicrofacetReflection(r * ks, rough, rough, &fresnel, mode);
            bsdf->Push(&mRefl);
        }
        
        if(!t.IsBlack()){
            BxDF mTran(BxDFImpl::MicrofacetTransmission);
            mTran.Init_MicrofacetTransmission(t * ks, 1.f, eta, rough, rough, mode);
            bsdf->Push(&mTran);
        }
    }
}