#include <material.h>

__bidevice__ void Material::Init_Glass(Spectrum Kr, Spectrum _Kt, Float _uRough,
                                       Float _vRough, Float _index)
{
    Kt.Init_ConstantTexture(_Kt);
    K.Init_ConstantTexture(Kr);
    uRough.Init_ConstantTexture(Spectrum(_uRough));
    vRough.Init_ConstantTexture(Spectrum(_vRough));
    index.Init_ConstantTexture(Spectrum(_index));
    type = MaterialType::Glass;
}

__bidevice__ void Material::Init_Glass(Texture Kr, Texture _Kt, Texture _uRough,
                                       Texture _vRough, Texture _index)
{
    Kt = _Kt;
    K = Kr;
    index = _index;
    vRough = _vRough;
    uRough = _uRough;
    type = MaterialType::Glass;
}


__bidevice__ void Material::Init_Glass(Spectrum Kr, Spectrum _Kt, Float _index){
    Init_Glass(Kr, _Kt, 0.0f, 0.0f, _index);
}

__bidevice__ void Material::ComputeScatteringFunctionsGlass(BSDF *bsdf, 
                                                            SurfaceInteraction *si, 
                                                            TransportMode mode, 
                                                            bool mLobes) const
{
    
    Float eta = index.Evaluate(si)[0];
    Float urough = uRough.Evaluate(si)[0];
    Float vrough = vRough.Evaluate(si)[0];
    Spectrum R = K.Evaluate(si);
    Spectrum T = Kt.Evaluate(si);
    
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