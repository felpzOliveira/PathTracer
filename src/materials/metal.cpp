#include <material.h>

__bidevice__ void Material::Init_Metal(Texture R, Texture etaI, Texture etaT, 
                                       Texture k)
{
    K = k;
    Kr = R;
    I = etaI;
    T = etaT;
    type = MaterialType::Metal;
}

__bidevice__ void Material::Init_Metal(Spectrum R, Float etaI, Float etaT, Float k){
    Kr.Init_ConstantTexture(R);
    K.Init_ConstantTexture(Spectrum(k));
    I.Init_ConstantTexture(Spectrum(etaI));
    T.Init_ConstantTexture(Spectrum(etaT));
    type = MaterialType::Metal;
}

__bidevice__ void Material::ComputeScatteringFunctionsMetal(BSDF *bsdf, 
                                                            SurfaceInteraction *si, 
                                                            TransportMode mode, 
                                                            bool mLobes) const
{
    Spectrum etaI = I.Evaluate(si);
    Spectrum etaT = T.Evaluate(si);
    Spectrum k    = K.Evaluate(si);
    Spectrum R    = Kr.Evaluate(si);
    
    if(R.IsBlack()) return;
    
    Fresnel fresnel;
    fresnel.Init_Conductor(etaI, etaT, k);
    
    BxDF bxdf(BxDFImpl::MicrofacetReflection);
    bxdf.Init_MicrofacetReflection(R, 0.001, 0.001, &fresnel, mode);
    bsdf->Push(&bxdf);
    
    //BxDF bxdf2(BxDFImpl::SpecularReflection);
    //bxdf2.Init_SpecularReflection(R, &fresnel);
    //bsdf->Push(&bxdf2);
    
    BxDF bxdfD(BxDFImpl::FresnelBlend);
    bxdfD.Init_FresnelBlend(R*0.1, R*0.1, 0.1, 0.1);
    bsdf->Push(&bxdfD);
}