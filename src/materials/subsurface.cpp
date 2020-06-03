#include <material.h>
#include <bssrdf.h>

__bidevice__ SubsurfaceMaterial::SubsurfaceMaterial(Texture<Spectrum> *kr, 
                                                    Texture<Spectrum> *kt,
                                                    Texture<Spectrum> *sa, 
                                                    Texture<Spectrum> *ss,
                                                    Texture<Float> *urough, 
                                                    Texture<Float> *vrough,
                                                    Float g, Float e, Float s)
: Kr(kr), Kt(kt), sigma_a(sa), sigma_s(ss), uRough(urough), vRough(vrough), eta(e)
{
    scale = s;
    table = new BSSRDFTable(100, 64);
    ComputeBeamDiffusionBSSRDF(g, eta, table);
}


__bidevice__ void SubsurfaceMaterial::ComputeScatteringFunctions(BSDF *bsdf, 
                                                                 SurfaceInteraction *si, 
                                                                 TransportMode mode, 
                                                                 bool mLobes,
                                                                 Material *sourceMat) const
{
    Spectrum R = Clamp(Kr->Evaluate(si));
    Spectrum T = Clamp(Kt->Evaluate(si));
    
    Float urough = uRough->Evaluate(si);
    Float vrough = vRough->Evaluate(si);
    
    if(R.IsBlack() && T.IsBlack()) return;
    
    bool isSpecular = IsZero(urough) && IsZero(vrough);
    
    if(isSpecular && mLobes){
        BxDF bxdf(BxDFImpl::FresnelSpecular);
        bxdf.Init_FresnelSpecular(R, T, 1.f, eta, mode);
        bsdf->Push(&bxdf);
    }else{
        if(!R.IsBlack()){
            Fresnel fresnel;
            fresnel.Init_Dieletric(1.f, eta);
            if(isSpecular){
                BxDF bxdf(BxDFImpl::SpecularReflection);
                bxdf.Init_SpecularReflection(R, &fresnel);
                bsdf->Push(&bxdf);
            }else{
                BxDF bxdf(BxDFImpl::MicrofacetReflection);
                bxdf.Init_MicrofacetReflection(R, urough, vrough, &fresnel, mode, false);
                bsdf->Push(&bxdf);
            }
        }
        
        if(!T.IsBlack()){
            if(isSpecular){
                BxDF bxdf(BxDFImpl::SpecularTransmission);
                bxdf.Init_SpecularTransmission(T, 1.f, eta, mode);
                bsdf->Push(&bxdf);
            }else{
                BxDF bxdf(BxDFImpl::MicrofacetTransmission);
                bxdf.Init_MicrofacetTransmission(T, 1.f, eta, urough, vrough, mode, false);
                bsdf->Push(&bxdf);
            }
        }
    }
    
    Spectrum sig_a = scale * sigma_a->Evaluate(si);
    Spectrum sig_s = scale * sigma_s->Evaluate(si);
    SeparableBSSRDF bssrdf(*si, eta, sourceMat, mode);
    bssrdf.Init_TabulatedBSSRDF(sig_a, sig_s, table);
    bsdf->PushBSSRDF(&bssrdf);
}