#include <material.h>

__bidevice__ KdSubsurfaceMaterial::KdSubsurfaceMaterial(Texture<Spectrum> *kd, 
                                                        Texture<Spectrum> *kr,
                                                        Texture<Spectrum> *kt, 
                                                        Texture<Spectrum> *mffp,
                                                        Texture<Float> *urough, 
                                                        Texture<Float> *vrough,
                                                        Texture<Float> *map, Float e, 
                                                        Float s)
: Kd(kd), Kt(kt), Kr(kr), mfp(mffp), uRough(urough), vRough(vrough), eta(e), bumpMap(map),
scale(s)
{
    has_bump = 1;
    table = new BSSRDFTable(100, 64);
    //g = 0
    ComputeBeamDiffusionBSSRDF(0, eta, table);
}

__bidevice__ KdSubsurfaceMaterial::KdSubsurfaceMaterial(Texture<Spectrum> *kd, 
                                                        Texture<Spectrum> *kr,
                                                        Texture<Spectrum> *kt, 
                                                        Texture<Spectrum> *mffp,
                                                        Texture<Float> *urough, 
                                                        Texture<Float> *vrough,
                                                        Float e, Float s)
: Kd(kd), Kt(kt), Kr(kr), mfp(mffp), uRough(urough), vRough(vrough), eta(e), scale(s)
{
    has_bump = 0;
    table = new BSSRDFTable(100, 64);
    //g = 0
    ComputeBeamDiffusionBSSRDF(0, eta, table);
}

__bidevice__ void KdSubsurfaceMaterial::ComputeScatteringFunctions(BSDF *bsdf, 
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
    
    Spectrum kd = Kd->Evaluate(si);
    Spectrum mfree = scale * mfp->Evaluate(si);
    Spectrum sig_a, sig_s;
    SubsurfaceFromDiffuse(table, kd, mfree, &sig_a, &sig_s);
    SeparableBSSRDF bssrdf(*si, eta, sourceMat, mode);
    bssrdf.Init_TabulatedBSSRDF(sig_a, sig_s, table);
    bsdf->PushBSSRDF(&bssrdf);
}