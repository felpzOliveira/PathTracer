#include <material.h>
#include <cutil.h>
#include <image_util.h>

__bidevice__ Float FrDieletric(Float cosThetaI, Float etaI, Float etaT){
    cosThetaI = Clamp(cosThetaI, -1, 1);
    // Potentially swap indices of refraction
    bool entering = cosThetaI > 0.f;
    if (!entering) {
        swap(etaI, etaT);
        cosThetaI = Absf(cosThetaI);
    }
    
    // Compute _cosThetaT_ using Snell's law
    Float sinThetaI = std::sqrt(Max((Float)0, 1 - cosThetaI * cosThetaI));
    Float sinThetaT = etaI / etaT * sinThetaI;
    
    // Handle total internal reflection
    if(sinThetaT >= 1) return 1;
    Float cosThetaT = std::sqrt(Max((Float)0, 1 - sinThetaT * sinThetaT));
    Float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
        ((etaT * cosThetaI) + (etaI * cosThetaT));
    Float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
        ((etaI * cosThetaI) + (etaT * cosThetaT));
    return (Rparl * Rparl + Rperp * Rperp) / 2;
}

__bidevice__ Spectrum FrConductor(Float cosThetaI, const Spectrum &etai,
                                  const Spectrum &etat, const Spectrum &k)
{
    cosThetaI = Clamp(cosThetaI, -1, 1);
    Spectrum eta = etat / etai;
    Spectrum etak = k / etai;
    
    Float cosThetaI2 = cosThetaI * cosThetaI;
    Float sinThetaI2 = 1. - cosThetaI2;
    Spectrum eta2 = eta * eta;
    Spectrum etak2 = etak * etak;
    
    Spectrum t0 = eta2 - etak2 - sinThetaI2;
    Spectrum a2plusb2 = Sqrt(t0 * t0 + 4 * eta2 * etak2);
    Spectrum t1 = a2plusb2 + cosThetaI2;
    Spectrum a = Sqrt(0.5f * (a2plusb2 + t0));
    Spectrum t2 = (Float)2 * cosThetaI * a;
    Spectrum Rs = (t1 - t2) / (t1 + t2);
    
    Spectrum t3 = cosThetaI2 * a2plusb2 + sinThetaI2 * sinThetaI2;
    Spectrum t4 = t2 * sinThetaI2;
    Spectrum Rp = Rs * (t3 - t4) / (t3 + t4);
    
    return 0.5 * (Rp + Rs);
}

__bidevice__ Spectrum Fresnel::EvaluateDieletric(Float cosI) const{
    return Spectrum(FrDieletric(cosI, etaI, etaT));
}

__bidevice__ Spectrum Fresnel::EvaluateConductor(Float cosI) const{
    return FrConductor(Absf(cosI), setaI, setaT, sk);
}

__bidevice__ Spectrum Fresnel::Evaluate(Float cosI) const{
    switch(type){
        case FresnelType::Dieletric: return EvaluateDieletric(cosI);
        case FresnelType::Conductor: return EvaluateConductor(cosI);
        case FresnelType::NoOp : return Spectrum(1.f);
        default:{
            printf("Unknown Fresnel type\n");
        }
    }
    
    return Spectrum(1.f);
}

__bidevice__ void Fresnel::Init_Conductor(Spectrum etaI, Spectrum etaT, Spectrum k){
    type = FresnelType::Conductor;
    setaI = etaI;
    setaT = etaT;
    sk = k;
}

__bidevice__ void Fresnel::Init_Dieletric(Float _etaI, Float _etaT){
    type = FresnelType::Dieletric;
    etaI = _etaI;
    etaT = _etaT;
}

__bidevice__ const char * GetMaterialName(MaterialType type){
    switch(type){
        case MaterialType::Matte: return "Matte";
        case MaterialType::Mirror: return "Mirror";
        case MaterialType::Plastic: return "Plastic";
        case MaterialType::Uber: return "Uber";
        case MaterialType::Glass: return "Glass";
        case MaterialType::Metal: return "Metal";
        case MaterialType::Translucent: return "Translucent";
        case MaterialType::Subsurface: return "Subsurface";
        case MaterialType::KdSubsurface: return "KdSubsurface";
        default: return "(None)";
    }
}

__bidevice__ void Material::ComputeScatteringFunctions(BSDF *bsdf, SurfaceInteraction *si, 
                                                       TransportMode mode, bool mLobes)
{
    if(material){
        switch(type){
            case MaterialType::Matte:{
                MatteMaterial *matte = (MatteMaterial *) material;
                matte->ComputeScatteringFunctions(bsdf, si, mode, mLobes);
            } break;
            
            case MaterialType::Mirror:{
                MirrorMaterial *mirror = (MirrorMaterial *) material;
                mirror->ComputeScatteringFunctions(bsdf, si, mode, mLobes);
            } break;
            
            case MaterialType::Glass:{
                GlassMaterial *glass = (GlassMaterial *) material;
                glass->ComputeScatteringFunctions(bsdf, si, mode, mLobes);
            } break;
            
            case MaterialType::Metal:{
                MetalMaterial *metal = (MetalMaterial *) material;
                metal->ComputeScatteringFunctions(bsdf, si, mode, mLobes);
            } break;
            
            case MaterialType::Translucent:{
                TranslucentMaterial *transl = (TranslucentMaterial *) material;
                transl->ComputeScatteringFunctions(bsdf, si, mode, mLobes);
            } break;
            
            case MaterialType::Plastic:{
                PlasticMaterial *plastic = (PlasticMaterial *) material;
                plastic->ComputeScatteringFunctions(bsdf, si, mode, mLobes);
            } break;
            
            case MaterialType::Uber:{
                UberMaterial *uber = (UberMaterial *) material;
                uber->ComputeScatteringFunctions(bsdf, si, mode, mLobes);
            } break;
            
            case MaterialType::Subsurface:{
                SubsurfaceMaterial *sub = (SubsurfaceMaterial *) material;
                sub->ComputeScatteringFunctions(bsdf, si, mode, mLobes, this);
            } break;
            
            case MaterialType::KdSubsurface:{
                KdSubsurfaceMaterial *kdsub = (KdSubsurfaceMaterial *) material;
                kdsub->ComputeScatteringFunctions(bsdf, si, mode, mLobes, this);
            } break;
            
            case MaterialType::PlayGround:{
                PlayGroundMaterial *pg = (PlayGroundMaterial *)material;
                pg->ComputeScatteringFunctions(bsdf, si, mode, mLobes, this);
            } break;
            
            default:{
                printf("Unknown material type\n");
            }
        }
    }else{
        printf("Invalid pointer in Material::Evaluate for %s\n", GetMaterialName(type));
    }
}
