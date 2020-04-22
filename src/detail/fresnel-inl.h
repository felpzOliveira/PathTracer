#if !defined(FRESNEL_H)
#error "Please include fresnel.h instead of this file"
#else

inline __bidevice__
float FrDieletric(float cosThetaI, float etaI, float etaT) {
    cosThetaI = Clamp(cosThetaI, -1, 1);
    // Potentially swap indices of refraction
    bool entering = cosThetaI > 0.f;
    if (!entering) {
        float aux = etaT;
        etaT = etaI;
        etaI = aux;
        cosThetaI = glm::abs(cosThetaI);
    }
    
    // Compute _cosThetaT_ using Snell's law
    float sinThetaI = glm::sqrt(glm::max((float)0, 1 - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;
    
    // Handle total internal reflection
    if (sinThetaT >= 1) return 1;
    float cosThetaT = glm::sqrt(glm::max((float)0, 1 - sinThetaT * sinThetaT));
    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
        ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
        ((etaI * cosThetaI) + (etaT * cosThetaT));
    return (Rparl * Rparl + Rperp * Rperp) / 2;
}


inline __bidevice__
Spectrum FrConductor(float cosThetaI, const Spectrum &etai,
                     const Spectrum &etat, const Spectrum &k)
{
    cosThetaI = Clamp(cosThetaI, -1, 1);
    Spectrum eta = etat / etai;
    Spectrum etak = k / etai;
    
    float cosThetaI2 = cosThetaI * cosThetaI;
    float sinThetaI2 = 1. - cosThetaI2;
    Spectrum eta2 = eta * eta;
    Spectrum etak2 = etak * etak;
    
    Spectrum t0 = eta2 - etak2 - sinThetaI2;
    Spectrum a2plusb2 = Sqrt(t0 * t0 + 4 * eta2 * etak2);
    Spectrum t1 = a2plusb2 + cosThetaI2;
    Spectrum a = Sqrt(0.5f * (a2plusb2 + t0));
    Spectrum t2 = (float)2 * cosThetaI * a;
    Spectrum Rs = (t1 - t2) / (t1 + t2);
    
    Spectrum t3 = cosThetaI2 * a2plusb2 + sinThetaI2 * sinThetaI2;
    Spectrum t4 = t2 * sinThetaI2;
    Spectrum Rp = Rs * (t3 - t4) / (t3 + t4);
    
    return 0.5 * (Rp + Rs);
}



inline __bidevice__
void fresnel_conductor_init(Fresnel *fresnel, Spectrum etaI, 
                            Spectrum etaT, Spectrum k)
{
    if(fresnel){
        fresnel->type = FresnelType::FresnelConductor;
        fresnel->sEtaI = etaI;
        fresnel->sEtaT = etaT;
        fresnel->k = k;
    }else{
        printf("Bad pointer for Fresnel::Conductor\n");
    }
}

inline __bidevice__
void fresnel_dieletric_init(Fresnel *fresnel, float etaI, float etaT){
    if(fresnel){
        fresnel->type = FresnelType::FresnelDieletric;
        fresnel->etaI = etaI;
        fresnel->etaT = etaT;
    }else{
        printf("Bad pointer for Fresnel::Dieletric\n");
    }
}

inline __bidevice__
Spectrum fresnel_evalutate(Fresnel *fresnel, float cosThetaI){
    if(fresnel){
        switch(fresnel->type){
            case FresnelType::FresnelDieletric: {
                return FrDieletric(cosThetaI, fresnel->etaI, fresnel->etaT);
            } break;
            
            case FresnelType::FresnelConductor: {
                return FrConductor(glm::abs(cosThetaI), fresnel->sEtaI,
                                   fresnel->sEtaT, fresnel->k);
            } break;
            
            default:{
                printf("Unkown frenel format\n");
            }
        }
    }else{
        printf("Bad pointer at Fresnel::Evaluate\n");
    }
    
    return Spectrum(1.f);
}

#endif