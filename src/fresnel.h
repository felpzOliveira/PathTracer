#if !defined(FRESNEL_H)
#define FRESNEL_H
#include <spectrum.h>
#include <glm/glm.hpp>

__host__ __device__ inline float fclamp(float a, float b, float x){
    if(x < a) return a;
    if(x > b) return b;
    return x;
}

__host__ __device__ inline glm::vec3 Reflect(const glm::vec3 &wo, const glm::vec3 &n) {
    return -wo + 2.0f * glm::dot(wo, n) * n;
}

__host__ __device__ inline bool Refract(const glm::vec3 &wi, const glm::vec3 &n, 
                                        float eta, glm::vec3 *wt) 
{
    // Compute cos theta_t using Snell's law
    float cosThetaI = glm::dot(n, wi);
    float sin2ThetaI = glm::max(float(0), float(1 - cosThetaI * cosThetaI));
    
    float sin2ThetaT = eta * eta * sin2ThetaI;
    // Handle total internal reflection for transmission
    if (sin2ThetaT >= 1.0f) return false;
    float cosThetaT = glm::sqrt(1 - sin2ThetaT);
    *wt = eta * -wi + (eta * cosThetaI - cosThetaT) * n;
    return true;
}


typedef enum{
    FresnelDieletric,
    FresnelConductor
}FresnelType;

struct Fresnel{
    Spectrum sEtaI, sEtaT, k;
    float etaI, etaT;
    FresnelType type;
};

__host__ __device__ inline
float FrDielectric(float cosThetaI, float etaI, float etaT) {
    cosThetaI = fclamp(-1.0f, 1.0f, cosThetaI);
    // Potentially swap indices of refraction
    bool entering = cosThetaI > 0.f;
    if (!entering) {
        float aux = etaI;
        etaI = etaT;
        etaT = aux;
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

__host__ __device__ inline
Spectrum FrConductor(float cosThetaI, const Spectrum &etai,
                     const Spectrum &etat, const Spectrum &k) 
{
    cosThetaI = fclamp(-1.0f, 1.0f, cosThetaI);
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

__host__ __device__ inline 
void Fresnel_init(Fresnel *fresnel, float etaI, float etaT){
    fresnel->type = FresnelType::FresnelDieletric;
    fresnel->etaI = etaI;
    fresnel->etaT = etaT;
}

__host__ __device__ inline 
void Fresnel_init(Fresnel *fresnel, Spectrum etaI, Spectrum etaT, Spectrum k){
    fresnel->type = FresnelType::FresnelConductor;
    fresnel->sEtaI = etaI;
    fresnel->sEtaT = etaT;
    fresnel->k = k;
}

__host__ __device__ inline Spectrum Fresnel_Conductor_eval(Fresnel *fresnel,
                                                           float cosThetaI)
{
    return FrConductor(glm::abs(cosThetaI), fresnel->sEtaI,
                       fresnel->sEtaT, fresnel->k);
}

__host__ __device__ inline Spectrum Fresnel_Dieletric_eval(Fresnel *fresnel,
                                                           float cosThetaI)
{
    return FrDielectric(cosThetaI, fresnel->etaI, fresnel->etaT);
}

__host__ __device__ inline Spectrum Fresnel_evaluate(Fresnel *fresnel,
                                                     float cosThetaI)
{
    Spectrum r(0.0f);
    if(fresnel){
        switch(fresnel->type){
            case FresnelType::FresnelDieletric:{
                r = Fresnel_Dieletric_eval(fresnel, cosThetaI);
            } break;
            
            case FresnelType::FresnelConductor:{
                r = Fresnel_Conductor_eval(fresnel, cosThetaI);
            } break;
            
            default:{ //no op
                r = Spectrum(1.0f);
                printf("Fresnel No OP\n");
            }
        }
    }
    
    return r;
}

#endif