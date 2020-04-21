#if !defined(BSDF_H)
#define BSDF_H
#include <onb.h>
#include <types.h>
#include <spectrum.h>
#include <fresnel.h>

__host__ __device__ inline Spectrum BxDF_f(BxDF *bxdf, glm::vec3 &wo, glm::vec3 &wi);
__host__ __device__ inline Spectrum BxDF_Sample_f(BxDF *bxdf, glm::vec3 &wo, glm::vec3 *wi,
                                                  glm::vec2 &u, float *pdf,
                                                  BxDFType *sampledType);
//TODO: Build Orthonormal basis


static const float ShadowEpsilon = 0.0001f;
static const float Pi = 3.14159265358979323846;
static const float InvPi = 0.31830988618379067154;
static const float Inv2Pi = 0.15915494309189533577;
static const float Inv4Pi = 0.07957747154594766788;
static const float PiOver2 = 1.57079632679489661923;
static const float PiOver4 = 0.78539816339744830961;
static const float Sqrt2 = 1.41421356237309504880;
static const float OneMinusEpsilon = 0.99999994;

#define ABS(x) ((x) > 0.0001 ? (x) : -(x))

// UTILITIES
__host__ __device__ inline glm::vec2 ConcentricSampleDisk(const glm::vec2 &u) {
    // Map uniform random numbers to [-1,1]
    glm::vec2 uOffset = 2.f * u - glm::vec2(1, 1); 
    
    // Handle degeneracy at the origin
    if (ABS(uOffset.x) < 0.0001 && ABS(uOffset.y) < 0.0001) return glm::vec2(0, 0); 
    
    // Apply concentric mapping to point
    float theta, r;
    if (glm::abs(uOffset.x) > glm::abs(uOffset.y)) {
        r = uOffset.x;
        theta = PiOver4 * (uOffset.y / uOffset.x);
    } else {
        r = uOffset.y;
        theta = PiOver2 - PiOver4 * (uOffset.x / uOffset.y);
    }   
    return r * glm::vec2(glm::cos(theta), glm::sin(theta));
}

__host__ __device__ inline glm::vec3 CosineSampleHemisphere(const glm::vec2 &u) {
    glm::vec2 d = ConcentricSampleDisk(u);
    float z = glm::sqrt(glm::max((float)0, 1 - d.x * d.x - d.y * d.y));
    return glm::vec3(d.x, d.y, z); 
}

////////////////////////////////////////////////////////////////////////////////////

// GENERIC
__host__ __device__ inline 
float BxDF_Gen_pdf(glm::vec3 &wo, glm::vec3 &wi){
    return SameHemisphere(wo, wi) ? AbsCosTheta(wi) * InvPi : 0;
}


__host__ __device__ inline 
Spectrum BxDF_Gen_Sample_f(BxDF *bxdf, glm::vec3 &wo, glm::vec3 *wi,
                           glm::vec2 &u, float *pdf, BxDFType *sampledType)
{
    // Cosine-sample the hemisphere, flipping the direction if necessary
    *wi = CosineSampleHemisphere(u); 
    if (wo.z < 0) wi->z *= -1;
    *pdf = BxDF_Gen_pdf(wo, *wi);
    return BxDF_f(bxdf, wo, *wi);
}

////////////////////////////////////////////////////////////////////////////////////

// SPECULAR REFLECTION
__host__ inline void BxDF_SpecularReflection_init(BxDF *bxdf, Spectrum R,
                                                  Fresnel fresnel)
{
    bxdf->self.type = BxDFType(BSDF_REFLECTION | BSDF_SPECULAR);
    bxdf->self.impl = SpecularReflectionBxDF;
    bxdf->self.id = 0;
    bxdf->R = R;
    bxdf->fresnel = fresnel;
}

__host__ __device__ inline Spectrum BxDF_SpecularReflection_f(BxDF *bxdf,
                                                              glm::vec3 &wo,
                                                              glm::vec3 &wi)
{
    return Spectrum(0.f);
}

__host__ __device__ inline float BxDF_SpecularReflection_pdf(BxDF *bxdf,
                                                             glm::vec3 &wo,
                                                             glm::vec3 &wi)
{
    return 0.0f;
}

__host__ __device__ inline
Spectrum BxDF_SpecularReflection_Sample_f(BxDF *bxdf, glm::vec3 &wo, glm::vec3 *wi,
                                          glm::vec2 &u, float *pdf,
                                          BxDFType *sampledType)
{
    *pdf = 1.0f; //allways this direction
    *wi = glm::vec3(-wo.x, -wo.y, wo.z); //reflection when n = (0,0,1)
    //This Value comes from Page 524 it is the Fresnel * Dirac / ModuleOf(costhetai)
    return Spectrum(Fresnel_evaluate(&bxdf->fresnel, CosTheta(*wi))
                    * bxdf->R / AbsCosTheta(*wi));
}

////////////////////////////////////////////////////////////////////////////////////

// SPECULAR TRANSMISSION
__host__ __device__ inline
void BxDF_SpecularTransmission_init(BxDF *bxdf, Spectrum T, float etaA, 
                                    float etaB)
{
    bxdf->self.type = BxDFType(BSDF_TRANSMISSION | BSDF_SPECULAR);
    bxdf->self.impl = SpecularTransmissionBxDF;
    bxdf->self.id = 0;
    bxdf->T = T;
    bxdf->etaA = etaA;
    bxdf->etaB = etaB;
    Fresnel_init(&bxdf->fresnel, etaA, etaB);
}


__host__ __device__ inline Spectrum BxDF_SpecularTransmission_f(BxDF *bxdf,
                                                                glm::vec3 &wo,
                                                                glm::vec3 &wi)
{
    return Spectrum(0.f);
}

__host__ __device__ inline float BxDF_SpecularTransmission_pdf(BxDF *bxdf,
                                                               glm::vec3 &wo,
                                                               glm::vec3 &wi)
{
    return 0.0f;
}


__host__ __device__ inline
Spectrum BxDF_SpecularTransmission_Sample_f(BxDF *bxdf, glm::vec3 &wo, glm::vec3 *wi,
                                            glm::vec2 &u, float *pdf,
                                            BxDFType *sampledType)
{
    //check if we are going in or out
    bool entering = CosTheta(wo) > 0;
    float etaI = entering ? bxdf->etaA : bxdf->etaB;
    float etaT = entering ? bxdf->etaB : bxdf->etaA;
    
    //compute refracted direction, if not possible return 0
    if(!Refract(wo, Faceforward(glm::vec3(0,0,1), wo), etaI / etaT, wi)){
        return 0.0f;
    }
    
    *pdf = 1.0f; //allways this direction
    Spectrum fre = Fresnel_evaluate(&bxdf->fresnel, CosTheta(*wi));
    Spectrum ft = bxdf->T * (Spectrum(1.0f) - fre);
    
    if(1){
        ft *= (etaI * etaI)/(etaT * etaT);
    }
    
    return ft / AbsCosTheta(*wi);
}

////////////////////////////////////////////////////////////////////////////////////

// LAMBERTIAN REFLECTION
__host__ __device__ inline 
void BxDF_LambertianReflection_init(BxDF *bxdf, Spectrum R){
    bxdf->self.type = BxDFType(BSDF_REFLECTION | BSDF_DIFFUSE);
    bxdf->self.impl = LambertianReflectionBxDF;
    bxdf->self.id = 0;
    bxdf->R = R;
}

__host__ __device__ inline 
Spectrum BxDF_LambertianReflection_f(BxDF *bxdf, glm::vec3 &wo,
                                     glm::vec3 &wi)
{
    return bxdf->R * InvPi;
}

__host__ __device__ inline 
Spectrum BxDF_LambertianReflection_Sample_f(BxDF *bxdf, glm::vec3 &wo, glm::vec3 *wi,
                                            glm::vec2 &u, float *pdf,
                                            BxDFType *sampledType)
{
    return BxDF_Gen_Sample_f(bxdf, wo, wi, u, pdf, sampledType);
} 

__host__ __device__ inline float BxDF_LambertianReflection_pdf(BxDF *bxdf, glm::vec3 &wo,
                                                               glm::vec3 &wi)
{
    return BxDF_Gen_pdf(wo, wi);
}

////////////////////////////////////////////////////////////////////////////////////

// LAMBERTIAN TRANSMISSION
__host__ __device__ inline
void BxDF_LambertianTransmission_init(BxDF *bxdf, Spectrum T){
    bxdf->self.type = BxDFType(BSDF_TRANSMISSION | BSDF_DIFFUSE);
    bxdf->self.impl = LambertianTransmissionBxDF;
    bxdf->self.id = 0;
    bxdf->T = T;
}

__host__ __device__ inline 
Spectrum BxDF_LambertianTransmission_f(BxDF *bxdf, glm::vec3 &wo,
                                       glm::vec3 &wi)
{
    return bxdf->T * InvPi;
}

__host__ __device__ inline float BxDF_LambertianTransmission_pdf(BxDF *bxdf, glm::vec3 &wo,
                                                                 glm::vec3 &wi)
{
    return !SameHemisphere(wo, wi) ? AbsCosTheta(wi) * InvPi : 0.0f;
}

__host__ __device__ inline 
Spectrum BxDF_LambertianTransmission_Sample_f(BxDF *bxdf, glm::vec3 &wo, glm::vec3 *wi,
                                              glm::vec2 &u, float *pdf,
                                              BxDFType *sampledType)
{
    *wi = CosineSampleHemisphere(u);
    if (wo.z > 0) wi->z *= -1;
    *pdf = BxDF_LambertianTransmission_pdf(bxdf, wo, *wi);
    return BxDF_LambertianTransmission_f(bxdf, wo, *wi);
} 


////////////////////////////////////////////////////////////////////////////////////
// OREN-NAYAR
__host__ __device__ inline 
void BxDF_OrenNayar_init(BxDF *bxdf, Spectrum R, float sigma){
    bxdf->self.type = BxDFType(BSDF_REFLECTION | BSDF_DIFFUSE);
    bxdf->self.impl = OrenNayarBxDF;
    bxdf->self.id = 0;
    bxdf->R = R;
    
    sigma = glm::radians(sigma);
    float sigma2 = sigma * sigma;
    bxdf->A = 1.f - (sigma2 / (2.f * (sigma2 + 0.33f)));
    bxdf->B = 0.45f * sigma2 / (sigma2 + 0.09f);
}

__host__ __device__ inline Spectrum BxDF_OrenNayar_f(BxDF *bxdf, glm::vec3 &wo,
                                                     glm::vec3 &wi)
{
    float sinThetaI = SinTheta(wi);
    float sinThetaO = SinTheta(wo);
    float maxCos = 0.0f;
    if(sinThetaI > 1e-4 && sinThetaO > 1e-4){
        float sinPhiI = SinPhi(wi), cosPhiI = CosPhi(wi);
        float sinPhiO = SinPhi(wo), cosPhiO = CosPhi(wo);
        float dCos = cosPhiI * cosPhiO + sinPhiI *sinPhiO;
        maxCos = glm::max(0.0f, dCos);
    }
    
    float sinAlpha, tanBeta;
    if(AbsCosTheta(wi) > AbsCosTheta(wo)){
        sinAlpha = sinThetaO;
        tanBeta = sinThetaI / AbsCosTheta(wi);
    }else{
        sinAlpha = sinThetaI;
        tanBeta = sinThetaO / AbsCosTheta(wo);
    }
    
    return bxdf->R * InvPi * (bxdf->A + bxdf->B * maxCos * sinAlpha * tanBeta);
}


__host__ __device__ inline 
Spectrum BxDF_OrenNayar_Sample_f(BxDF *bxdf, glm::vec3 &wo, glm::vec3 *wi,
                                 glm::vec2 &u, float *pdf, BxDFType *sampledType)
{
    return BxDF_Gen_Sample_f(bxdf, wo, wi, u, pdf, sampledType);
} 

__host__ __device__ inline float BxDF_OrenNayar_pdf(BxDF *bxdf, glm::vec3 &wo,
                                                    glm::vec3 &wi)
{
    return BxDF_Gen_pdf(wo, wi);
}

////////////////////////////////////////////////////////////////////////////////////

// CALLABLE
__host__ __device__ inline Spectrum BxDF_f(BxDF *bxdf, glm::vec3 &wo, glm::vec3 &wi){
    Spectrum f(0.0f);
    if(bxdf){
        switch(bxdf->self.impl){
            case LambertianReflectionBxDF: {
                f = BxDF_LambertianReflection_f(bxdf, wo, wi);
            } break;
            
            case LambertianTransmissionBxDF:{
                f = BxDF_LambertianTransmission_f(bxdf, wo, wi);
            }break;
            
            case OrenNayarBxDF:{
                f = BxDF_OrenNayar_f(bxdf, wo, wi);
            } break;
            
            case SpecularReflectionBxDF: {
                f = BxDF_SpecularReflection_f(bxdf, wo, wi);
            } break;
            
            case SpecularTransmissionBxDF: {
                f = BxDF_SpecularTransmission_f(bxdf, wo, wi);
            };
            
            default: {}
        }
    }
    
    return f;
}

__host__ __device__ inline Spectrum BxDF_Sample_f(BxDF *bxdf, glm::vec3 &wo,
                                                  glm::vec3 *wi, glm::vec2 &u,
                                                  float *pdf,BxDFType *sampledType)
{
    Spectrum f(0.0f);
    *pdf = 0.0f;
    if(bxdf){
        switch(bxdf->self.impl){
            case LambertianReflectionBxDF:{
                f = BxDF_LambertianReflection_Sample_f(bxdf, wo, wi, u, pdf,
                                                       sampledType);
            } break;
            
            case LambertianTransmissionBxDF:{
                f = BxDF_LambertianTransmission_Sample_f(bxdf, wo, wi, u, pdf,
                                                         sampledType);
            } break;
            
            case OrenNayarBxDF: {
                f = BxDF_OrenNayar_Sample_f(bxdf, wo, wi, u, pdf, sampledType);
            } break;
            
            case SpecularReflectionBxDF: {
                f = BxDF_SpecularReflection_Sample_f(bxdf, wo, wi, u, pdf,
                                                     sampledType);
            } break;
            
            case SpecularTransmissionBxDF: {
                f = BxDF_SpecularTransmission_Sample_f(bxdf, wo, wi, u, pdf,
                                                       sampledType);
            } break;
            
            default: {}
        }
    }
    
    return f;
}

__host__ __device__ inline float BxDF_pdf(BxDF *bxdf, glm::vec3 &wo, glm::vec3 &wi){
    float pdf = 0.0f;
    if(bxdf){
        switch(bxdf->self.impl){
            case LambertianReflectionBxDF:{
                pdf = BxDF_LambertianReflection_pdf(bxdf, wo, wi);
            } break;
            
            case LambertianTransmissionBxDF:{
                pdf = BxDF_LambertianTransmission_pdf(bxdf, wo, wi);
            } break;
            
            case OrenNayarBxDF: {
                pdf = BxDF_OrenNayar_pdf(bxdf, wo, wi);
            } break;
            
            case SpecularReflectionBxDF: {
                pdf = BxDF_SpecularReflection_pdf(bxdf, wo, wi);
            } break;
            
            case SpecularTransmissionBxDF: {
                pdf = BxDF_SpecularTransmission_pdf(bxdf, wo, wi);
            } break;
            
            default: {}
        }
    }
    return pdf;
}

__host__ __device__ inline Spectrum BSDF_Sample_f(BxDF *bxdf, glm::vec3 &woW,
                                                  glm::vec3 normal, glm::vec3 *wiW,
                                                  glm::vec2 &u, float *pdf)
{
    Onb uvw;
    onb_from_w(&uvw, normal);
    glm::vec3 wi, wo = onb_world_to_local(&uvw, woW);
    if (ABS(wo.z) < 0.0001) {
        return 0.;
    }
    
    *pdf = 0;
    
    Spectrum f = BxDF_Sample_f(bxdf, wo, &wi, u, pdf, nullptr);
    if (ABS(*pdf) < 0.0001) {
        return 0.;
    }
    
    *wiW = onb_local_to_world(&uvw, wi);
    
    //TODO: Multiple BxDF need to compute f and pdf on them all
    
    return f;
}

#endif