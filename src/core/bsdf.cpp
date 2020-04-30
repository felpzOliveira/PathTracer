#include <bsdf.h>

static __bidevice__ 
glm::vec3 Reflect(const glm::vec3 &wo, const glm::vec3 &n) {
    return -wo + 2.0f * glm::dot(wo, n) * n;
}

static __bidevice__ 
bool Refract(const glm::vec3 &wi, const glm::vec3 &n, float eta, glm::vec3 *wt) {
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

__bidevice__
bool BxDF_Matches(BxDF *bxdf, BxDFType type){
    return (bxdf->self.type & type) == bxdf->self.type;
}

/***************************************************************
 > Init functions.
****************************************************************/
__bidevice__
void BxDF_LambertianReflection_init(BxDF *bxdf, Spectrum R){
    if(bxdf){
        bxdf->self.type = BxDFType(BSDF_REFLECTION | BSDF_DIFFUSE);
        bxdf->self.model = LambertianReflectionBxDF;
        bxdf->S = R;
    }else{
        printf("Bad pointer at Lambertian Reflector init\n");
    }
}

__bidevice__
void BxDF_LambertianTransmission_init(BxDF *bxdf, Spectrum T){
    if(bxdf){
        bxdf->self.type = BxDFType(BSDF_TRANSMISSION | BSDF_DIFFUSE);
        bxdf->self.model = LambertianTransmissionBxDF;
        bxdf->S = T;
    }else{
        printf("Bad pointer at Lambertian Reflector init\n");
    }
}


__bidevice__
void BxDF_OrenNayar_init(BxDF *bxdf, Spectrum R, float sigma){
    if(bxdf){
        bxdf->self.type = BxDFType(BSDF_REFLECTION | BSDF_DIFFUSE);
        bxdf->self.model = OrenNayarBxDF;
        bxdf->S = R;
        sigma = glm::radians(sigma);
        float sigma2 = sigma * sigma;
        bxdf->OrenNayar.A = 1.f - (sigma2 / (2.f * (sigma2 + 0.33f)));
        bxdf->OrenNayar.B = 0.45f * sigma2 / (sigma2 + 0.09f);
    }else{
        printf("Bad pointer at OrenNayar init!\n");
    }
}

__bidevice__
void BxDF_SpecularReflection_init(BxDF *bxdf, Spectrum R, Fresnel *fresnel){
    if(bxdf && fresnel){
        bxdf->self.type = BxDFType(BSDF_REFLECTION | BSDF_SPECULAR);
        bxdf->self.model = SpecularReflectionBxDF;
        bxdf->Specular.fresnel = *fresnel;
        bxdf->S = R;
    }else{
        printf("Bad pointer at SpecularReflector init!\n");
    }
}

__bidevice__
void BxDF_SpecularTransmission_init(BxDF *bxdf, Spectrum T, float A, float B){
    if(bxdf){
        bxdf->self.type = BxDFType(BSDF_TRANSMISSION | BSDF_SPECULAR);
        bxdf->self.model = SpecularTransmissionBxDF;
        bxdf->S = T;
        bxdf->Specular.etaA = A;
        bxdf->Specular.etaB = B;
        fresnel_dieletric_init(&bxdf->Specular.fresnel, A, B);
    }else{
        printf("bad pointer at SpecularTransmission init!\n");
    }
}

__bidevice__
void BxDF_FresnelSpecular_init(BxDF *bxdf, Spectrum R, Spectrum T, 
                               float A, float B)
{
    bxdf->self.type = BxDFType(BSDF_REFLECTION | BSDF_TRANSMISSION | BSDF_SPECULAR);
    bxdf->self.model = FresnelSpecularBxDF;
    bxdf->S = T;
    bxdf->R = R;
    bxdf->Specular.etaA = A;
    bxdf->Specular.etaB = B;
}

__bidevice__
void BxDF_Microfacet_Reflection_init(BxDF *bxdf, Spectrum R,
                                     MicrofacetDistribution *dist,
                                     Fresnel *fresnel)
{
    if(bxdf && dist && fresnel){
        bxdf->self.type = BxDFType(BSDF_REFLECTION | BSDF_GLOSSY);
        bxdf->self.model = MicrofacetReflectionBxDF;
        bxdf->Microfacet.fresnel = *fresnel;
        bxdf->Microfacet.dist = *dist;
        bxdf->S = R;
    }else{
        printf("Bad pointer at MicrofacetReflection init\n");
    }
}

__bidevice__
void BxDF_Microfacet_Transmission_init(BxDF *bxdf, Spectrum T,
                                       MicrofacetDistribution *dist,
                                       float etaA, float etaB)
{
    if(bxdf && dist){
        bxdf->self.type = BxDFType(BSDF_TRANSMISSION | BSDF_GLOSSY);
        bxdf->self.model = MicrofacetTransmissionBxDF;
        bxdf->Microfacet.dist = *dist;
        bxdf->Microfacet.etaA = etaA;
        bxdf->Microfacet.etaB = etaB;
        bxdf->S = T;
        fresnel_dieletric_init(&bxdf->Microfacet.fresnel, etaA, etaB);
    }else{
        printf("Bad pointer at MicrofacetTransmission init\n");
    }
}

/***************************************************************
 > Compute pdf(wo, wi).
****************************************************************/
__bidevice__
float BxDF_Gen_Pdf(BxDF *bxdf, glm::vec3 wo, glm::vec3 wi){
    return SameHemisphere(wo, wi) ? AbsCosTheta(wi) * InvPi : 0;
}

__bidevice__
float BxDF_LambertianReflection_Pdf(BxDF *bxdf, glm::vec3 wo, glm::vec3 wi){
    return BxDF_Gen_Pdf(bxdf, wo, wi);
}

__bidevice__
float BxDF_LambertianTransmission_Pdf(BxDF *bxdf, glm::vec3 wo, glm::vec3 wi){
    return !SameHemisphere(wo, wi) ? AbsCosTheta(wi) * InvPi : 0.0f;
}

__bidevice__
float BxDF_OrenNayar_Pdf(BxDF *bxdf, glm::vec3 wo, glm::vec3 wi){
    return BxDF_Gen_Pdf(bxdf, wo, wi);
}

__bidevice__
float BxDF_SpecularReflection_Pdf(BxDF *bxdf, glm::vec3 wo, glm::vec3 wi){
    return 0.f;
}

__bidevice__
float BxDF_SpecularTransmission_Pdf(BxDF *bxdf, glm::vec3 wo, glm::vec3 wi){
    return 0.f;
}

__bidevice__
float BxDF_FresnelSpecular_Pdf(BxDF *bxdf, glm::vec3 wo, glm::vec3 wi){
    return 0.0f;
}

__bidevice__
float BxDF_MicrofacetReflection_Pdf(BxDF *bxdf, glm::vec3 wo, glm::vec3 wi){
    if (!SameHemisphere(wo, wi)) return 0;
    glm::vec3 wh = glm::normalize(wo + wi);
    float pdf = MicrofacetDistribution_Pdf(&bxdf->Microfacet.dist, wo, wh);
    return pdf / (4.f * glm::dot(wo, wh));
}

__bidevice__
float BxDF_MicrofacetTransmission_Pdf(BxDF *bxdf, glm::vec3 wo, glm::vec3 wi){
    if (SameHemisphere(wo, wi)) return 0;
    
    float etaA = bxdf->Microfacet.etaA;
    float etaB = bxdf->Microfacet.etaB;
    float eta = CosTheta(wo) > 0 ? (etaB / etaA) : (etaA / etaB);
    glm::vec3 wh = glm::normalize(wo + wi * eta);
    
    float sqrtDenom = glm::dot(wo, wh) + eta * glm::dot(wi, wh);
    float pdf = MicrofacetDistribution_Pdf(&bxdf->Microfacet.dist, wo, wh);
    
    float dwh_dwi =
        glm::abs((eta * eta * glm::dot(wi, wh)) / (sqrtDenom * sqrtDenom));
    return pdf * dwh_dwi;
}

__bidevice__ 
float BxDF_Pdf(BxDF *bxdf, glm::vec3 &wo, glm::vec3 &wi){
    float pdf = 0.0f;
    if(bxdf){
        switch(bxdf->self.model){
            case LambertianReflectionBxDF:{
                pdf = BxDF_LambertianReflection_Pdf(bxdf, wo, wi);
            } break;
            
            case LambertianTransmissionBxDF:{
                pdf = BxDF_LambertianTransmission_Pdf(bxdf, wo, wi);
            } break;
            
            case OrenNayarBxDF: {
                pdf = BxDF_OrenNayar_Pdf(bxdf, wo, wi);
            } break;
            
            case SpecularReflectionBxDF:{
                pdf = BxDF_SpecularReflection_Pdf(bxdf, wo, wi);
            } break;
            
            case SpecularTransmissionBxDF:{
                pdf = BxDF_SpecularTransmission_Pdf(bxdf, wo, wi);
            } break;
            
            case FresnelSpecularBxDF:{
                return BxDF_FresnelSpecular_Pdf(bxdf, wo, wi);
            } break;
            
            case MicrofacetReflectionBxDF:{
                return BxDF_MicrofacetReflection_Pdf(bxdf, wo, wi);
            } break;
            
            case MicrofacetTransmissionBxDF:{
                return BxDF_MicrofacetTransmission_Pdf(bxdf, wo, wi);
            } break;
            
            default: {}
        }
    }
    return pdf;
}


/***************************************************************
 > Compute f(wo, wi).
****************************************************************/
__bidevice__
Spectrum BxDF_LambertianReflection_f(BxDF *bxdf, glm::vec3 wo, glm::vec3 wi){
    return bxdf->S * InvPi;
}

__bidevice__
Spectrum BxDF_LambertianTransmission_f(BxDF *bxdf, glm::vec3 wo, glm::vec3 wi){
    return bxdf->S * InvPi;
}

__bidevice__
Spectrum BxDF_OrenNayar_f(BxDF *bxdf, glm::vec3 wo, glm::vec3 wi){
    float sinThetaI = SinTheta(wi);
    float sinThetaO = SinTheta(wo);
    float maxCos = 0.0f;
    float A = bxdf->OrenNayar.A;
    float B = bxdf->OrenNayar.B;
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
    
    return bxdf->S * InvPi * (A + B * maxCos * sinAlpha * tanBeta);
}

__bidevice__
Spectrum BxDF_SpecularReflection_f(BxDF *bxdf, glm::vec3 wo, glm::vec3 wi){
    return Spectrum(0.f);
}

__bidevice__
Spectrum BxDF_SpecularTransmission_f(BxDF *bxdf, glm::vec3 wo, glm::vec3 wi){
    return Spectrum(0.f);
}

__bidevice__
Spectrum BxDF_FresnelSpecular_f(BxDF *bxdf, glm::vec3 wo, glm::vec3 wi){
    return Spectrum(0.f);
}

__bidevice__
Spectrum BxDF_MicrofacetReflection_f(BxDF *bxdf, glm::vec3 wo, glm::vec3 wi){
    float cosThetaO = AbsCosTheta(wo), cosThetaI = AbsCosTheta(wi);
    glm::vec3 wh = wi + wo;
    // Handle degenerate cases for microfacet reflection
    if (cosThetaI == 0 || cosThetaO == 0) return Spectrum(0.);
    if (wh.x == 0 && wh.y == 0 && wh.z == 0) return Spectrum(0.);
    wh = glm::normalize(wh);
    // For the Fresnel call, make sure that wh is in the same hemisphere
    // as the surface normal, so that TIR is handled correctly.
    glm::vec3 v = Faceforward(wh, glm::vec3(0,0,1));
    Spectrum F = fresnel_evalutate(&bxdf->Microfacet.fresnel, glm::dot(wi, v));
    float Dwh = MicrofacetDistribution_D(&bxdf->Microfacet.dist, wh);
    float G = MicrofacetDistribution_G(&bxdf->Microfacet.dist, wo, wi);
    return bxdf->S * Dwh * G * F / (4.f * cosThetaI * cosThetaO);
}

__bidevice__
Spectrum BxDF_MicrofacetTransmission_f(BxDF *bxdf, glm::vec3 wo, glm::vec3 wi){
    if (SameHemisphere(wo, wi)) return 0;  // transmission only
    
    float cosThetaO = CosTheta(wo);
    float cosThetaI = CosTheta(wi);
    if (cosThetaI == 0 || cosThetaO == 0) return Spectrum(0);
    
    float etaA = bxdf->Microfacet.etaA;
    float etaB = bxdf->Microfacet.etaB;
    
    float eta = CosTheta(wo) > 0 ? (etaB / etaA) : (etaA / etaB);
    glm::vec3 wh = glm::normalize(wo + wi * eta);
    if (wh.z < 0) wh = -wh;
    
    Spectrum F = fresnel_evalutate(&bxdf->Microfacet.fresnel, glm::dot(wo, wh));
    
    float sqrtDenom = glm::dot(wo, wh) + eta * glm::dot(wi, wh);
    float factor = (1) ? (1.f / eta) : 1.f;
    
    float Dwh = MicrofacetDistribution_D(&bxdf->Microfacet.dist, wh);
    float G = MicrofacetDistribution_G(&bxdf->Microfacet.dist, wo, wi);
    
    return (Spectrum(1.f) - F) * bxdf->S *
        glm::abs(Dwh * G * eta * eta * AbsDot(wi, wh) * AbsDot(wo, wh)
                 * factor * factor / (cosThetaI * cosThetaO * sqrtDenom * sqrtDenom));
}

__bidevice__
Spectrum BxDF_f(BxDF *bxdf, glm::vec3 &wo, glm::vec3 &wi){
    Spectrum f(0.0f);
    if(bxdf){
        switch(bxdf->self.model){
            case LambertianReflectionBxDF: {
                f = BxDF_LambertianReflection_f(bxdf, wo, wi);
            } break;
            
            case LambertianTransmissionBxDF:{
                f = BxDF_LambertianTransmission_f(bxdf, wo, wi);
            }break;
            
            case OrenNayarBxDF:{
                f = BxDF_OrenNayar_f(bxdf, wo, wi);
            } break;
            
            case SpecularReflectionBxDF:{
                f = BxDF_SpecularReflection_f(bxdf, wo, wi);
            } break;
            
            case SpecularTransmissionBxDF:{
                f = BxDF_SpecularTransmission_f(bxdf, wo, wi);
            } break;
            
            case FresnelSpecularBxDF:{
                return BxDF_FresnelSpecular_f(bxdf, wo, wi);
            } break;
            
            case MicrofacetReflectionBxDF:{
                return BxDF_MicrofacetReflection_f(bxdf, wo, wi);
            } break;
            
            case MicrofacetTransmissionBxDF:{
                return BxDF_MicrofacetTransmission_f(bxdf, wo, wi);
            } break;
            
            default: {}
        }
    }
    
    return f;
}

/***************************************************************
 > Computes a output direction and pdf, Sample_f.
****************************************************************/
__bidevice__
Spectrum BxDF_Gen_Sample_f(BxDF *bxdf, glm::vec3 &wo, glm::vec3 *wi,
                           glm::vec2 &u, float *pdf, BxDFType *sampledType)
{
    *wi = CosineSampleHemisphere(u); 
    if (wo.z < 0) wi->z *= -1;
    *pdf = BxDF_Pdf(bxdf, wo, *wi);
    return BxDF_f(bxdf, wo, *wi);
}

__bidevice__
Spectrum BxDF_LambertianReflection_Sample_f(BxDF *bxdf, glm::vec3 wo, glm::vec3 *wi, 
                                            glm::vec2 u, float *pdf,
                                            BxDFType *sampled)
{
    return BxDF_Gen_Sample_f(bxdf, wo, wi, u, pdf, sampled);
}

__bidevice__
Spectrum BxDF_LambertianTransmission_Sample_f(BxDF *bxdf, glm::vec3 wo,glm::vec3 *wi,
                                              glm::vec2 u, float *pdf,
                                              BxDFType *sampled)
{
    *wi = CosineSampleHemisphere(u);
    if (wo.z > 0) wi->z *= -1;
    *pdf = BxDF_LambertianTransmission_Pdf(bxdf, wo, *wi);
    return BxDF_LambertianTransmission_f(bxdf, wo, *wi);
}

__bidevice__
Spectrum BxDF_OrenNayar_Sample_f(BxDF *bxdf, glm::vec3 wo, glm::vec3 *wi, 
                                 glm::vec2 u, float *pdf, BxDFType *sampled)
{
    return BxDF_Gen_Sample_f(bxdf, wo, wi, u, pdf, sampled);
}

__bidevice__
Spectrum BxDF_SpecularReflection_Sample_f(BxDF *bxdf, glm::vec3 wo, glm::vec3 *wi, 
                                          glm::vec2 u, float *pdf, BxDFType *sampled)
{
    *wi = glm::vec3(-wo.x, -wo.y, wo.z);
    *pdf = 1.f;
    float cost = CosTheta(*wi);
    float abscost = AbsCosTheta(*wi);
    Spectrum e = fresnel_evalutate(&bxdf->Specular.fresnel, cost);
    return e * bxdf->S / abscost;
}

__bidevice__
Spectrum BxDF_SpecularTransmission_Sample_f(BxDF *bxdf, glm::vec3 wo, glm::vec3 *wi, 
                                            glm::vec2 u, float *pdf,
                                            BxDFType *sampled)
{
    bool entering = CosTheta(wo) > 0;
    float etaI = entering ? bxdf->Specular.etaA : bxdf->Specular.etaB;
    float etaT = entering ? bxdf->Specular.etaB : bxdf->Specular.etaA;
    
    if (!Refract(wo, Faceforward(glm::vec3(0, 0, 1), wo), etaI / etaT, wi))
        return 0;
    
    *pdf = 1.f;
    Spectrum e = fresnel_evalutate(&bxdf->Specular.fresnel, CosTheta(*wi));
    Spectrum ft = bxdf->S * (Spectrum(1.) - e);
    if(1) ft *= (etaI * etaI) / (etaT * etaT);
    return ft / AbsCosTheta(*wi);
}

__bidevice__
Spectrum BxDF_FresnelSpecular_Sample_f(BxDF *bxdf, glm::vec3 wo, glm::vec3 *wi, 
                                       glm::vec2 u, float *pdf,
                                       BxDFType *sampledType)
{
    float F = FrDieletric(CosTheta(wo), bxdf->Specular.etaA, bxdf->Specular.etaB);
    if (u[0] < F) {
        *wi = glm::vec3(-wo.x, -wo.y, wo.z);
        if(sampledType)
            *sampledType = BxDFType(BSDF_SPECULAR | BSDF_REFLECTION);
        *pdf = F;
        return F * bxdf->R / AbsCosTheta(*wi);
    }else{
        bool entering = CosTheta(wo) > 0;
        float etaI = entering ? bxdf->Specular.etaA : bxdf->Specular.etaB;
        float etaT = entering ? bxdf->Specular.etaB : bxdf->Specular.etaA;
        
        if(!Refract(wo, Faceforward(glm::vec3(0, 0, 1), wo), etaI / etaT, wi))
            return 0;
        Spectrum ft = bxdf->S * (1 - F);
        
        ft *= (etaI * etaI) / (etaT * etaT);
        if(sampledType)
            *sampledType = BxDFType(BSDF_SPECULAR | BSDF_TRANSMISSION);
        *pdf = 1 - F;
        return ft / AbsCosTheta(*wi);
    }
}

__bidevice__
Spectrum BxDF_MicrofacetReflection_Sample_f(BxDF *bxdf, glm::vec3 wo, glm::vec3 *wi, 
                                            glm::vec2 u, float *pdf, BxDFType *sampled)
{
    if (IsZero(wo.z)) return 0.;
    glm::vec3 wh = MicrofacetDistribution_Sample_wh(&bxdf->Microfacet.dist, wo, u);
    if (glm::dot(wo, wh) < 0) return 0.;   // Should be rare
    *wi = Reflect(wo, wh);
    if (!SameHemisphere(wo, *wi)) return Spectrum(0.f);
    
    // Compute PDF of _wi_ for microfacet reflection
    float pp = MicrofacetDistribution_Pdf(&bxdf->Microfacet.dist, wo, wh);
    *pdf = pp / (4.f * glm::dot(wo, wh));
    return BxDF_f(bxdf, wo, *wi);
}

__bidevice__
Spectrum BxDF_MicrofacetTransmission_Sample_f(BxDF *bxdf, glm::vec3 wo,
                                              glm::vec3 *wi, glm::vec2 u, float *pdf,
                                              BxDFType *sampled)
{
    if (IsZero(wo.z)) return 0.;
    glm::vec3 wh = MicrofacetDistribution_Sample_wh(&bxdf->Microfacet.dist, wo, u);
    if(glm::dot(wo, wh) < 0) return 0.;  // Should be rare
    
    float etaA = bxdf->Microfacet.etaA;
    float etaB = bxdf->Microfacet.etaB;
    float eta = CosTheta(wo) > 0 ? (etaA / etaB) : (etaB / etaA);
    if (!Refract(wo, (glm::vec3)wh, eta, wi)) return 0.f;
    *pdf = BxDF_Pdf(bxdf, wo, *wi);
    return BxDF_f(bxdf, wo, *wi);
}

__bidevice__ 
Spectrum BxDF_Sample_f(BxDF *bxdf, glm::vec3 &wo, glm::vec3 *wi, glm::vec2 &u,
                       float *pdf,BxDFType *sampledType)
{
    Spectrum f(0.0f);
    *pdf = 0.0f;
    if(bxdf){
        switch(bxdf->self.model){
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
            
            case SpecularReflectionBxDF:{
                f = BxDF_SpecularReflection_Sample_f(bxdf, wo, wi, u, pdf,
                                                     sampledType);
            } break;
            
            case SpecularTransmissionBxDF:{
                f = BxDF_SpecularTransmission_Sample_f(bxdf, wo, wi, u, pdf,
                                                       sampledType);
            } break;
            
            case FresnelSpecularBxDF:{
                return BxDF_FresnelSpecular_Sample_f(bxdf, wo, wi, u, pdf,
                                                     sampledType);
            } break;
            
            case MicrofacetReflectionBxDF:{
                return BxDF_MicrofacetReflection_Sample_f(bxdf, wo, wi, u, pdf,
                                                          sampledType);
            } break;
            
            case MicrofacetTransmissionBxDF:{
                return BxDF_MicrofacetTransmission_Sample_f(bxdf, wo, wi, u, pdf,
                                                            sampledType);
            } break;
            
            default: {}
        }
    }
    
    return f;
}

__bidevice__
Spectrum BSDF_f(BSDF *bsdf, glm::vec3 woW, glm::vec3 wiW, BxDFType flags){
    Spectrum f(0.f);
    if(bsdf){
        if(bsdf->nBxDFs > 0){
            glm::vec3 wi = bsdf->WorldToLocal(wiW);
            glm::vec3 wo = bsdf->WorldToLocal(woW);
            if(IsZero(wo.z)) return Spectrum(0.f);
            bool reflect = glm::dot(wiW, bsdf->ng) * glm::dot(woW, bsdf->ng) > 0;
            for(int i = 0; i < bsdf->nBxDFs; i++){
                BxDF *bxdf = &bsdf->bxdfs[i];
                if(BxDF_Matches(bxdf, flags) &&
                   ((reflect && (bxdf->self.type & BSDF_REFLECTION))||
                    (!reflect && (bxdf->self.type & BSDF_TRANSMISSION))))
                {
                    f += BxDF_f(bxdf, wo, wi);
                }
            }
        }else{
            printf("Empty BSDF at BSDF_f!\n");
        }
    }else{
        printf("Bad pointer at BSDF_f!\n");
    }
    
    return f;
}

__bidevice__
float BSDF_Pdf(BSDF *bsdf, glm::vec3 woW, glm::vec3 wiW, BxDFType flags){
    float pdf = 0.f;
    if(bsdf){
        if(bsdf->nBxDFs > 0){
            glm::vec3 wi = bsdf->WorldToLocal(wiW);
            glm::vec3 wo = bsdf->WorldToLocal(woW);
            if(IsZero(wo.z)) return 0.f;
            int matchingComps = 0;
            for(int i = 0; i < bsdf->nBxDFs; i++){
                BxDF *bxdf = &bsdf->bxdfs[i];
                if(BxDF_Matches(bxdf, flags)){
                    matchingComps += 1;
                    pdf += BxDF_Pdf(bxdf, wo, wi);
                }
            }
            
            float fpdf = matchingComps > 0 ? pdf / (float)matchingComps : 0.f;
            pdf = fpdf;
        }else{
            printf("Empty BSDF at BSDF_Pdf!\n");
        }
    }else{
        printf("Bad pointer at BSDF_Pdf!\n");
    }
    
    return pdf;
}

__bidevice__
int BSDF_NumComponents(BSDF *bsdf, BxDFType type){
    int matches = 0;
    for(int i = 0; i < bsdf->nBxDFs; i++){
        BxDF *bxdf = &bsdf->bxdfs[i];
        if(BxDF_Matches(bxdf, type)) matches++;
    }
    
    return matches;
}

__bidevice__
Spectrum BSDF_Sample_f(BSDF *bsdf, glm::vec3 &woW, glm::vec3 *wiW, 
                       glm::vec2 &u, float *pdf, BxDFType type, BxDFType *sampled)
{
    int matchingComps = BSDF_NumComponents(bsdf, type);
    if(matchingComps == 0){
        *pdf = 0.f;
        if(sampled) *sampled = BxDFType(0);
        return Spectrum(0.f);
    }
    
    int comp =
        glm::min((int)glm::floor(u[0] * matchingComps), matchingComps - 1);
    
    BxDF *bxdf = nullptr;
    int count = comp;
    for(int i = 0; i < bsdf->nBxDFs; i++){
        BxDF *b = &bsdf->bxdfs[i];
        if(BxDF_Matches(b, type) && count-- == 0){
            bxdf = b;
            break;
        }
    }
    
    glm::vec2 uRemapped(glm::min(u[0] * matchingComps - comp, 
                                 OneMinusEpsilon), u[1]);
    glm::vec3 wi, wo = bsdf->WorldToLocal(woW);
    
    if(IsZero(wo.z)) return 0.f;
    *pdf = 0.f;
    if(sampled) *sampled = bxdf->self.type;
    Spectrum f = BxDF_Sample_f(bxdf, wo, &wi, uRemapped, pdf, sampled);
    if(IsZero(*pdf)){
        if(sampled) *sampled = BxDFType(0);
        return 0.f;
    }
    
    *wiW = bsdf->LocalToWorld(wi);
    if(!(bxdf->self.type & BSDF_SPECULAR)){
        bool reflect = glm::dot(*wiW, bsdf->ng) * glm::dot(woW, bsdf->ng) > 0;
        f = Spectrum(0.f);
        for(int i = 0; i < bsdf->nBxDFs; i++){
            BxDF *b = &bsdf->bxdfs[i];
            if(BxDF_Matches(b, type) &&
               ((reflect && (b->self.type & BSDF_REFLECTION))||
                (!reflect && (b->self.type & BSDF_TRANSMISSION))))
            {
                f += BxDF_f(b, wo, wi);
            }
        }
    }
    
    return f;
}