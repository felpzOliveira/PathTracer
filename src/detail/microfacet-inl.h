#if !defined(MICROFACET_H)
#error "Please include microfacet.h instead of this file"
#else

#include <utilities.h>
#define SQRT_PI_INV (1.f / glm::sqrt(Pi))

inline __bidevice__
void TrowbridgeReitzSample11(float cosTheta, float U1, float U2,
                             float *slope_x, float *slope_y)
{
    // special case (normal incidence)
    if (cosTheta > .9999) {
        float r = glm::sqrt(U1 / (1 - U1));
        float phi = 6.28318530718 * U2;
        *slope_x = r * glm::cos(phi);
        *slope_y = r * glm::sin(phi);
        return;
    }
    
    float sinTheta =
        glm::sqrt(glm::max((float)0, (float)1 - cosTheta * cosTheta));
    float tanTheta = sinTheta / cosTheta;
    float a = 1 / tanTheta;
    float G1 = 2 / (1 + glm::sqrt(1.f + 1.f / (a * a)));
    
    // sample slope_x
    float A = 2 * U1 / G1 - 1;
    float tmp = 1.f / (A * A - 1.f);
    if (tmp > 1e10) tmp = 1e10;
    float B = tanTheta;
    float D = glm::sqrt(
        glm::max(float(B * B * tmp * tmp - (A * A - B * B) * tmp), float(0)));
    float slope_x_1 = B * tmp - D;
    float slope_x_2 = B * tmp + D;
    *slope_x = (A < 0 || slope_x_2 > 1.f / tanTheta) ? slope_x_1 : slope_x_2;
    
    // sample slope_y
    float S;
    if (U2 > 0.5f) {
        S = 1.f;
        U2 = 2.f * (U2 - .5f);
    } else {
        S = -1.f;
        U2 = 2.f * (.5f - U2);
    }
    float z =
        (U2 * (U2 * (U2 * 0.27385f - 0.73369f) + 0.46341f)) /
        (U2 * (U2 * (U2 * 0.093073f + 0.309420f) - 1.000000f) + 0.597999f);
    *slope_y = S * z * glm::sqrt(1.f + *slope_x * *slope_x);
}

__bidevice__
glm::vec3 TrowbridgeReitzSample(const glm::vec3 &wi, float alpha_x,
                                float alpha_y, float U1, float U2) {
    // 1. stretch wi
    glm::vec3 wiStretched =
        glm::normalize(glm::vec3(alpha_x * wi.x, alpha_y * wi.y, wi.z));
    
    // 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
    float slope_x, slope_y;
    TrowbridgeReitzSample11(CosTheta(wiStretched), U1, U2, &slope_x, &slope_y);
    
    // 3. rotate
    float tmp = CosPhi(wiStretched) * slope_x - SinPhi(wiStretched) * slope_y;
    slope_y = SinPhi(wiStretched) * slope_x + CosPhi(wiStretched) * slope_y;
    slope_x = tmp;
    
    // 4. unstretch
    slope_x = alpha_x * slope_x;
    slope_y = alpha_y * slope_y;
    
    // 5. compute normal
    return glm::normalize(glm::vec3(-slope_x, -slope_y, 1.));
}

inline __bidevice__
void BeckmannSample11(float cosThetaI, float U1, float U2,
                      float *slope_x, float *slope_y) {
    /* Special case (normal incidence) */
    if (cosThetaI > .9999) {
        float r = glm::sqrt(-glm::log(1.0f - U1));
        float sinPhi = glm::sin(2 * Pi * U2);
        float cosPhi = glm::cos(2 * Pi * U2);
        *slope_x = r * cosPhi;
        *slope_y = r * sinPhi;
        return;
    }
    
    /* The original inversion routine from the paper contained
       discontinuities, which causes issues for QMC integration
       and techniques like Kelemen-style MLT. The following code
       performs a numerical inversion with better behavior */
    float sinThetaI =
        glm::sqrt(glm::max((float)0, (float)1 - cosThetaI * cosThetaI));
    float tanThetaI = sinThetaI / cosThetaI;
    float cotThetaI = 1 / tanThetaI;
    
    /* Search interval -- everything is parameterized
       in the Erf() domain */
    float a = -1, c = Erf(cotThetaI);
    float sample_x = glm::max(U1, (float)1e-6f);
    
    /* Start with a good initial guess */
    // float b = (1-sample_x) * a + sample_x * c;
    
    /* We can do better (inverse of an approximation computed in
     * Mathematica) */
    float thetaI = glm::acos(cosThetaI);
    float fit = 1 + thetaI * (-0.876f + thetaI * (0.4265f - 0.0594f * thetaI));
    float b = c - (1 + c) * glm::pow(1 - sample_x, fit);
    
    /* Normalization factor for the CDF */
    float normalization =
        1 /
        (1 + c + SQRT_PI_INV * tanThetaI * glm::exp(-cotThetaI * cotThetaI));
    
    int it = 0;
    while (++it < 10) {
        /* Bisection criterion -- the oddly-looking
           Boolean expression are intentional to check
           for NaNs at little additional cost */
        if (!(b >= a && b <= c)) b = 0.5f * (a + c);
        
        /* Evaluate the CDF and its derivative
           (i.e. the density function) */
        float invErf = ErfInv(b);
        float value =
            normalization *
            (1 + b + SQRT_PI_INV 
             * tanThetaI * glm::exp(-invErf * invErf)) -
            sample_x;
        float derivative = normalization * (1 - invErf * tanThetaI);
        
        if (glm::abs(value) < 1e-5f) break;
        
        /* Update bisection intervals */
        if (value > 0)
            c = b;
        else
            a = b;
        
        b -= value / derivative;
    }
    
    /* Now convert back into a slope value */
    *slope_x = ErfInv(b);
    
    /* Simulate Y component */
    *slope_y = ErfInv(2.0f * glm::max(U2, (float)1e-6f) - 1.0f);
}


inline __bidevice__
glm::vec3 BeckmannSample(const glm::vec3 &wi, float alpha_x, float alpha_y,
                         float U1, float U2) 
{
    // 1. stretch wi
    glm::vec3 wiStretched =
        glm::normalize(glm::vec3(alpha_x * wi.x, alpha_y * wi.y, wi.z));
    
    // 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
    float slope_x, slope_y;
    BeckmannSample11(CosTheta(wiStretched), U1, U2, &slope_x, &slope_y);
    
    // 3. rotate
    float tmp = CosPhi(wiStretched) * slope_x - SinPhi(wiStretched) * slope_y;
    slope_y = SinPhi(wiStretched) * slope_x + CosPhi(wiStretched) * slope_y;
    slope_x = tmp;
    
    // 4. unstretch
    slope_x = alpha_x * slope_x;
    slope_y = alpha_y * slope_y;
    
    // 5. compute normal
    return glm::normalize(glm::vec3(-slope_x, -slope_y, 1.f));
}

/***************************************************************
 > Computes D(w) for given distribution.
****************************************************************/
inline __bidevice__
float MicrofacetDistributionBeckmann_D(MicrofacetDistribution *dist, glm::vec3 wh)
{
    float tan2Theta = Tan2Theta(wh);
    if (isinf(tan2Theta)) return 0.;
    float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);
    return glm::exp(-tan2Theta * (Cos2Phi(wh) / (dist->alphax * dist->alphax) +
                                  Sin2Phi(wh) / (dist->alphay * dist->alphay))) /
        (Pi * dist->alphax * dist->alphay * cos4Theta);
}

inline __bidevice__
float MicrofacetDistributionTrowbridgeReitz_D(MicrofacetDistribution *dist,
                                              glm::vec3 wh)
{
    float tan2Theta = Tan2Theta(wh);
    if (isinf(tan2Theta)) return 0.f;
    float alphax = dist->alphax;
    float alphay = dist->alphay;
    const float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);
    float e =
        (Cos2Phi(wh) / (alphax * alphax) + Sin2Phi(wh) / (alphay * alphay)) *
        tan2Theta;
    return 1.f / (Pi * alphax * alphay * cos4Theta * (1.f + e) * (1.f + e));
}

inline __bidevice__
float MicrofacetDistribution_D(MicrofacetDistribution *dist, glm::vec3 w){
    if(dist){
        switch(dist->type){
            case MicrofacetType::Beckmann:{
                return MicrofacetDistributionBeckmann_D(dist, w);
            } break;
            
            case MicrofacetType::TrowbridgeReitz:{
                return MicrofacetDistributionTrowbridgeReitz_D(dist, w);
            } break;
            
            default:{
                printf("Unknown Microfacet distribution\n");
            }
        }
    }else{
        printf("Bad pointer at MicrofacetDistribution::D\n");
    }
    
    return 0.f;
}

/***************************************************************
 > Computes Lambda(w) for given distribution.
****************************************************************/
inline __bidevice__
float MicrofacetDistributionBeckmann_L(MicrofacetDistribution *dist, glm::vec3 w)
{
    float absTanTheta = ABS(TanTheta(w));
    if (isinf(absTanTheta)) return 0.;
    float alphax = dist->alphax;
    float alphay = dist->alphay;
    float alpha =
        glm::sqrt(Cos2Phi(w) * alphax * alphax + Sin2Phi(w) * alphay * alphay);
    float a = 1.f / (alpha * absTanTheta);
    if (a >= 1.6f) return 0.f;
    return (1.f - 1.259f * a + 0.396f * a * a) / (3.535f * a + 2.181f * a * a);
}

inline __bidevice__
float MicrofacetDistributionTrowbridgeReitz_L(MicrofacetDistribution *dist,
                                              glm::vec3 w)
{
    float absTanTheta = ABS(TanTheta(w));
    if (isinf(absTanTheta)) return 0.;
    float alphax = dist->alphax;
    float alphay = dist->alphay;
    float alpha =
        glm::sqrt(Cos2Phi(w) * alphax * alphax + Sin2Phi(w) * alphay * alphay);
    float alpha2Tan2Theta = (alpha * absTanTheta) * (alpha * absTanTheta);
    return (-1.f + glm::sqrt(1.f + alpha2Tan2Theta)) / 2.f;
}

inline __bidevice__
float MicrofacetDistribution_Lambda(MicrofacetDistribution *dist, glm::vec3 w){
    if(dist){
        switch(dist->type){
            case MicrofacetType::Beckmann:{
                return MicrofacetDistributionBeckmann_L(dist, w);
            } break;
            
            case MicrofacetType::TrowbridgeReitz:{
                return MicrofacetDistributionTrowbridgeReitz_L(dist, w);
            } break;
            
            default:{
                printf("Unknown Microfacet distribution\n");
            }
        }
    }else{
        printf("Bad pointer at MicrofacetDistribution::Lambda\n");
    }
    
    return 0.f;
}

/***************************************************************
 > Computes Sample_wh.
****************************************************************/
inline __bidevice__
glm::vec3 MicrofacetDistributionBeckmann_Sample_wh(MicrofacetDistribution *dist,
                                                   glm::vec3 &wo, glm::vec2 &u)
{
    if(!dist->sampleVisibleArea) {
        // Sample full distribution of normals for Beckmann distribution
        
        // Compute $\tan^2 \theta$ and $\phi$ for Beckmann distribution sample
        float alphax = dist->alphax;
        float alphay = dist->alphay;
        float tan2Theta, phi;
        if (alphax == alphay) {
            float logSample = glm::log(1 - u[0]);
            tan2Theta = -alphax * alphax * logSample;
            phi = u[1] * 2 * Pi;
        } else {
            // Compute _tan2Theta_ and _phi_ for anisotropic Beckmann
            // distribution
            float logSample = glm::log(1 - u[0]);
            phi = std::atan(alphay / alphax *
                            glm::tan(2 * Pi * u[1] + 0.5f * Pi));
            if (u[1] > 0.5f) phi += Pi;
            float sinPhi = glm::sin(phi), cosPhi = glm::cos(phi);
            float alphax2 = alphax * alphax, alphay2 = alphay * alphay;
            tan2Theta = -logSample /
                (cosPhi * cosPhi / alphax2 + sinPhi * sinPhi / alphay2);
        }
        
        // Map sampled Beckmann angles to normal direction _wh_
        float cosTheta = 1 / glm::sqrt(1 + tan2Theta);
        float sinTheta = glm::sqrt(glm::max((float)0, 1 - cosTheta * cosTheta));
        glm::vec3 wh = SphericalDirection(sinTheta, cosTheta, phi);
        if (!SameHemisphere(wo, wh)) wh = -wh;
        return wh;
    } else {
        // Sample visible area of normals for Beckmann distribution
        glm::vec3 wh;
        bool flip = wo.z < 0;
        wh = BeckmannSample(flip ? -wo : wo, dist->alphax, dist->alphay, u[0], u[1]);
        if (flip) wh = -wh;
        return wh;
        
    }
}

inline __bidevice__
glm::vec3 MicrofacetDistributionTR_Sample_wh(MicrofacetDistribution *dist,
                                             glm::vec3 &wo, glm::vec2 &u)
{
    glm::vec3 wh;
    float alphax = dist->alphax;
    float alphay = dist->alphay;
    if (!dist->sampleVisibleArea) {
        float cosTheta = 0, phi = (2 * Pi) * u[1];
        if (alphax == alphay) {
            float tanTheta2 = alphax * alphax * u[0] / (1.0f - u[0]);
            cosTheta = 1 / glm::sqrt(1 + tanTheta2);
        } else {
            phi =
                glm::atan(alphay / alphax * glm::tan(2 * Pi * u[1] + .5f * Pi));
            if (u[1] > .5f) phi += Pi;
            float sinPhi = glm::sin(phi), cosPhi = glm::cos(phi);
            const float alphax2 = alphax * alphax, alphay2 = alphay * alphay;
            const float alpha2 =
                1 / (cosPhi * cosPhi / alphax2 + sinPhi * sinPhi / alphay2);
            float tanTheta2 = alpha2 * u[0] / (1 - u[0]);
            cosTheta = 1 / glm::sqrt(1 + tanTheta2);
        }
        float sinTheta =
            glm::sqrt(glm::max((float)0., (float)1. - cosTheta * cosTheta));
        wh = SphericalDirection(sinTheta, cosTheta, phi);
        if (!SameHemisphere(wo, wh)) wh = -wh;
    } else {
        bool flip = wo.z < 0;
        wh = TrowbridgeReitzSample(flip ? -wo : wo, alphax, alphay, u[0], u[1]);
        if (flip) wh = -wh;
    }
    return wh;
}

inline __bidevice__
glm::vec3 MicrofacetDistribution_Sample_wh(MicrofacetDistribution *dist,
                                           glm::vec3 &wo, glm::vec2 &u)
{
    if(dist){
        switch(dist->type){
            case MicrofacetType::Beckmann:{
                return MicrofacetDistributionBeckmann_Sample_wh(dist, wo, u);
            } break;
            
            case MicrofacetType::TrowbridgeReitz:{
                return MicrofacetDistributionTR_Sample_wh(dist, wo, u);
            } break;
            
            default:{
                printf("Unknown Microfacet distribution\n");
            }
        }
    }else{
        printf("Bad pointer at MicrofacetDistribution::Sample_wh\n");
    }
    
    return glm::vec3(0.f);
}



inline __bidevice__
float MicrofacetRoughnessToAlpha(float roughness){
    roughness = glm::max(roughness, (float)1e-3);
    float x = glm::log(roughness);
    return 1.62142f + 0.819955f * x + 0.1734f * x * x +
        0.0171201f * x * x * x + 0.000640711f * x * x * x * x;
}

inline __bidevice__
float MicrofacetDistribution_G1(MicrofacetDistribution *dist, glm::vec3 w){
    return (1.0f / (1.0f + MicrofacetDistribution_Lambda(dist, w)));
}

inline __bidevice__
float MicrofacetDistribution_G(MicrofacetDistribution *dist,
                               glm::vec3 wo, glm::vec3 wi)
{
    float sum = MicrofacetDistribution_Lambda(dist, wo) +
        MicrofacetDistribution_Lambda(dist, wi);
    return (1.0f / (1.0f + sum));
}

inline __bidevice__
float MicrofacetDistribution_Pdf(MicrofacetDistribution *dist, 
                                 glm::vec3 wo, glm::vec3 wh)
{
    float Dwh = MicrofacetDistribution_D(dist, wh);
    float G1 = MicrofacetDistribution_G1(dist, wo);
    if(dist->sampleVisibleArea){
        return Dwh * G1 * AbsDot(wo, wh) / AbsCosTheta(wo);
    }else{
        return Dwh * AbsCosTheta(wh);
    }
}

#endif