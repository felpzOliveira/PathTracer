#include <microfacet.h>

__bidevice__ Float MicrofacetDistribution::Lambda(const vec3f &w) const{
    Float absTanTheta = Absf(TanTheta(w));
    if (std::isinf(absTanTheta)) return 0.;
    
    Float alpha =
        std::sqrt(Cos2Phi(w) * alphax * alphax + Sin2Phi(w) * alphay * alphay);
    Float alpha2Tan2Theta = (alpha * absTanTheta) * (alpha * absTanTheta);
    return (-1 + std::sqrt(1.f + alpha2Tan2Theta)) / 2;
}

__bidevice__ Float MicrofacetDistribution::D(const vec3f &wh) const{
    Float tan2Theta = Tan2Theta(wh);
    if (std::isinf(tan2Theta)) return 0.;
    const Float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);
    Float e =
        (Cos2Phi(wh) / (alphax * alphax) + Sin2Phi(wh) / (alphay * alphay)) *
        tan2Theta;
    return 1 / (Pi * alphax * alphay * cos4Theta * (1 + e) * (1 + e));
}


static __bidevice__ void Sample11(Float cosTheta, Float U1, Float U2,
                                  Float *slope_x, Float *slope_y)
{
    if(cosTheta > .9999){
        Float r = sqrt(U1 / (1 - U1));
        Float phi = 6.28318530718 * U2;
        *slope_x = r * cos(phi);
        *slope_y = r * sin(phi);
        return;
    }
    
    Float sinTheta =
        std::sqrt(Max((Float)0, (Float)1 - cosTheta * cosTheta));
    Float tanTheta = sinTheta / cosTheta;
    Float a = 1 / tanTheta;
    Float G1 = 2 / (1 + std::sqrt(1.f + 1.f / (a * a)));
    
    Float A = 2 * U1 / G1 - 1;
    Float tmp = 1.f / (A * A - 1.f);
    if(tmp > 1e10) tmp = 1e10;
    Float B = tanTheta;
    Float D = std::sqrt(Max(Float(B * B * tmp * tmp - (A * A - B * B) * tmp), Float(0)));
    Float slope_x_1 = B * tmp - D;
    Float slope_x_2 = B * tmp + D;
    *slope_x = (A < 0 || slope_x_2 > 1.f / tanTheta) ? slope_x_1 : slope_x_2;
    
    Float S;
    if(U2 > 0.5f){
        S = 1.f;
        U2 = 2.f * (U2 - .5f);
    }else{
        S = -1.f;
        U2 = 2.f * (.5f - U2);
    }
    
    Float z = (U2 * (U2 * (U2 * 0.27385f - 0.73369f) + 0.46341f)) /
        (U2 * (U2 * (U2 * 0.093073f + 0.309420f) - 1.000000f) + 0.597999f);
    *slope_y = S * z * std::sqrt(1.f + *slope_x * *slope_x);
}

static __bidevice__ vec3f DistributionSample(const vec3f &wi, Float alpha_x,
                                             Float alpha_y, Float U1, Float U2)
{
    vec3f wiStretched =
        Normalize(vec3f(alpha_x * wi.x, alpha_y * wi.y, wi.z));
    
    Float slope_x, slope_y;
    Sample11(CosTheta(wiStretched), U1, U2, &slope_x, &slope_y);
    
    Float tmp = CosPhi(wiStretched) * slope_x - SinPhi(wiStretched) * slope_y;
    slope_y = SinPhi(wiStretched) * slope_x + CosPhi(wiStretched) * slope_y;
    slope_x = tmp;
    
    slope_x = alpha_x * slope_x;
    slope_y = alpha_y * slope_y;
    
    return Normalize(vec3f(-slope_x, -slope_y, 1.));
}

__bidevice__ Float MicrofacetDistribution::Pdf(const vec3f &wo, const vec3f &wh) const{
    if (sampleVisibleArea)
        return D(wh) * G1(wo) * AbsDot(wo, wh) / AbsCosTheta(wo);
    else
        return D(wh) * AbsCosTheta(wh);
}

__bidevice__ vec3f MicrofacetDistribution::Sample_wh(const vec3f &wo, 
                                                     const Point2f &u) const
{
    vec3f wh;
    if(!sampleVisibleArea){
        Float cosTheta = 0, phi = (2 * Pi) * u[1];
        if(alphax == alphay){
            Float tanTheta2 = alphax * alphax * u[0] / (1.0f - u[0]);
            cosTheta = 1 / std::sqrt(1 + tanTheta2);
        }else{
            phi =
                std::atan(alphay / alphax * std::tan(2 * Pi * u[1] + .5f * Pi));
            if (u[1] > .5f) phi += Pi;
            Float sinPhi = std::sin(phi), cosPhi = std::cos(phi);
            const Float alphax2 = alphax * alphax, alphay2 = alphay * alphay;
            const Float alpha2 =
                1 / (cosPhi * cosPhi / alphax2 + sinPhi * sinPhi / alphay2);
            Float tanTheta2 = alpha2 * u[0] / (1 - u[0]);
            cosTheta = 1 / std::sqrt(1 + tanTheta2);
        }
        
        Float sinTheta = std::sqrt(Max((Float)0., (Float)1. - cosTheta * cosTheta));
        wh = SphericalDirection(sinTheta, cosTheta, phi);
        if(!SameHemisphere(wo, wh)) wh = -wh;
    }else{
        bool flip = wo.z < 0;
        wh = DistributionSample(flip ? -wo : wo, alphax, alphay, u[0], u[1]);
        if(flip) wh = -wh;
    }
    
    return wh;
}