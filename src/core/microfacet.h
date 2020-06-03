#pragma once
#include <geometry.h>

class MicrofacetDistribution{
    public:
    Float alphax, alphay;
    bool sampleVisibleArea;
    
    __bidevice__ MicrofacetDistribution(){}
    __bidevice__ MicrofacetDistribution(Float alphax, Float alphay,
                                        bool sampleVis = true)
        : alphax(alphax), alphay(alphay), sampleVisibleArea(sampleVis){}
    
    __bidevice__ void Set(Float _alphax, Float _alphay,
                          bool sampleVis = true)
    {
        alphax = RoughnessToAlpha(_alphax);
        alphay = RoughnessToAlpha(_alphay);
        sampleVisibleArea = sampleVis;
    }
    
    __bidevice__ void SetUnmapped(Float _alphax, Float _alphay,
                                  bool sampleVis = true)
    {
        alphax = _alphax;
        alphay = _alphay;
        sampleVisibleArea = sampleVis;
    }
    
    __bidevice__ Float G1(const vec3f &w) const{
        return 1 / (1 + Lambda(w));
    }
    
    __bidevice__ Float G(const vec3f &wo, const vec3f &wi) const{
        return 1 / (1 + Lambda(wo) + Lambda(wi));
    }
    
    __bidevice__ Float RoughnessToAlpha(Float roughness) const{
        roughness = Max(roughness, (Float)1e-3);
        Float x = std::log(roughness);
        return 1.62142f + 0.819955f * x + 0.1734f * x * x + 0.0171201f * x * x * x +
            0.000640711f * x * x * x * x;
    }
    
    __bidevice__ Float Pdf(const vec3f &wo, const vec3f &wh) const;
    __bidevice__ Float Lambda(const vec3f &w) const;
    __bidevice__ Float D(const vec3f &wh) const;
    __bidevice__ vec3f Sample_wh(const vec3f &wo, const Point2f &u) const;
};