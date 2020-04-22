#if !defined(TYPE_MICROFACET_H)
#define TYPE_MICROFACET_H

#include <cutil.h>

enum MicrofacetType{
    Beckmann,
    TrowbridgeReitz,
};

struct MicrofacetDistribution{
    bool sampleVisibleArea;
    float alphax, alphay;
    MicrofacetType type;
    
    __bidevice__ MicrofacetDistribution() {sampleVisibleArea = true;}
    __bidevice__ MicrofacetDistribution(float ax, float ay, MicrofacetType t)
    {
        sampleVisibleArea = true;
        alphax = ax;
        alphay = ay;
        type = t;
    }
};

#endif