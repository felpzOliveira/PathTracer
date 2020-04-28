#if !defined(TYPE_FRESNEL_H)
#define TYPE_FRESNEL_H
#include <spectrum.h>

typedef enum{
    FresnelDieletric,
    FresnelConductor,
    FresnelNoOp
}FresnelType;

struct Fresnel{
    Spectrum sEtaI, sEtaT, k;
    float etaI, etaT;
    FresnelType type;
};


#endif