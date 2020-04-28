#if !defined(BSDF_H)
#define BSDF_H

#include <types/type_bsdf.h>
#include <fresnel.h>
#include <microfacet.h>


inline __bidevice__
void BxDF_LambertianTransmission_init(BxDF *bxdf, Spectrum R);

inline __bidevice__
void BxDF_LambertianReflection_init(BxDF *bxdf, Spectrum T);

inline __bidevice__
void BxDF_OrenNayar_init(BxDF *bxdf, Spectrum R, float sigma);

inline __bidevice__
void BxDF_SpecularReflection_init(BxDF *bxdf, Spectrum R, Fresnel *fresnel);

inline __bidevice__
void BxDF_SpecularTransmission_init(BxDF *bxdf, Spectrum T, float A, float B);

inline __bidevice__
void BxDF_FresnelSpecular_init(BxDF *bxdf, Spectrum R, Spectrum T, 
                               float A, float B);

inline __bidevice__
void BxDF_Microfacet_Reflection_init(BxDF *bxdf, Spectrum R,
                                     MicrofacetDistribution *dist,
                                     Fresnel *fresnel);
inline __bidevice__
void BxDF_Microfacet_Transmission_init(BxDF *bxdf, Spectrum T,
                                       MicrofacetDistribution *dist,
                                       float etaA, float etaB);

inline __bidevice__
Spectrum BSDF_f(BSDF *bsdf, glm::vec3 woW, glm::vec3 wiW, BxDFType flags);

inline __bidevice__
float BSDF_Pdf(BSDF *bsdf, glm::vec3 woW, glm::vec3 wiW, BxDFType flags);

inline __bidevice__
Spectrum BSDF_Sample_f(BSDF *bsdf, glm::vec3 &woW, glm::vec3 *wiW, 
                       glm::vec2 &u, float *pdf, BxDFType type, BxDFType *sampled);

inline __bidevice__
int BSDF_NumComponents(BSDF *bsdf, BxDFType type);


#include <detail/bsdf-inl.h>

#endif