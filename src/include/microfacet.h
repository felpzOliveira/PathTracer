#if !defined(MICROFACET_H)
#define MICROFACET_H

#include <types/type_microfacet.h>
#include <glm/glm.hpp>

inline __bidevice__
glm::vec3 MicrofacetDistribution_Sample_wh(MicrofacetDistribution *dist,
                                           glm::vec3 &wo, glm::vec2 &u);
inline __bidevice__
float MicrofacetDistribution_Lambda(MicrofacetDistribution *dist, glm::vec3 w);

inline __bidevice__
float MicrofacetDistribution_D(MicrofacetDistribution *dist, glm::vec3 w);

inline __bidevice__
float MicrofacetRoughnessToAlpha(float roughness);

inline __bidevice__
float MicrofacetDistribution_G1(MicrofacetDistribution *dist, glm::vec3 w);

inline __bidevice__
float MicrofacetDistribution_G(MicrofacetDistribution *dist,
                               glm::vec3 wo, glm::vec3 wi);

inline __bidevice__
float MicrofacetDistribution_Pdf(MicrofacetDistribution *dist, 
                                 glm::vec3 wo, glm::vec3 wh);

#include <detail/microfacet-inl.h>

#endif