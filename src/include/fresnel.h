#if !defined(FRESNEL_H)
#define FRESNEL_H
#include <spectrum.h>
#include <glm/glm.hpp>

#include <types/type_fresnel.h>


inline __bidevice__
void fresnel_conductor_init(Fresnel *fresnel, Spectrum etaI, 
                            Spectrum etaT, Spectrum k);


inline __bidevice__
void fresnel_dieletric_init(Fresnel *fresnel, float etaI, float etaT);


inline __bidevice__
void fresnel_noop_init(Fresnel *fresnel);

inline __bidevice__
Spectrum fresnel_evalutate(Fresnel *fresnel, float cosThetaI);

#include <detail/fresnel-inl.h>

#endif