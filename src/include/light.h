#if !defined(LIGHT_H)
#define LIGHT_H

#include <types.h>

inline __bidevice__
bool IsDeltaLight(int flags){
    return flags & (int)LightFlags::DeltaPosition ||
        flags & (int)LightFlags::DeltaDirection;
}

inline __bidevice__
void Light_Point_init(Light *light, Transform LightToWorld, Spectrum I);

inline __bidevice__
Spectrum Light_Sample_Li(Light *light, Scene *scene, hit_record *record, 
                         glm::vec2 u, glm::vec3 *wi, float *pdf, glm::vec3 *xl);

inline __bidevice__
Spectrum Light_Power(Light *light);

inline __bidevice__
Spectrum Light_Le(Light *light, glm::vec3 rd);

#include <light-inl.h>

#endif