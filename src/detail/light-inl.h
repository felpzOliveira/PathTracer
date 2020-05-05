#if !defined(LIGHT_H)
#error "Please include light.h instead of this file"
#else

#include <utilities.h>
#include <stdio.h>

inline __bidevice__
void Light_Point_init(Light *light, Transform LightToWorld, Spectrum I){
    if(light){
        light->type = LightType::Point;
        light->flags = LightFlags::DeltaPosition;
        light->nSamples = 1;
        light->Point.LightToWorld = LightToWorld;
        light->Point.WorldToLight = Inverse(LightToWorld);
        light->Point.pLight = LightToWorld.point(glm::vec3(0.f));
        light->Point.I = I;
    }else{
        printf("Bad pointer at Light::Point init\n");
    }
}


/***************************************************************
 > Computes Sample_Li.
****************************************************************/
inline __bidevice__
Spectrum Light_Point_Sample_Li(Light *light, Scene *scene, hit_record *record, 
                               glm::vec2 u, glm::vec3 *wi, float *pdf, glm::vec3 *xl)
{
    *wi = glm::normalize(light->Point.pLight - record->p);
    *pdf = 1.f;
    *xl = light->Point.pLight;
    return light->Point.I / DistanceSquared(*xl, record->p);
}


inline __bidevice__
Spectrum Light_Sample_Li(Light *light, Scene *scene, hit_record *record, 
                         glm::vec2 u, glm::vec3 *wi, float *pdf, glm::vec3 *xl)
{
    if(light && record && scene && wi && pdf && xl){
        switch(light->type){
            case LightType::Point: {
                return Light_Point_Sample_Li(light, scene, record, u, wi, pdf, xl);
            } break;
            
            default:{
                printf("Unknown light type\n");
            }
        }
    }else{
        printf("Bad pointer at Light::Sample_Li\n");
    }
    
    *pdf = 0.f;
    return Spectrum(0.f);
}


/***************************************************************
 > Computes Power.
****************************************************************/
inline __bidevice__
Spectrum Light_Point_Power(Light *light){
    return 4.f * light->Point.I * Pi;
}

inline __bidevice__
Spectrum Light_Power(Light *light){
    if(light){
        switch(light->type){
            case LightType::Point: {
                return Light_Point_Power(light);
            } break;
            
            default:{
                printf("Unknown light type\n");
            }
        }
    }else{
        printf("Bad pointer at Light::Power\n");
    }
    
    return Spectrum(0.f);
}


/***************************************************************
 > Computes Le.
****************************************************************/
//TODO
inline __bidevice__
Spectrum Light_Le(Light *light, glm::vec3 rd){
    return Spectrum(0.f);
}

#endif