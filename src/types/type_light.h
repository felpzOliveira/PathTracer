#if !defined(TYPE_LIGHT_H)
#define TYPE_LIGHT_H
#include <cutil.h>
#include <transform.h>
#include <spectrum.h>

enum LightFlags{
    DeltaPosition = 1,
    DeltaDirection = 2,
    Area = 4,
    Infinite = 8
};

enum LightType{
    Point,
};

struct Light{
    int flags;
    int nSamples;
    LightType type;
    
    __bidevice__ Light(){}
    __bidevice__ Light(const Light &light){
        memcpy(this, &light, sizeof(Light));
    }
    
    union{
        struct{
            Transform WorldToLight, LightToWorld;
            Spectrum I;
            glm::vec3 pLight;
        }Point;
    };
};


#endif