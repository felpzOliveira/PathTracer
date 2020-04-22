#if !defined(TYPE_MATERIAL_H)
#define TYPE_MATERIAL_H
#include <cutil.h>
#include <spectrum.h>
#include <types/type_texture.h>

typedef unsigned int material_handle;

enum MaterialType{
    Matte,
    Glass,
    Plastic,
    Emitter
};

struct Material{
    MaterialType type;
    
    int has_Le;
    Spectrum Le;
    __bidevice__ Material(){ has_Le = false; Le = Spectrum(0.f); }
    
    __bidevice__ void SetLe(Spectrum _Le){
        has_Le = true;
        Le = _Le;
    }
    
    union{
        struct{
            texture_handle Kd;
            texture_handle sigma;
        }Matte;
        
        struct{
            texture_handle Kd, Ks;
            float roughness;
        }Plastic;
        
        struct{
            texture_handle Kt, Kr;
            float uroughness, vroughness;
            float index;
        }Glass;
    };
};

#endif