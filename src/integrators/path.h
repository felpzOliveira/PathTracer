#if !defined(PATH_H)
#define PATH_H

#include <types.h>
#include <scene.h>
#include <geometry.h>
#include <material.h>
#include <camera.h>
#include <bsdf.h>


inline __device__
Spectrum Li(Ray &ray, Scene *scene, curandState *state)
{
    hit_record record;
    if(!hit_scene(scene, ray, 0.0001f, FLT_MAX, &record, state)){
        return Spectrum(0.f);
    }
    
    return Spectrum(1.f);
}


#endif