#if !defined(MATERIAL_H)
#define MATERIAL_H

#include <types.h>
#include <texture.h>
#include <bsdf.h>

inline __bidevice__
void material_matte_init(Material *material, texture_handle kd,
                         texture_handle sigma);

inline __bidevice__
void material_plastic_init(Material *material, texture_handle kd, 
                           texture_handle ks, float roughness);

inline __bidevice__
void material_glass_init(Material *material, texture_handle kr, texture_handle kt,
                         float uroughness, float vroughness, float index);

inline __bidevice__
void material_sample(Material *material, hit_record *record, 
                     BSDF *bsdf, Scene *scene);

#include <detail/material-inl.h>

#endif