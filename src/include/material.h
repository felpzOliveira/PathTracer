#if !defined(MATERIAL_H)
#define MATERIAL_H

#include <types.h>
#include <texture.h>
#include <bsdf.h>

/*
 * Classical matte material
 */
__bidevice__
void material_matte_init(Material *material, texture_handle kd,
                         texture_handle sigma);

/*
 * Classical mirro material, reflects in one direction with pdf = 1
 */
__bidevice__
void material_mirror_init(Material *material, texture_handle kr);

/*
 * Combines lambertian  reflection BRDF with microfacet reflection, pdf varies.
 */
__bidevice__
void material_plastic_init(Material *material, texture_handle kd, 
                           texture_handle ks, float roughness);

/*
 * This is the 'real' glass that mixes transmission/reflection with microfacets.
 * This is really good but it gets weird to compare with others because
 * the look is allways a little bit different.
 */
__bidevice__
void material_glass_init(Material *material, texture_handle kr, texture_handle kt,
                         float uroughness, float vroughness, float index);

/*
 * This is the classical glass material made of perfect transmission/reflection.
 * It is not real, but it is good.
 */
__bidevice__
void material_glass_reflector_init(Material *material, texture_handle kt,
                                   texture_handle kr, float index);

/*
 * Samples a material: computes all BRDFs that must be evaluated and inserts
 * them in a single BSDF instance.
 */
__bidevice__
void material_sample(Material *material, hit_record *record, 
                     BSDF *bsdf, Scene *scene);

#endif