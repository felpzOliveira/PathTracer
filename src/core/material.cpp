#include <material.h>
#include <bsdf.h>

__bidevice__
void material_matte_init(Material *material, texture_handle kd,
                         texture_handle sigma)
{
    material->type = MaterialType::Matte;
    material->Matte.Kd = kd;
    material->Matte.sigma = sigma;
}

__bidevice__
void material_mirror_init(Material *material, texture_handle kr){
    material->type = MaterialType::Mirror;
    material->Mirror.Kr = kr;
}

__bidevice__
void material_plastic_init(Material *material, texture_handle kd, texture_handle ks,
                           float roughness)
{
    material->type = MaterialType::Plastic;
    material->Plastic.Kd = kd;
    material->Plastic.Ks = ks;
    material->Plastic.roughness = MicrofacetRoughnessToAlpha(roughness);
}

__bidevice__
void material_glass_reflector_init(Material *material, texture_handle kt,
                                   texture_handle kr, float index)
{
    material->type = MaterialType::GlassReflector;
    material->GlassReflector.Kt = kt;
    material->GlassReflector.Kr = kr;
    material->GlassReflector.index = index;
}

__bidevice__
void material_glass_init(Material *material, texture_handle kr, texture_handle kt,
                         float uroughness, float vroughness, float index)
{
    material->type = MaterialType::Glass;
    material->Glass.Kt = kt;
    material->Glass.Kr = kr;
    material->Glass.uroughness = MicrofacetRoughnessToAlpha(uroughness);
    material->Glass.vroughness = MicrofacetRoughnessToAlpha(vroughness);
    material->Glass.index = index;
}

__bidevice__
void material_sample_matte(Material *material, hit_record *record, 
                           BSDF *bsdf, Scene *scene)
{
    Texture *kd = &scene->texture_table[material->Matte.Kd];
    Texture *tsig = &scene->texture_table[material->Matte.sigma];
    
    Spectrum r = texture_value(kd, record, scene).Clamp();
    Spectrum ssigma = texture_value(tsig, record, scene);
    float sigma = Clamp(ssigma.c[0], 0, 90);
    if(!r.IsBlack()){
        BxDF bxdf;
        if(IsZero(sigma)){
            BxDF_LambertianReflection_init(&bxdf, r);
        }else{
            BxDF_OrenNayar_init(&bxdf, r, sigma);
        }
        
        BSDF_Insert(bsdf, &bxdf);
    }
}

__bidevice__
void material_sample_mirror(Material *material, hit_record *record, 
                            BSDF *bsdf, Scene *scene)
{
    Texture *kr = &scene->texture_table[material->Mirror.Kr];
    Spectrum r = texture_value(kr, record, scene).Clamp();
    
    if(!r.IsBlack()){
        BxDF bxdf;
        Fresnel fresnel;
        fresnel_noop_init(&fresnel);
        BxDF_SpecularReflection_init(&bxdf, r, &fresnel);
        //BxDF_LambertianReflection_init(&bxdf, r);
        BSDF_Insert(bsdf, &bxdf);
    }
}

__bidevice__
void material_sample_plastic(Material *material, hit_record *record, 
                             BSDF *bsdf, Scene *scene)
{
    Texture *kd = &scene->texture_table[material->Plastic.Kd];
    Texture *ks = &scene->texture_table[material->Plastic.Ks];
    
    Spectrum r = texture_value(kd, record, scene).Clamp();
    Spectrum s = texture_value(ks, record, scene).Clamp();
    
    if(!r.IsBlack()){
        BxDF bxdf;
        BxDF_LambertianReflection_init(&bxdf, r);
        BSDF_Insert(bsdf, &bxdf);
    }
    
    if(!s.IsBlack()){
        BxDF bxdf;
        Fresnel fresnel;
        MicrofacetDistribution dist(material->Plastic.roughness,
                                    material->Plastic.roughness,
                                    MicrofacetType::TrowbridgeReitz);
        
        fresnel_dieletric_init(&fresnel, 1.5f, 1.f);
        BxDF_Microfacet_Reflection_init(&bxdf, s, &dist, &fresnel);
        BSDF_Insert(bsdf, &bxdf);
    }
}


__bidevice__
void material_sample_glass_reflector(Material *material, hit_record *record, 
                                     BSDF *bsdf, Scene *scene)
{
    Texture *kr = &scene->texture_table[material->GlassReflector.Kr];
    Texture *kt = &scene->texture_table[material->GlassReflector.Kt];    
    Spectrum R = texture_value(kr, record, scene).Clamp();
    Spectrum T = texture_value(kt, record, scene).Clamp();
    float eta = material->GlassReflector.index;
    
    if(!R.IsBlack() && !T.IsBlack()){
        BxDF bxdf;
        BxDF_FresnelSpecular_init(&bxdf, R, T, 1.f, eta);
        BSDF_Insert(bsdf, &bxdf);
    }
}

__bidevice__
void material_sample_glass(Material *material, hit_record *record, 
                           BSDF *bsdf, Scene *scene)
{
    Texture *kr = &scene->texture_table[material->Glass.Kr];
    Texture *kt = &scene->texture_table[material->Glass.Kt];
    float eta = material->Glass.index;
    float uroughness = material->Glass.uroughness;
    float vroughness = material->Glass.vroughness;
    Spectrum R = texture_value(kr, record, scene).Clamp();
    Spectrum T = texture_value(kt, record, scene).Clamp();
    
    if(R.IsBlack() && T.IsBlack()) return;
    /////////////////////////////////////////////
    bool allowMultipleLobes = false;
    ////////////////////////////////////////////
    
    bool isSpecular = IsZero(uroughness) && IsZero(vroughness);
    if(isSpecular && allowMultipleLobes){
        BxDF bxdf;
        BxDF_FresnelSpecular_init(&bxdf, R, T, 1.f, eta);
        BSDF_Insert(bsdf, &bxdf);
    }else{
        MicrofacetDistribution dist(uroughness, vroughness,
                                    MicrofacetType::TrowbridgeReitz);
        if(!R.IsBlack()){
            BxDF bxdf;
            Fresnel fresnel;
            fresnel_dieletric_init(&fresnel, 1.f, eta);
            if(isSpecular){
                BxDF_SpecularReflection_init(&bxdf, R, &fresnel);
            }else{
                BxDF_Microfacet_Reflection_init(&bxdf, R, &dist, &fresnel);
            }
            
            BSDF_Insert(bsdf, &bxdf);
        }
        
        if(!T.IsBlack()){
            BxDF bxdf;
            if(isSpecular){
                BxDF_SpecularTransmission_init(&bxdf, T, 1.f, eta);
            }else{
                BxDF_Microfacet_Transmission_init(&bxdf, T, &dist, 1.f, eta);
            }
            
            BSDF_Insert(bsdf, &bxdf);
        }
    }
}

__bidevice__
void material_sample(Material *material, hit_record *record, 
                     BSDF *bsdf, Scene *scene)
{
    switch(material->type){
        case MaterialType::Matte:{
            material_sample_matte(material, record, bsdf, scene);
        } break;
        
        case MaterialType::Plastic:{
            material_sample_plastic(material, record, bsdf, scene);
        } break;
        
        case MaterialType::Glass:{
            material_sample_glass(material, record, bsdf, scene);
        } break;
        
        case MaterialType::GlassReflector:{
            material_sample_glass_reflector(material, record, bsdf, scene);
        } break;
        
        case MaterialType::Mirror:{
            material_sample_mirror(material, record, bsdf, scene);
        } break;
        
        default:{
            printf("Unknown material!\n");
        }
    }
}