#if !defined(MATERIAL_H)
#define MATERIAL_H

#include <types.h>
#include <texture.h>
#include <onb.h>

inline __device__ void sample_material(Ray ray, Scene *scene, 
                                       Material *material, hit_record *record,
                                       LightEval *eval, curandState *state)
{
    glm::vec3 emitted(0.0f);
    Texture *texture0 = &scene->texture_table[material->albedo];
    Texture *texture1 = &scene->texture_table[material->emitted];
    eval->attenuation = texture_value(texture0, record->u, 
                                      record->v, record->p, scene);
    
    if(glm::dot(record->normal, ray.direction) < 0.0f){
        emitted = texture_value(texture1, record->u, record->v, record->p, scene);
    }
    
    eval->emitted = emitted;
    eval->emitted *= material->intensity;
}

/////////////////////////////////////////////////////////////////////////////////////
//                                L A M B E R T I A N                              //
/////////////////////////////////////////////////////////////////////////////////////
inline __device__ bool scatter_lambertian(Ray ray, hit_record *record, Scene *scene,
                                          Ray *scattered, Material *material, 
                                          curandState *state)
{
    Onb uvw;
    onb_from_w(&uvw, record->normal);
    glm::vec3 direction = onb_local(&uvw, random_cosine_direction(state));
    scattered->origin = record->p;
    scattered->direction = glm::normalize(direction);
    return true;
}

inline __device__ float material_brdf_lambertian(Ray ray, hit_record *record,
                                                 Ray scattered)
{
    //return 1.0f / M_PI;
    float cosine = glm::dot(record->normal, scattered.direction);
    return cosine < 0 ? 0 : cosine / M_PI;
}
/////////////////////////////////////////////////////////////////////////////////////
//                                    M E T A L                                    //
/////////////////////////////////////////////////////////////////////////////////////
inline __device__ bool scatter_metal(Ray ray, hit_record *record, Scene *scene,
                                     Ray *scattered, Material *material, 
                                     curandState *state)
{
    scattered->origin = record->p;
    glm::vec3 refl = glm::reflect(glm::normalize(ray.direction), record->normal);
    refl += material->fuzz * random_in_sphere(state);
    scattered->direction = refl;
    float outside = glm::dot(scattered->direction, record->normal);
    return outside > 0;
}

/////////////////////////////////////////////////////////////////////////////////////
//                                     E M I T T E R                               //
/////////////////////////////////////////////////////////////////////////////////////

//TODO: Shouldn't this depends on the intensity as we could get relection happening?
inline __device__ bool scatter_emitter(Ray ray, hit_record *record, Scene *scene,
                                       Ray *scattered, Material *material, 
                                       curandState *state)
{
    //TODO: Maybe return lambertian here?
    return false;
}

/////////////////////////////////////////////////////////////////////////////////////
//                                    D I E L E T R I C                            //
/////////////////////////////////////////////////////////////////////////////////////

inline __device__ bool scatter_dieletric(Ray ray, hit_record *record, Scene *scene,
                                         Ray *scattered, Material *material, 
                                         curandState *state)
{
    glm::vec3 outnormal;
    glm::vec3 dir = glm::normalize(ray.direction);
    glm::vec3 reflected = glm::reflect(dir, record->normal);
    float ni_over_nt;
    float cosine;
    float reflect_prob;
    glm::vec3 refracted;
    
    if(glm::dot(dir, record->normal) > 0){
        outnormal = -record->normal;
        ni_over_nt = material->ref_idx;
        cosine = material->ref_idx * glm::dot(ray.direction, record->normal)/
            glm::length(ray.direction);
    }else{
        outnormal = record->normal;
        ni_over_nt = 1.0f/material->ref_idx;
        cosine = -glm::dot(ray.direction, record->normal)/glm::length(ray.direction);
    }
    
    if(refract(ray.direction, outnormal, ni_over_nt, refracted)){
        reflect_prob = schlick(cosine, material->ref_idx);
    }else{
        reflect_prob = 1.0f;
    }
    
    if(random_float(state) < reflect_prob){
        scattered->origin = record->p;
        scattered->direction = reflected;
    }else{
        scattered->origin = record->p;
        scattered->direction = refracted;
    }
    
    return true;
}

/////////////////////////////////////////////////////////////////////////////////////
//                                    I S O T R O P I C                            //
/////////////////////////////////////////////////////////////////////////////////////

inline __device__ bool scatter_isotropic(Ray ray, hit_record *record, Scene *scene,
                                         Ray *scattered, Material *material, 
                                         curandState *state)
{
    scattered->origin = record->p;
    scattered->direction = random_in_sphere(state);
    return true;
}

inline __device__ bool scatter(Ray ray, hit_record *record, Scene *scene,
                               LightEval *eval, Ray *scattered,
                               Material *material, curandState *state)
{
    bool rv = false;
    record->is_specular = true;
    if(material){
        switch(material->mattype){
            case LAMBERTIAN: {
                rv = scatter_lambertian(ray, record, scene, scattered, 
                                        material, state); 
                record->is_specular = false;
            } break;
            
            case METAL: {
                rv = scatter_metal(ray, record, scene, scattered, material, state);
            } break;
            
            case DIELETRIC: {
                rv = scatter_dieletric(ray, record, scene, scattered, 
                                       material, state);
            } break;
            
            case EMITTER: {
                rv = scatter_emitter(ray, record, scene, scattered, 
                                     material, state);
            } break;
            
            case ISOTROPIC: {
                rv = scatter_isotropic(ray, record, scene, scattered,
                                       material, state);
            } break;
            
            default: return false;
        }
    }
    
    return rv;
}

inline __device__ float material_brdf(Ray ray, hit_record *record, Ray scattered,
                                      Material *material)
{
    float r = 1.0f;
    switch(material->mattype){
        case LAMBERTIAN: {
            r = material_brdf_lambertian(ray, record, scattered);
        } break;
        
        default: r = 1.0f;
    }
    return r;
}

inline __device__ void ray_sample_material(Ray ray, Scene *scene, Material *material,
                                           hit_record *record, LightEval *eval,
                                           curandState *state)
{
    eval->emitted = glm::vec3(0.0f);
    eval->attenuation = glm::vec3(0.0f);
    sample_material(ray, scene, material, record, eval, state);
}

#endif