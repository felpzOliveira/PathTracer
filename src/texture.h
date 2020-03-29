#if !defined(TEXTURE_H)
#define TEXTURE_H

#include <types.h>
#include <noise.h>

inline __host__ __device__ glm::vec3 texture_value_no_recursion(Texture *texture, float u, 
                                                                float v, glm::vec3 &p,
                                                                Scene *scene);

inline __host__ __device__ glm::vec3 texture_value_const(Texture *texture, float u, 
                                                         float v, glm::vec3 &p,
                                                         Scene *scene)
{
    return texture->color;
}

inline __host__ __device__ glm::vec3 texture_value_checker(Texture *texture, 
                                                           float u, float v,
                                                           glm::vec3 &p,
                                                           Scene *scene)
{
    Texture *odd = &scene->texture_table[texture->odd];
    Texture *even = &scene->texture_table[texture->even];
    
    if(odd->type == TEXTURE_CHECKER || even->type == TEXTURE_CHECKER){
        return glm::vec3(0.0f);
    }
    
    float freq = 10.0f;
    float sines = glm::sin(freq * p.x) * glm::sin(freq * p.y) * glm::sin(freq * p.z);
    if(sines < 0){
        //Compiler is complaining about this, because of stack bullshit
        //return texture_value(odd, u, v, p, scene);
        return texture_value_no_recursion(odd, u, v, p, scene);
    }else{
        return texture_value_no_recursion(even, u, v, p, scene);
    }
}

inline __host__ __device__ glm::vec3 texture_value_noise(Texture *texture, 
                                                         float u, float v,
                                                         glm::vec3 &p,
                                                         Scene *scene)
{
    switch(texture->noise_type){
        case NOISE_SIMPLE: return texture->color * noise31(scene->perlin, p*4.0f, 0);
        case NOISE_TRILINEAR: {
            float f = 0.5f*(1.0f+glm::sin(0.1f*p.x+5.0f*turb(scene->perlin, p)));
            return texture->color * f;
        }
        
        //case NOISE_TRILINEAR: return texture->color * snoise(p*4.0f);
        default: return glm::vec3(0.0f);
    }
}

inline __host__ __device__ glm::vec2 texture_apply_distribution(Scene *scene,
                                                                float u, float v,
                                                                Texture *texture)
{
    TextureWrapMode wrap_mode = texture->props.wrap_mode;
    if(wrap_mode == TEXTURE_WRAP_CLAMP){
        u = (u > 0.999f) ? 0.999f : u;
        u = (u < 0.001f) ? 0.001f : u;
        v = (v > 0.999f) ? 0.999f : v;
        v = (v < 0.001f) ? 0.001f : v;
    }else if(wrap_mode == TEXTURE_WRAP_REPEAT){
        u = glm::mod(u * texture->props.scale, 1.0f);
        v = glm::mod(v * texture->props.scale, 1.0f);
    }
    
    return glm::vec2(u, v);
}

inline __host__ __device__ glm::vec3 texture_value_image(Texture *texture,
                                                         float u, float v,
                                                         glm::vec3 &p,
                                                         Scene *scene)
{
    glm::vec2 uv = texture_apply_distribution(scene, u, v, texture);
    u = uv.x;
    v = uv.y;
    
    int i = u * texture->image_x;
    int j = (1.0f - v)*texture->image_y-0.001f;
    if(i < 0) i = 0;
    if(j < 0) j = 0;
    if(i > texture->image_x-1) i = texture->image_x-1;
    if(j > texture->image_y-1) j = texture->image_y-1;
    float r = int(texture->image[3*i + 3*texture->image_x*j+0])/255.0f;
    float g = int(texture->image[3*i + 3*texture->image_x*j+1])/255.0f;
    float b = int(texture->image[3*i + 3*texture->image_x*j+2])/255.0f;
    return glm::vec3(r,g,b);
}

inline __host__ __device__ glm::vec3 texture_value_no_recursion(Texture *texture, float u, 
                                                                float v, glm::vec3 &p,
                                                                Scene *scene)
{
    switch(texture->type){
        case TEXTURE_CONST: return texture_value_const(texture, u, v, p, scene);
        case TEXTURE_NOISE: return texture_value_noise(texture, u, v, p, scene);
        case TEXTURE_IMAGE: return texture_value_image(texture, u, v, p, scene);
        default: return glm::vec3(0.0f);
    }
}

inline __host__ __device__ glm::vec3 texture_value(Texture *texture, float u, 
                                                   float v, glm::vec3 &p,
                                                   Scene *scene)
{
    switch(texture->type){
        case TEXTURE_CONST: return texture_value_const(texture, u, v, p, scene);
        case TEXTURE_NOISE: return texture_value_noise(texture, u, v, p, scene);
        case TEXTURE_IMAGE: return texture_value_image(texture, u, v, p, scene);
        case TEXTURE_CHECKER: return texture_value_checker(texture, u, v, p, scene);
        default: return glm::vec3(0.0f);
    }
}

#endif