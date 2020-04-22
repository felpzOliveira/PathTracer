#if !defined(CAMERA_H)
#error "Please include camera.h instead of this file"
#else
#include <types.h>

inline __host__ 
Camera * camera_new(glm::vec3 origin, glm::vec3 at, glm::vec3 up, 
                    float vfov, float aspect)
{
    Camera *camera = nullptr;
    camera = (Camera *)cudaAllocate(sizeof(Camera));
    float theta = glm::radians(vfov);
    float half_height = glm::tan(theta/2.0f);
    float half_width = aspect * half_height;
    camera->origin = origin;
    camera->with_focus = 0;
    camera->lens_radius = 0.0f;
    camera->w = glm::normalize(origin - at);
    camera->u = glm::normalize(glm::cross(up, camera->w));
    camera->v = glm::cross(camera->w, camera->u);
    
    glm::vec3 u = camera->u;
    glm::vec3 v = camera->v;
    glm::vec3 w = camera->w;
    camera->lower_left = origin - half_width*u - half_height*v - w;
    camera->horizontal = 2.0f * half_width * u;
    camera->vertical = 2.0f * half_height * v;
    return camera;
}

inline __host__ 
Camera * camera_new(glm::vec3 origin,  glm::vec3 at, glm::vec3 up,
                    float vfov, float aspect, float aperture, float focus_dist)
{
    Camera *camera = nullptr;
    camera = (Camera *)cudaAllocate(sizeof(Camera));
    
    float theta = glm::radians(vfov);
    float half_height = glm::tan(theta/2.0f);
    float half_width = aspect * half_height;
    camera->origin = origin;
    camera->w = glm::normalize(origin - at);
    camera->u = glm::normalize(glm::cross(up, camera->w));
    camera->v = glm::cross(camera->w, camera->u);
    glm::vec3 u = camera->u;
    glm::vec3 v = camera->v;
    glm::vec3 w = camera->w;
    
    camera->lower_left = origin - half_width*focus_dist*u - half_height*focus_dist*v - focus_dist*w;
    camera->horizontal = 2.0f*half_width*focus_dist*u;
    camera->vertical = 2.0f*half_height*focus_dist*v;
    camera->with_focus = 1;
    camera->lens_radius = aperture/2.0f;
    return camera;
}


inline __bidevice__ 
Ray camera_get_ray(Camera *camera, float u, float v)
{
    Ray r;
    r.origin = camera->origin;
    r.direction = camera->lower_left + u * camera->horizontal + v * camera->vertical - camera->origin;
    return r;
}

inline __device__ 
Ray camera_get_ray(Camera *camera, float u, float v, curandState *state){
    if(!camera->with_focus){
        return camera_get_ray(camera, u, v);
    }
    
    Ray r;
    glm::vec3 rd = random_in_disk(state) * camera->lens_radius;
    glm::vec3 offset = camera->u * rd.x + camera->v * rd.y;
    r.origin = camera->origin + offset;
    r.direction = camera->lower_left + u * camera->horizontal + v * camera->vertical - camera->origin - offset;
    return r;
}

#endif