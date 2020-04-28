#if !defined(CAMERA_H)
#define CAMERA_H
#include <types.h>

inline __host__ 
Camera * camera_new(glm::vec3 origin, glm::vec3 at, glm::vec3 up, 
                    float vfov, float aspect);

inline __host__ 
Camera * camera_new(glm::vec3 origin,  glm::vec3 at, glm::vec3 up,
                    float vfov, float aspect, float aperture, float focus_dist);

inline __bidevice__ 
Ray camera_get_ray(Camera *camera, float u, float v);

inline __device__ 
Ray camera_get_ray(Camera *camera, float u, float v, curandState *state);



#include <detail/camera-inl.h>

#endif