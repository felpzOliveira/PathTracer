#if !defined(GEOMETRY_H)
#define GEOMETRY_H
#include <types.h>
#include <math.h>
#include <limits>
#include <cutil.h>

__bidevice__
Ray camera_get_ray(Camera *camera, float u, float v);

__device__
Ray camera_get_ray(Camera *camera, float u, float v, curandState *state);

__bidevice__
bool hit_aabb(AABB *aabb, Ray r, float tmin, float tmax);

//TODO:Instances
__device__
bool hit_mesh(Mesh *mesh, Ray ray, float t_min, float t_max, 
              hit_record *record, curandState *state);


__device__
bool hit_scene(Scene *scene, Ray r, float t_min, float t_max,
               hit_record *record, curandState *state);


template<typename Q>
inline __device__ 
bool hit_bvh(Q *locator, Ray ray, BVHNode *root, float tmin, 
             float tmax, hit_record *record, curandState *state);

#include <detail/geometry-inl.h>

#endif