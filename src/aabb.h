#if !defined(AABB_H)
#define AABB_H
#include <types.h>

inline __host__ __device__ void aabb_init(AABB *aabb, glm::vec3 a, glm::vec3 b, 
                                          ObjectType type)
{
    if(aabb){
        aabb->_min = a;
        aabb->_max = b;
    }
}

inline __host__ __device__ void surrounding_box(AABB *output, 
                                                AABB *box0, 
                                                AABB *box1)
{
    glm::vec3 small( ffmin(box0->_min.x, box1->_min.x),
                    ffmin(box0->_min.y, box1->_min.y),
                    ffmin(box0->_min.z, box1->_min.z));
    glm::vec3 big  ( ffmax(box0->_max.x, box1->_max.x),
                    ffmax(box0->_max.y, box1->_max.y),
                    ffmax(box0->_max.z, box1->_max.z));
    
    aabb_init(output, small, big, OBJECT_AABB);
}

inline __host__ __device__ bool sphere_bounding_box(Sphere *sphere, AABB *aabb){
    aabb_init(aabb, sphere->center - glm::vec3(glm::abs(sphere->radius)),
              sphere->center + glm::vec3(glm::abs(sphere->radius)), 
              OBJECT_SPHERE);
    return true;
}

inline __host__ __device__ bool xy_rect_bounding_box(Rectangle *rect, AABB *aabb){
    aabb_init(aabb, glm::vec3(rect->x0, rect->y0, rect->k-0.001f),
              glm::vec3(rect->x1, rect->y1, rect->k+0.001f),
              OBJECT_XY_RECTANGLE);
    return true;
}

inline __host__ __device__ bool xz_rect_bounding_box(Rectangle *rect, AABB *aabb){
    aabb_init(aabb, glm::vec3(rect->x0, rect->k-0.001f, rect->z0),
              glm::vec3(rect->x1, rect->k+0.001f, rect->z1),
              OBJECT_XZ_RECTANGLE);
    return true;
}

inline __host__ __device__ bool yz_rect_bounding_box(Rectangle *rect, AABB *aabb){
    aabb_init(aabb, glm::vec3(rect->k-0.001f, rect->y0, rect->z0),
              glm::vec3(rect->k+0.001f, rect->y1, rect->z1),
              OBJECT_YZ_RECTANGLE);
    return true;
}

inline __host__ __device__ bool box_bounding_box(Box *box, AABB *aabb){
    glm::vec3 v0 = toWorld(box->p0, box->transforms);
    glm::vec3 v1 = toWorld(box->p1, box->transforms);
    
    float lx = v1.x - v0.x;
    float ly = v1.y - v0.y;
    float lz = v1.z - v0.z;
    
    float m = lx > ly ? lx : ly;
    m = m > lz ? m : lz;
    m *= 1.0f/glm::sqrt(2.0f);
    v0 = v0 - glm::vec3(m);
    v1 = v1 + glm::vec3(m);
    aabb_init(aabb, v0, v1, OBJECT_BOX);
    
    return true;
}

inline __host__ void _get_aabb(Scene *scene, Object obj, AABB *aabb){
    if(obj.object_type == OBJECT_SPHERE){
        Sphere *sphere = &scene->spheres[obj.object_handle];
        sphere_bounding_box(sphere, aabb);
    }else if(obj.object_type == OBJECT_XY_RECTANGLE){
        Rectangle *rect = &scene->rectangles[obj.object_handle];
        xy_rect_bounding_box(rect, aabb);
    }else if(obj.object_type == OBJECT_XZ_RECTANGLE){
        Rectangle *rect = &scene->rectangles[obj.object_handle];
        xz_rect_bounding_box(rect, aabb);
    }else if(obj.object_type == OBJECT_YZ_RECTANGLE){
        Rectangle *rect = &scene->rectangles[obj.object_handle];
        yz_rect_bounding_box(rect, aabb);
    }else if(obj.object_type == OBJECT_BOX){
        Box *box = &scene->boxes[obj.object_handle];
        box_bounding_box(box, aabb);
    }
}

//NOTE: Update when adding new primitives
inline __host__ void get_aabb(Scene *scene, Object obj, AABB *aabb){
    if(obj.object_type != OBJECT_MEDIUM){
        _get_aabb(scene, obj, aabb);
    }else{
        Medium *med = &scene->mediums[obj.object_handle];
        _get_aabb(scene, med->geometry, aabb);
    }
}

#endif
