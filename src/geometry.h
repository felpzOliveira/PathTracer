#if !defined(GEOMETRY_H)
#define GEOMETRY_H
#include <types.h>
#include <math.h>
#include <limits>

inline __host__ __device__ Ray camera_get_ray(Camera *camera, float u, float v){
    Ray r;
    r.origin = camera->origin;
    r.direction = camera->lower_left + u * camera->horizontal + v * camera->vertical - camera->origin;
    return r;
}

inline __device__ Ray camera_get_ray(Camera *camera, float u, float v,
                                     curandState *state)
{
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

inline __host__ __device__ void sphere_get_uv_of(Sphere *sphere, glm::vec3 p,
                                                 float *u, float *v)
{
    float pi = 3.14169f;
    glm::vec3 op = glm::normalize(p - sphere->center);
    *u = atan2(op.x, op.z) / (2.0f * pi) + 0.5f;
    *v = op.y * 0.5f + 0.5f;
}

inline __host__ __device__ bool hit_sphere(Sphere *sphere, Ray r, float t_min,
                                           float t_max, hit_record *record)
{
    glm::vec3 oc = r.origin - sphere->center;
    float a = dot(r.direction, r.direction);
    float b = dot(oc, r.direction);
    float c = dot(oc, oc) - sphere->radius * sphere->radius;
    float delta = b*b - a*c;
    if(delta > 0.0f){
        float tmp = (-b -sqrt(delta))/a;
        if(tmp < t_max && tmp > t_min){
            record->t = tmp;
            record->p = r.origin + tmp * r.direction;
            record->normal = (record->p - sphere->center)/sphere->radius;
            record->mat_handle = sphere->mat_handle;
            sphere_get_uv_of(sphere, record->p, &record->u, &record->v);
            return true;
        }
        
        tmp = (-b +sqrt(delta))/a;
        if(tmp < t_max && tmp > t_min){
            record->t = tmp;
            record->p = r.origin + tmp * r.direction;
            record->normal = (record->p - sphere->center)/sphere->radius;
            record->mat_handle = sphere->mat_handle;
            sphere_get_uv_of(sphere, record->p, &record->u, &record->v);
            return true;
        }
    }
    
    return false;
}

inline __host__ __device__ bool hit_xy_rect(Rectangle *rect, Ray r, float t_min,
                                            float t_max, hit_record *record)
{
    float f = glm::abs(r.direction.z);
    if(f < 0.0001f) return false;
    //Check if hits z-plane
    float t = (rect->k - r.origin.z) / r.direction.z;
    
    if(t < t_min || t > t_max){
        return false;
    }
    
    //Check if bounds are inside
    float x = r.origin.x + t*r.direction.x;
    float y = r.origin.y + t*r.direction.y;
    if (x < rect->x0 || x > rect->x1 || y < rect->y0 || y > rect->y1){
        return false;
    }
    
    //Compute uv, simple projection
    record->u = (x - rect->x0) / (rect->x1 - rect->x0);
    record->v = (y - rect->y0) / (rect->y1 - rect->y0);
    record->t = t;
    record->p = r.origin + t * r.direction;
    record->normal = glm::vec3(0.0f, 0.0f, 1.0f);
    if(rect->flip_normals){
        record->normal = -record->normal;
    }
    record->mat_handle = rect->mat_handle;
    return true;
}

inline __host__ __device__ bool hit_xz_rect(Rectangle *rect, Ray r, float t_min,
                                            float t_max, hit_record *record)
{
    float f = glm::abs(r.direction.y);
    if(f < 0.0001f) return false;
    //Check if hits y-plane
    float t = (rect->k - r.origin.y) / r.direction.y;
    
    if(t < t_min || t > t_max){
        return false;
    }
    
    //Check if bounds are inside
    float x = r.origin.x + t*r.direction.x;
    float z = r.origin.z + t*r.direction.z;
    if (x < rect->x0 || x > rect->x1 || z < rect->z0 || z > rect->z1){
        return false;
    }
    
    //Compute uv, simple projection
    record->u = (x - rect->x0) / (rect->x1 - rect->x0);
    record->v = (z - rect->z0) / (rect->z1 - rect->z0);
    record->t = t;
    record->p = r.origin + t * r.direction;
    record->normal = glm::vec3(0.0f, 1.0f, 0.0f);
    if(rect->flip_normals){
        record->normal = -record->normal;
    }
    record->mat_handle = rect->mat_handle;
    return true;
}

inline __host__ __device__ bool hit_yz_rect(Rectangle *rect, Ray r, float t_min,
                                            float t_max, hit_record *record)
{
    //Check if hits x-plane
    float f = glm::abs(r.direction.x);
    if(f < 0.0001f) return false;
    float t = (rect->k - r.origin.x) / r.direction.x;
    
    if(t < t_min || t > t_max){
        return false;
    }
    
    //Check if bounds are inside
    float y = r.origin.y + t*r.direction.y;
    float z = r.origin.z + t*r.direction.z;
    if (y < rect->y0 || y > rect->y1 || z < rect->z0 || z > rect->z1){
        return false;
    }
    
    //Compute uv, simple projection
    record->u = (y - rect->y0) / (rect->y1 - rect->y0);
    record->v = (z - rect->z0) / (rect->z1 - rect->z0);
    record->t = t;
    record->p = r.origin + t * r.direction;
    record->normal = glm::vec3(1.0f, 0.0f, 0.0f);
    if(rect->flip_normals){
        record->normal = -record->normal;
    }
    record->mat_handle = rect->mat_handle;
    return true;
}

inline __host__ __device__ bool hit_box(Box *box, Ray ray, float t_min, 
                                        float t_max, hit_record *record)
{
    bool hit_anything = false;
    float closest = t_max;
    
    hit_record temp;
    //Ray r = ray;
    Ray r = ray_to_local_space(ray, box->transforms);
    
    for(int i = 0; i < 6; i += 1){
        bool hit = false;
        Rectangle *rec = &box->rects[i];
        switch(rec->rect_type){
            case OBJECT_XY_RECTANGLE:{
                hit = hit_xy_rect(rec, r, t_min, closest, &temp);
            } break;
            case OBJECT_XZ_RECTANGLE:{
                hit = hit_xz_rect(rec, r, t_min, closest, &temp);
            } break;
            case OBJECT_YZ_RECTANGLE:{
                hit = hit_yz_rect(rec, r, t_min, closest, &temp);
            } break;
            default: return false;
        }
        
        if(hit){
            if(temp.t > t_min && temp.t < closest){
                hit_anything = true;
                closest = temp.t;
                hit_record_remap_to_world(&temp, box->transforms, ray);
            }
        }
    }
    
    if(hit_anything){
        *record = temp;
    }
    
    return hit_anything;
}

inline __host__ __device__ bool hit_aabb(AABB *aabb, Ray r, float tmin, float tmax){
    glm::vec3 _min = aabb->_min;
    glm::vec3 _max = aabb->_max;
    for (int a = 0; a < 3; a++) {
        float t0 = ffmin((_min[a] - r.origin[a]) / r.direction[a],
                         (_max[a] - r.origin[a]) / r.direction[a]);
        float t1 = ffmax((_min[a] - r.origin[a]) / r.direction[a],
                         (_max[a] - r.origin[a]) / r.direction[a]);
        tmin = ffmax(t0, tmin);
        tmax = ffmin(t1, tmax);
        if (tmax <= tmin)
            return false;
    }
    return true;
}

//NOTE: Update when adding new primitives
inline __host__ __device__ bool _hit_object(Scene *scene, Ray ray, Object object,
                                            float tmin, float tmax,
                                            hit_record *record)
{
    bool hit_anything = false;
    object_handle id = object.object_handle;
    switch(object.object_type){
        case OBJECT_BOX: {
            Box *box = &scene->boxes[id];
            hit_anything = hit_box(box, ray, tmin, tmax, record);
        } break;
        case OBJECT_XY_RECTANGLE: {
            Rectangle *rect = &scene->rectangles[id];
            hit_anything = hit_xy_rect(rect, ray, tmin, tmax, record);
        } break;
        case OBJECT_XZ_RECTANGLE: {
            Rectangle *rect = &scene->rectangles[id];
            hit_anything = hit_xz_rect(rect, ray, tmin, tmax, record);
        } break;
        case OBJECT_YZ_RECTANGLE: {
            Rectangle *rect = &scene->rectangles[id];
            hit_anything = hit_yz_rect(rect, ray, tmin, tmax, record);
        } break;
        case OBJECT_SPHERE: {
            Sphere *sphere = &scene->spheres[id];
            hit_anything = hit_sphere(sphere, ray, tmin, tmax, record);
        } break;
        
        default: return false;
    }
    
    return hit_anything;
}

inline __device__ bool hit_medium(Scene *scene, Ray ray, Medium *medium,
                                  float tmin, float tmax, hit_record *record,
                                  curandState *state)
{
    hit_record rec1, rec2;
    bool hit_geometry = _hit_object(scene, ray, medium->geometry, 
                                    -FLT_MAX, FLT_MAX, &rec1);
    if(hit_geometry){
        bool hit_outer = _hit_object(scene, ray, medium->geometry,
                                     rec1.t+0.001f, FLT_MAX, &rec2);
        if(hit_outer){
            if(rec1.t < tmin) rec1.t = tmin;
            if(rec2.t > tmax) rec2.t = tmax;
            
            if(rec1.t >= rec2.t) return false;
            
            if(rec1.t < 0.0f) rec1.t = 0.0f;
            
            float d_inside = (rec2.t - rec1.t)*glm::length(ray.direction);
            float hit_distance = -(1.0f/medium->density)*log(random_float(state));
            if(hit_distance < d_inside){
                record->t = rec1.t + hit_distance/glm::length(ray.direction);
                record->p = ray.origin + record->t * ray.direction;
                record->normal = glm::vec3(1.0f, 0.0f, 0.0f); //dont matter
                record->mat_handle = medium->phase_function;
                record->u = 0.0f;
                record->v = 0.0f;
                
                return true;
            }
        }
    }
    
    return false;
}

//NOTE: Update when adding new primitives
inline __device__ bool hit_object(Scene *scene, Ray ray, Object object,
                                  float tmin, float tmax, hit_record *record, 
                                  curandState *state)
{
    //prevent binded objects
    if(!object.isbinded){
        //prevent Medium recursion
        if(object.object_type != OBJECT_MEDIUM){
            return _hit_object(scene, ray, object, tmin, tmax, record);
        }else{
            Medium *medium = &scene->mediums[object.object_handle];
            return hit_medium(scene, ray, medium, tmin, tmax, record, state);
            //return _hit_object(scene, ray, medium->geometry, tmin, tmax, record);
        }
    }
    
    return false;
}

inline __device__ bool hit_bvhnode_objects(BVHNodePtr node, Ray ray,
                                           Scene *scene, float tmin,
                                           float tmax, hit_record *record,
                                           curandState *state)
{
    bool hit_anything = false;
    float closest = tmax;
    hit_record temp;
    for(int i = 0; i < node->n_handles; i += 1){
        Object handle = node->handles[i];
        if(hit_object(scene, ray, handle, tmin, 
                      closest, &temp, state))
        {
            hit_anything = true;
            closest = temp.t;
        }
    }
    
    if(hit_anything){
        *record = temp;
    }
    return hit_anything;
}

inline __device__ bool hit_bvh(Scene *scene, Ray ray, BVHNode *root, float tmin, 
                               float tmax, hit_record *record, curandState *state)
{
    BVHNodePtr stack[BVH_MAX_DEPTH];
    BVHNodePtr *stackPtr = stack;
    *stackPtr++ = NULL;
    
    BVHNodePtr node = root;
    float closest = tmax;
    hit_record temp;
    bool hit_anything = false;
    bool hitbox = hit_aabb(&node->box, ray, tmin, tmax);
    if(hitbox && root->is_leaf){
        return hit_bvhnode_objects(node, ray, scene, tmin, 
                                   closest, record, state);
    }
    
    do{
        if(hitbox){
            BVHNodePtr childL = node->left;
            BVHNodePtr childR = node->right;
            bool hitl = false;
            bool hitr = false;
            if(childL->n_handles > 0 || childR->n_handles > 0){
                hitl = hit_aabb(&childL->box, ray, tmin, tmax);
                hitr = hit_aabb(&childR->box, ray, tmin, tmax);
            }
            
            if(hitl && childL->is_leaf){
                if(hit_bvhnode_objects(childL, ray, scene, tmin, 
                                       closest, &temp, state))
                {
                    hit_anything = true;
                    closest = temp.t;
                }
            }
            
            if(hitr && childR->is_leaf){
                if(hit_bvhnode_objects(childR, ray, scene, tmin, 
                                       closest, &temp, state))
                {
                    hit_anything = true;
                    closest = temp.t;
                }
            }
            
            bool transverseL = (hitl && !childL->is_leaf);
            bool transverseR = (hitr && !childR->is_leaf);
            if(!transverseR && !transverseL){
                node = *--stackPtr;
            }else{
                node = (transverseL) ? childL : childR;
                if(transverseL && transverseR){
                    *stackPtr++ = childR;
                }
            }
        }else{
            node = *--stackPtr;
        }
        
        if(node)
            hitbox = hit_aabb(&node->box, ray, tmin, tmax);
    }while(node != NULL);
    
    if(hit_anything){
        *record = temp;
    }
    
    return hit_anything;
}

inline __device__ bool hit_scene(Scene *scene, Ray r, float t_min,
                                 float t_max, hit_record *record,
                                 curandState *state)
{
    return hit_bvh(scene, r, scene->bvh, t_min, t_max, record, state);
}

#endif