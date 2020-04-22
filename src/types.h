#if !defined(TYPES_H)
#define TYPES_H
#include <glm/glm.hpp>
#include <cfloat>
#include <cuda.h>
#include <curand_kernel.h>
#include <vector>
#include <spectrum.h>
#include <fresnel.h>

#include <types/type_bsdf.h>
#include <types/type_onb.h>
#include <types/type_texture.h>
#include <types/type_material.h>

#define BVH_MAX_DEPTH 12
#define BVH_MAX_STACK 1024 //2^(h+1)-1

typedef unsigned int ray_handle;

/* Ray definition */
typedef struct Ray_t{
    glm::vec3 origin;
    glm::vec3 direction;
}Ray;



/* Camera definition by image plane */
typedef struct Camera_t{
    glm::vec3 origin;
    glm::vec3 lower_left;
    glm::vec3 horizontal;
    glm::vec3 vertical;
    float lens_radius;
    int with_focus;
    glm::vec3 u,v,w;
}Camera;

typedef int aabb_handle;
typedef unsigned int object_handle;

typedef enum{
    OBJECT_SPHERE,
    OBJECT_XY_RECTANGLE,
    OBJECT_XZ_RECTANGLE,
    OBJECT_YZ_RECTANGLE,
    OBJECT_BOX,
    OBJECT_TRIANGLE,
    OBJECT_MESH,
    OBJECT_MEDIUM,
    OBJECT_AABB,
}ObjectType;

typedef struct Transforms_t{
    glm::mat4 toWorld;
    glm::mat4 toLocal;
    glm::mat4 normalMatrix;
}Transforms;

typedef struct Object_t{
    int isvalid;
    int isbinded;
    object_handle handle;
    ObjectType object_type;
}Object;

/* Hit information */
typedef struct hit_record_t{
    float t; //distance
    glm::vec3 p; //hit point
    glm::vec3 normal; //normal at p
    material_handle mat_handle; //handle of the material of hitted surface
    float u,v; //texture coordinates of hit point
    Object hitted;
    int is_specular;
}hit_record;

/* Sphere definition */
typedef struct Sphere_t{
    glm::vec3 center;
    float radius;
    material_handle mat_handle;
    object_handle handle;
}Sphere;

typedef struct Triangle_t{
    glm::vec3 v0,v1,v2;
    glm::vec2 uv0, uv1, uv2;
    material_handle mat_handle;
    object_handle handle;
    bool has_uvs;
}Triangle;

typedef struct Rectangle_t{
    float x0, x1;
    float y0, y1;
    float z0, z1;
    float k;
    ObjectType rect_type;
    object_handle handle;
    material_handle mat_handle;
    int flip_normals;
}Rectangle;

typedef struct Box_t{
    glm::vec3 p0, p1;
    Rectangle rects[6];
    object_handle handle;
    material_handle mat_handle;
    Transforms transforms;
}Box;


//TODO: We might need to some refactoring. It would be nice
//      if we could pass a more complex object (Mesh) or a BVH
//      to the Medium object so that we can fill complex objects
//      that cannot be described by a simple object handler.
typedef struct Medium_t{
    Object geometry;
    object_handle handle;
    material_handle phase_function;
    float density;
}Medium;

typedef struct AABB_t{
    glm::vec3 _min;
    glm::vec3 _max;
}AABB;

/* This is a simple container to keep perlin noise globals */
typedef struct Perlin_t{
    glm::vec3 *ranvec;
    int *permx;
    int *permy;
    int *permz;
    int size;
}Perlin;

typedef struct BVHNode_t{
    AABB box;
    struct BVHNode_t *left, *right;
    Object *handles;
    int n_handles;
    int is_leaf;
}BVHNode;

//TODO: Mesh instancing will require a list of Transformations
//      that are performed on rays and than test with the original
//      bounding box
typedef struct Mesh_t{
    AABB aabb;
    Triangle *triangles;
    int triangles_it;
    BVHNode *bvh;
    
    object_handle handle;
    material_handle mat_handle;
    
    Transforms *instances;
    int instances_it;
}Mesh;


typedef BVHNode * BVHNodePtr;

/* Properties of a scatter */
typedef struct LightEval_t{
    glm::vec3 attenuation;
    glm::vec3 emitted;
    bool has_emission;
}LightEval;

typedef struct SceneHostHelper_t{
    std::vector<Sphere> spheres;
    std::vector<Rectangle> rectangles;
    std::vector<Box> boxes;
    std::vector<Triangle> triangles;
    std::vector<Mesh *> meshes;
    std::vector<Medium> mediums;
    std::vector<Material> materials;
    std::vector<Texture> textures;
    std::vector<Object> samplers;
}SceneHostHelper;

/* Scene defintion, collection of geometry */
typedef struct Scene_t{
    Sphere *spheres;
    int sphere_it;
    int n_spheres;
    
    Rectangle *rectangles;
    int rectangle_it;
    int n_rectangles;
    
    Box *boxes;
    int box_it;
    int n_boxes;
    
    Triangle *triangles;
    int triangles_it;
    int n_triangles;
    
    Mesh **meshes;
    int meshes_it;
    int n_meshes;
    
    Medium *mediums;
    int mediums_it;
    int n_mediums;
    
    Material *material_table;
    int material_it;
    int n_materials;
    
    Object *samplers;
    int samplers_it;
    
    Texture *texture_table;
    texture_handle white_texture;
    texture_handle black_texture;
    
    int texture_it;
    int n_textures;
    BVHNode *bvh;
    Perlin *perlin;
    Camera *camera;
    
    SceneHostHelper *hostHelper;
}Scene;

/* Image definition, collection of rgb pixels */
typedef struct Image_t{
    int width;
    int height;
    int pixels_count;
    glm::vec3 *pixels;
    curandState *states;
}Image;


/* Few utilities */
inline __host__ __device__ float ffmax(float a, float b) { return a > b ? a : b; }
inline __host__ __device__ float ffmin(float a, float b) { return a < b ? a : b; }

inline __host__ __device__ void transform_from_matrixes(Transforms *transform,
                                                        glm::mat4 translate,
                                                        glm::mat4 scale,
                                                        glm::mat4 rotate)
{
    transform->toWorld = translate * scale * rotate;
    transform->toLocal = glm::inverse(rotate) * glm::inverse(scale) * 
        glm::inverse(translate);
    transform->normalMatrix = glm::transpose(glm::inverse(transform->toWorld));
}

inline __host__ __device__ glm::vec3 point_matrix(glm::vec3 p, glm::mat4 model){
    glm::vec4 x(p.x,p.y,p.z,1.0f);
    x = model * x;
    return glm::vec3(x.x,x.y,x.z);
}

inline __host__ __device__ glm::vec3 point_matrix(glm::vec3 p, glm::mat4 m0,
                                                  glm::mat4 m1, glm::mat4 m2)
{
    glm::vec4 x(p.x,p.y,p.z,1.0f);
    x = m0 * m1 * m2 * x;
    return glm::vec3(x.x,x.y,x.z);
}

inline __host__ __device__ glm::vec3 toWorld(glm::vec3 local, Transforms transform){
    glm::vec4 p = transform.toWorld * glm::vec4(local, 1.0f);
    return glm::vec3(p.x, p.y, p.z);
}

inline __host__ __device__ glm::vec3 toWorldDir(glm::vec3 local,
                                                Transforms transform)
{
    glm::vec4 p = transform.toWorld * glm::vec4(local, 0.0f);
    return glm::vec3(p.x, p.y, p.z);
}

inline __host__ __device__ glm::vec3 toLocal(glm::vec3 world, Transforms transform){
    glm::vec4 p = transform.toLocal * glm::vec4(world, 1.0f);
    return glm::vec3(p.x, p.y, p.z);
}

inline __host__ __device__ glm::vec3 toLocalDir(glm::vec3 world,
                                                Transforms transform)
{
    glm::vec4 p = transform.toLocal * glm::vec4(world, 0.0f);
    return glm::vec3(p.x, p.y, p.z);
}

inline __host__ __device__ Ray ray_to_local_space_normalized(Ray ray,
                                                             Transforms transform)
{
    Ray r = ray;
    r.origin = toLocal(ray.origin, transform);
    r.direction = toLocalDir(ray.direction, transform);
    r.direction = glm::normalize(r.direction);
    return r;
}

inline __host__ __device__ Ray ray_to_local_space(Ray ray, Transforms transform)
{
    Ray r = ray;
    r.origin = toLocal(ray.origin, transform);
    r.direction = toLocalDir(ray.direction, transform);
    return r;
}

inline __host__ __device__ void hit_record_remap_to_world(hit_record *record, 
                                                          Transforms transform,
                                                          Ray ray)
{
    glm::vec4 n = transform.normalMatrix * glm::vec4(record->normal, 0.0f);
    record->p = toWorld(record->p, transform);
    record->normal = glm::vec3(n.x, n.y, n.z);
    record->normal = glm::normalize(record->normal);
}

inline __device__ float random_float(curandState *state){
    return glm::max(0.0f, curand_uniform(state)-0.001f);
}


inline __device__ glm::vec3 random_to_sphere(float radius, float d2,
                                             curandState *state)
{
    float r1 = random_float(state);
    float r2 = random_float(state);
    float z = 1 + r2*(sqrt(1-radius*radius/d2) - 1);
    
    float phi = 2*M_PI*r1;
    float x = cos(phi)*sqrt(1-z*z);
    float y = sin(phi)*sqrt(1-z*z);
    
    return glm::vec3(x, y, z);
}


inline __host__ __device__ float length2(glm::vec3 p){
    return (p.x * p.x + p.y * p.y + p.z * p.z);
}

inline __device__ glm::vec3 random_cosine_direction(curandState *state){
    float r1 = random_float(state);
    float r2 = random_float(state);
    float z = glm::sqrt(1.0f - r2);
    float phi = 2.0f * M_PI * r1;
    float x = glm::cos(phi) * glm::sqrt(r2);
    float y = glm::sin(phi) * glm::sqrt(r2);
    return glm::vec3(x,y,z);
}

inline __device__ glm::vec3 random_in_sphere(curandState *state){
    glm::vec3 p;
    do{
        float x = random_float(state);
        float y = random_float(state);
        float z = random_float(state);
        p = 2.0f * glm::vec3(x, y, z) - glm::vec3(1.0f);
    }while(glm::dot(p, p) >= 1.0f);
    
    return glm::normalize(p);
}

inline __device__ glm::vec3 random_in_disk(curandState *state){
    glm::vec3 p;
    do{
        float x = random_float(state);
        float y = random_float(state);
        p = 2.0f * glm::vec3(x, y, 0.0f) - glm::vec3(1.0f,1.0f,0.0f);
    }while(glm::dot(p, p) >= 1.0f);
    
    return glm::normalize(p);
}

inline __host__ __device__ float schlick(float cosine, float ref_idx){
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0*r0;
    return r0 + (1.0f - r0)*glm::pow(1.0f - cosine, 5.0f);
}

inline __host__ __device__ bool refract(glm::vec3 v, glm::vec3 n, 
                                        float ni_over_nt, 
                                        glm::vec3 &refracted)
{
    glm::vec3 uv = glm::normalize(v);
    float dt = glm::dot(uv, n);
    float delta = 1.0 - ni_over_nt*ni_over_nt*(1.0 - dt*dt);
    if(delta > 0.0){
        refracted = ni_over_nt*(uv - n*dt)-n*sqrt(delta);
        return true;
    }
    return false;
}

////////////////////////////////////////////////////////////////////////////////////
#endif