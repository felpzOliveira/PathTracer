#if !defined(SCENE_H)
#define SCENE_H
#include <types.h>
#include <cutil.h>
#include <material.h>

inline __host__ 
Camera * camera_new(glm::vec3 origin, glm::vec3 at, glm::vec3 up, 
                    float vfov, float aspect);

inline __host__ 
Camera * camera_new(glm::vec3 origin, glm::vec3 at, glm::vec3 up,
                    float vfov, float aspect, float aperture, float focus_dist);


inline __host__ 
Scene * scene_new();

inline __host__ 
void scene_free(Scene *scene);


inline __host__ 
texture_handle scene_add_texture_solid(Scene *scene, glm::vec3 albedo);

inline __host__ 
texture_handle scene_add_texture_noise(Scene *scene, NoiseType ntype,
                                       glm::vec3 albedo);

inline __host__ 
texture_handle scene_add_texture_image(Scene *scene, const char *path);

inline __host__ 
texture_handle scene_add_texture_image(Scene *scene, const char *path,
                                       TextureProps props);

inline __host__ 
texture_handle scene_add_texture_checker(Scene *scene, texture_handle odd,
                                         texture_handle even);

inline __host__
material_handle scene_add_matte_material(Scene *scene, texture_handle kd,
                                         texture_handle sigma);

inline __host__
material_handle scene_add_mirror_material(Scene *scene, texture_handle kr);

inline __host__
material_handle scene_add_matte_materialLe(Scene *scene, texture_handle kd,
                                           texture_handle sigma, Spectrum Le);

inline __host__
material_handle scene_add_plastic_material(Scene *scene, texture_handle kd, 
                                           texture_handle ks, float roughness);

inline __host__
material_handle scene_add_plastic_materialLe(Scene *scene, texture_handle kd, 
                                             texture_handle ks, float roughness,
                                             Spectrum Le);

inline __host__
material_handle scene_add_glass_material(Scene *scene, texture_handle kr,
                                         texture_handle kt, float uroughness,
                                         float vroughness, float index);

inline __host__
material_handle scene_add_glass_reflector_material(Scene *scene, texture_handle kt,
                                                   texture_handle kr, float index);

/////////////////OBJECT
inline __host__
Object scene_add_mesh(Scene *scene, Mesh *mesh, Transforms transform);

inline __host__ 
Object scene_add_triangle(Scene *scene, glm::vec3 v0, glm::vec3 v1,
                          glm::vec3 v2, material_handle mat_handle);

inline __host__ 
Object scene_add_triangle(Scene *scene, glm::vec3 v0, glm::vec3 v1,
                          glm::vec3 v2, glm::vec2 uv0, glm::vec2 uv1,
                          glm::vec2 uv2, material_handle mat_handle);

inline __host__ 
Object scene_add_rectangle_xy(Scene *scene, float x0, float x1,
                              float y0, float y1, float k,
                              material_handle mat_handle,
                              int flip_normals=0, int sample = 0);

inline __host__ 
Object scene_add_rectangle_xz(Scene *scene, float x0, float x1,
                              float z0, float z1, float k,
                              material_handle mat_handle,
                              int flip_normals=0, int sample=0);


inline __host__ 
Object scene_add_rectangle_yz(Scene *scene, float y0, float y1,
                              float z0, float z1, float k,
                              material_handle mat_handle,
                              int flip_normals=0, int sample = 0);

inline __host__ 
Object scene_add_sphere(Scene *scene, glm::vec3 center, float radius,
                        material_handle mat_handle, int sample = 0);

inline __host__ 
Object scene_add_box(Scene *scene, glm::vec3 p, glm::vec3 scale,
                     glm::vec3 rotation, material_handle mat_handle);

inline __host__ 
Object scene_add_medium(Scene *scene, Object geometry,
                        float density, material_handle mat);

#include <detail/scene-inl.h>

#endif