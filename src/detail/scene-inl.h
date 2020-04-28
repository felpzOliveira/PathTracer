#if !defined(SCENE_H)
#error "Please include scene.h instead of this file"
#else
#include <types.h>
#include <aabb.h>
#include <bvh.h>
#include <box.h>
#include <cutil.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <glm/gtc/matrix_transform.hpp>


/* Scene and object integration */
inline __host__ 
Scene * scene_new(){
    Scene *scene = (Scene *)cudaAllocate(sizeof(Scene));
    scene->camera = nullptr;
    scene->hostHelper = new SceneHostHelper;
    scene->white_texture = (texture_handle)0;
    scene->black_texture = (texture_handle)1;
    return scene;
}

inline __host__ 
texture_handle scene_add_texture_solid(Scene *scene, glm::vec3 albedo){
    Texture texture;
    TextureProps props;
    props.scale = 1.0f;
    props.wrap_mode = TEXTURE_WRAP_CLAMP;
    texture.color = albedo;
    texture.type = TEXTURE_CONST;
    texture.props = props;
    scene->hostHelper->textures.push_back(texture);
    return (texture_handle)(scene->hostHelper->textures.size() - 1 + 2);
}

inline __host__ 
texture_handle scene_add_texture_noise(Scene *scene, NoiseType ntype,
                                       glm::vec3 albedo)
{
    Texture texture;
    TextureProps props;
    props.scale = 1.0f;
    props.wrap_mode = TEXTURE_WRAP_CLAMP;
    texture.color = albedo;
    texture.type = TEXTURE_NOISE;
    texture.noise_type = ntype;
    texture.props = props;
    scene->hostHelper->textures.push_back(texture);
    return (texture_handle)(scene->hostHelper->textures.size() - 1 + 2);
}



inline __host__ 
texture_handle _scene_add_texture_image(Scene *scene, const char *path,
                                        TextureProps props)
{
    int nx = 0, ny = 0, nn = 0;
    unsigned char *data = stbi_load(path, &nx, &ny, &nn, 0);
    int rv = -1;
    if(data && nx > 0 && ny > 0){
        Texture texture;
        texture.type = TEXTURE_IMAGE;
        texture.image = new unsigned char[nx * ny * 3];
        texture.image_x = nx;
        texture.image_y = ny;
        texture.props = props;
        memcpy(texture.image, data, nx*ny*3);
        scene->hostHelper->textures.push_back(texture);
        rv = scene->hostHelper->textures.size() - 1 + 2;
    }
    return (texture_handle)rv;
}

inline __host__ 
texture_handle scene_add_texture_image(Scene *scene, const char *path){
    TextureProps props;
    props.scale = 1.0f;
    props.wrap_mode = TEXTURE_WRAP_CLAMP;
    return _scene_add_texture_image(scene, path, props);
}

inline __host__ 
texture_handle scene_add_texture_image(Scene *scene, const char *path,
                                       TextureProps props)
{
    return _scene_add_texture_image(scene, path, props);
}


inline __host__ 
texture_handle scene_add_texture_checker(Scene *scene, texture_handle odd,
                                         texture_handle even)
{
    Texture texture;
    TextureProps props;
    props.scale = 1.0f;
    props.wrap_mode = TEXTURE_WRAP_CLAMP;
    texture.type = TEXTURE_CHECKER;
    texture.odd = odd;
    texture.even = even;
    texture.props = props;
    scene->hostHelper->textures.push_back(texture);
    return (texture_handle)(scene->hostHelper->textures.size() - 1 + 2);
}


inline __host__
material_handle scene_add_matte_material(Scene *scene, texture_handle kd,
                                         texture_handle sigma)
{
    Material material;
    material_matte_init(&material, kd, sigma);
    scene->hostHelper->materials.push_back(material);
    return (material_handle)(scene->hostHelper->materials.size() - 1);
}

inline __host__
material_handle scene_add_matte_materialLe(Scene *scene, texture_handle kd,
                                           texture_handle sigma, Spectrum Le)
{
    Material material;
    material_matte_init(&material, kd, sigma);
    material.SetLe(Le);
    scene->hostHelper->materials.push_back(material);
    return (material_handle)(scene->hostHelper->materials.size() - 1);
}

inline __host__
material_handle scene_add_glass_reflector_material(Scene *scene, texture_handle kt,
                                                   texture_handle kr, float index)
{
    Material material;
    material_glass_reflector_init(&material, kt, kr, index);
    scene->hostHelper->materials.push_back(material);
    return (material_handle)(scene->hostHelper->materials.size() - 1);    
}

inline __host__
material_handle scene_add_plastic_material(Scene *scene, texture_handle kd, 
                                           texture_handle ks, float roughness)
{
    Material material;
    material_plastic_init(&material, kd, ks, roughness);
    scene->hostHelper->materials.push_back(material);
    return (material_handle)(scene->hostHelper->materials.size() - 1);
}

inline __host__
material_handle scene_add_mirror_material(Scene *scene, texture_handle kr){
    Material material;
    material_mirror_init(&material, kr);
    scene->hostHelper->materials.push_back(material);
    return (material_handle)(scene->hostHelper->materials.size() - 1);
}

inline __host__
material_handle scene_add_plastic_materialLe(Scene *scene, texture_handle kd, 
                                             texture_handle ks, float roughness,
                                             Spectrum Le)
{
    Material material;
    material_plastic_init(&material, kd, ks, roughness);
    material.SetLe(Le);
    scene->hostHelper->materials.push_back(material);
    return (material_handle)(scene->hostHelper->materials.size() - 1);
}

inline __host__
material_handle scene_add_glass_material(Scene *scene, texture_handle kr, texture_handle kt, float uroughness,
                                         float vroughness, float index)
{
    Material material;
    material_glass_init(&material, kr, kt, uroughness, vroughness, index);
    scene->hostHelper->materials.push_back(material);
    return (material_handle)(scene->hostHelper->materials.size() - 1);
}

inline __host__ 
void _scene_internal_copy_rectangles(Scene *scene){
    int size = scene->hostHelper->rectangles.size();
    scene->rectangle_it = 0;
    if(size > 0){
        size_t memory = sizeof(Rectangle)*size;
        scene->rectangles = (Rectangle *)cudaAllocate(memory);
        scene->n_rectangles = size;
        for(int i = 0; i < size; i += 1){
            Rectangle rect = scene->hostHelper->rectangles[i];
            memcpy(&scene->rectangles[scene->rectangle_it], &rect, sizeof(Rectangle));
            scene->rectangle_it += 1;
        }
        
        scene->hostHelper->rectangles.clear();
    }
}

inline __host__ 
void _scene_internal_copy_boxes(Scene *scene){
    int size = scene->hostHelper->boxes.size();
    scene->box_it = 0;
    if(size > 0){
        size_t memory = sizeof(Box)*size;
        scene->boxes = (Box *)cudaAllocate(memory);
        scene->n_boxes = size;
        for(int i = 0; i < size; i += 1){
            Box box = scene->hostHelper->boxes[i];
            memcpy(&scene->boxes[scene->box_it], &box, sizeof(Box));
            scene->box_it += 1;
        }
        
        scene->hostHelper->boxes.clear();
    }
}

inline __host__ 
void _scene_internal_copy_mediums(Scene *scene){
    int size = scene->hostHelper->mediums.size();
    scene->mediums_it = 0;
    if(size > 0){
        size_t memory = sizeof(Medium)*size;
        scene->mediums = (Medium *)cudaAllocate(memory);
        scene->n_mediums = size;
        for(int i = 0; i < size; i += 1){
            Medium medium = scene->hostHelper->mediums[i];
            memcpy(&scene->mediums[i], &medium, sizeof(Medium));
            scene->mediums_it += 1;
        }
        
        scene->hostHelper->mediums.clear();
    }
}


inline __host__ 
void _scene_internal_copy_triangles(Scene *scene){
    int size = scene->hostHelper->triangles.size();
    scene->triangles_it = 0;
    if(size > 0){
        size_t memory = sizeof(Triangle) * size;
        scene->triangles = (Triangle *)cudaAllocate(memory);
        scene->n_triangles = size;
        for(int i = 0; i < size; i += 1){
            Triangle tri = scene->hostHelper->triangles[i];
            memcpy(&scene->triangles[scene->triangles_it], &tri, sizeof(Triangle));
            scene->triangles_it += 1;
        }
        
        scene->hostHelper->triangles.clear();
    }
}


inline __host__ 
void _scene_internal_copy_spheres(Scene *scene){
    int size = scene->hostHelper->spheres.size();
    scene->sphere_it = 0;
    if(size > 0){
        size_t memory = sizeof(Sphere)*size;
        scene->spheres = (Sphere *)cudaAllocate(memory);
        scene->n_spheres = size;
        for(int i = 0; i < size; i += 1){
            Sphere sph = scene->hostHelper->spheres[i];
            memcpy(&scene->spheres[scene->sphere_it], &sph, sizeof(Sphere));
            scene->sphere_it += 1;
        }
        
        scene->hostHelper->spheres.clear();
    }
}

inline __host__ 
void _scene_internal_copy_meshes(Scene *scene){
    int size = scene->hostHelper->meshes.size();
    scene->meshes_it = 0;
    if(size > 0){
        size_t memory = sizeof(Mesh*)*size;
        scene->meshes = (Mesh **)cudaAllocate(memory);
        scene->n_meshes = size;
        for(int i = 0; i < size; i += 1){
            Mesh *mesh = scene->hostHelper->meshes[i];
            scene->meshes[i] = mesh;
            scene->meshes_it += 1;
        }
        
        scene->hostHelper->meshes.clear();
    }
}

inline __host__ 
void _scene_internal_copy_materials(Scene *scene){
    int size = scene->hostHelper->materials.size();
    if(size > 0){
        size_t memory = sizeof(Material)*size;
        scene->material_table = (Material *)cudaAllocate(memory);
        scene->material_it = 0;
        scene->n_materials = size;
        
        for(int i = 0; i < size; i += 1){
            memcpy(&scene->material_table[scene->material_it], 
                   &scene->hostHelper->materials[i], sizeof(Material));
            scene->material_it += 1;
        }
        
        scene->hostHelper->materials.clear();
    }
}

inline __host__ 
void _scene_internal_copy_textures(Scene *scene){
    int size = scene->hostHelper->textures.size();
    if(size > 0){
        size_t memory = sizeof(Texture) * (size+2);
        scene->texture_table = (Texture *)cudaAllocate(memory);
        scene->texture_it = 0;
        scene->n_textures = size+2;
        
        scene->texture_table[scene->texture_it].color = glm::vec3(1.0f);
        scene->texture_table[scene->texture_it].type  = TEXTURE_CONST;
        scene->texture_it ++;
        
        scene->texture_table[scene->texture_it].color = glm::vec3(0.0f);
        scene->texture_table[scene->texture_it].type  = TEXTURE_CONST;
        scene->texture_it ++;
        
        for(int i = 0; i < size; i += 1){
            Texture texture = scene->hostHelper->textures[i];
            memcpy(&scene->texture_table[scene->texture_it], &texture, sizeof(Texture));
            
            if(texture.type == TEXTURE_IMAGE){
                memory = texture.image_x * texture.image_y * 3;
                scene->texture_table[scene->texture_it].image = (unsigned char *)cudaAllocate(memory);
                memcpy(scene->texture_table[scene->texture_it].image, 
                       texture.image, memory);
                free(texture.image);
            }
            
            scene->texture_it += 1;
        }
        
        scene->hostHelper->textures.clear();
    }
}

//NOTE: Update when adding new primitives
inline __host__ void scene_build_done(Scene *scene){
    //pass things to gpu
    _scene_internal_copy_textures(scene);
    _scene_internal_copy_materials(scene);
    _scene_internal_copy_spheres(scene);
    _scene_internal_copy_rectangles(scene);
    _scene_internal_copy_boxes(scene);
    _scene_internal_copy_triangles(scene);
    _scene_internal_copy_meshes(scene);
    _scene_internal_copy_mediums(scene);
    //build bvh
    int total = scene->sphere_it + scene->rectangle_it +
        scene->box_it + scene->mediums_it +
        scene->triangles_it + scene->meshes_it;
    
    int sph_start    = 0;
    int rect_start   = scene->sphere_it;
    int box_start    = rect_start + scene->rectangle_it;
    int tri_start    = box_start + scene->box_it;
    int mesh_start   = tri_start + scene->triangles_it;
    int medium_start = mesh_start + scene->meshes_it;
    
    Object *handles = new Object[total];
    
    int it = 0;
    for(int i = 0; i < scene->sphere_it; i += 1){
        Sphere *sphere = &scene->spheres[i];
        handles[it].object_type = OBJECT_SPHERE;
        handles[it].handle = sphere->handle;
        handles[it].isvalid = 1;
        handles[it].isbinded = 0;
        it ++;
    }
    
    for(int i = 0; i < scene->rectangle_it; i += 1){
        Rectangle *rect = &scene->rectangles[i];
        handles[it].object_type = rect->rect_type;
        handles[it].handle = rect->handle;
        handles[it].isvalid = 1;
        handles[it].isbinded = 0;
        it ++;
    }
    
    for(int i = 0; i < scene->box_it; i += 1){
        Box *box = &scene->boxes[i];
        handles[it].object_type = OBJECT_BOX;
        handles[it].handle = box->handle;
        handles[it].isvalid = 1;
        handles[it].isbinded = 0;
        it ++;
    }
    
    for(int i = 0; i < scene->triangles_it; i += 1){
        Triangle *tri = &scene->triangles[i];
        handles[it].object_type = OBJECT_TRIANGLE;
        handles[it].handle = tri->handle;
        handles[it].isvalid = 1;
        handles[it].isbinded = 0;
        it ++;
    }
    
    for(int i = 0; i < scene->meshes_it; i += 1){
        Mesh *mesh = scene->meshes[i];
        handles[it].object_type = OBJECT_MESH;
        handles[it].handle = mesh->handle;
        handles[it].isvalid = 1;
        handles[it].isbinded = 0;
        it ++;
    }
    
    for(int i = 0; i < scene->mediums_it; i += 1){
        Object bind;
        Medium *medium = &scene->mediums[i];
        handles[it].object_type = OBJECT_MEDIUM;
        handles[it].handle = medium->handle;
        handles[it].isvalid = 1;
        handles[it].isbinded = 0;
        
        bind = medium->geometry;
        switch(bind.object_type){
            case OBJECT_SPHERE:
            handles[sph_start + bind.handle].isbinded = 1; break;
            
            case OBJECT_XY_RECTANGLE:
            case OBJECT_XZ_RECTANGLE:
            case OBJECT_YZ_RECTANGLE:
            handles[rect_start + bind.handle].isbinded = 1; break;
            
            case OBJECT_BOX:
            handles[box_start + bind.handle].isbinded = 1; break;
            
            case OBJECT_MEDIUM:
            handles[medium_start + bind.handle].isbinded = 1; break;
            
            case OBJECT_TRIANGLE:
            handles[tri_start + bind.handle].isbinded = 1; break;
            
            case OBJECT_MESH:
            handles[mesh_start + bind.handle].isbinded = 1; break;
            
            default:{
                std::cout << "BUG! Unkown object type" << std::endl;
                exit(0);
            }
        }
        it ++;
    }
    
    total = it;
    
    scene->bvh = build_bvh<Scene>(scene, handles, total, 0, BVH_MAX_DEPTH);
    delete[] handles;
    
    scene->samplers_it = 0;
    int size = scene->hostHelper->samplers.size();
    if(size > 0){
        scene->samplers = (Object *)cudaAllocate(size * sizeof(Object));
        memcpy(scene->samplers, scene->hostHelper->samplers.data(), sizeof(Object)*size);
        scene->samplers_it = size;
    }
    
    std::cout << "BVH Node count: " << get_bvh_node_count(scene->bvh) << std::endl;
}

inline __host__ 
Object scene_add_mesh(Scene *scene, Mesh *mesh, Transforms transform){
    Object rv;
    int n = scene->hostHelper->meshes.size();
    mesh->handle = n;
    mesh->instances[0].toWorld = transform.toWorld;
    mesh->instances[0].toLocal = transform.toLocal;
    mesh->instances[0].normalMatrix = transform.normalMatrix;
    mesh->instances_it = 1;
    
    scene->hostHelper->meshes.push_back(mesh);
    rv.isvalid = 1;
    rv.isbinded = 0;
    rv.handle = n;
    rv.object_type = OBJECT_MESH;
    return rv;
}

inline __host__ 
Object scene_add_triangle(Scene *scene, glm::vec3 v0, glm::vec3 v1,
                          glm::vec3 v2, material_handle mat_handle)
{
    Triangle tri;
    Object rv;
    int n = scene->hostHelper->triangles.size();
    tri.v0 = v0;
    tri.v1 = v1;
    tri.v2 = v2;
    tri.uv0 = glm::vec2(0.0f);
    tri.uv1 = glm::vec2(0.0f);
    tri.uv2 = glm::vec2(0.0f);
    tri.has_uvs = false;
    tri.mat_handle = mat_handle;
    tri.handle = n;
    scene->hostHelper->triangles.push_back(tri);
    rv.isvalid = 1;
    rv.isbinded = 0;
    rv.handle = n;
    rv.object_type = OBJECT_TRIANGLE;
    return rv;
}

inline __host__ 
Object scene_add_triangle(Scene *scene, glm::vec3 v0, glm::vec3 v1,
                          glm::vec3 v2, glm::vec2 uv0, glm::vec2 uv1,
                          glm::vec2 uv2, material_handle mat_handle)
{
    Triangle tri;
    Object rv;
    int n = scene->hostHelper->triangles.size();
    tri.v0 = v0;
    tri.v1 = v1;
    tri.v2 = v2;
    tri.uv0 = uv0;
    tri.uv1 = uv1;
    tri.uv2 = uv2;
    tri.has_uvs = true;
    tri.mat_handle = mat_handle;
    tri.handle = n;
    scene->hostHelper->triangles.push_back(tri);
    rv.isvalid = 1;
    rv.isbinded = 0;
    rv.handle = n;
    rv.object_type = OBJECT_TRIANGLE;
    return rv;
}

inline __host__ 
Object scene_add_rectangle_xy(Scene *scene, float x0, float x1,
                              float y0, float y1, float k,
                              material_handle mat_handle,
                              int flip_normals, int sample)
{
    Rectangle rect;
    Object rv;
    int n = scene->hostHelper->rectangles.size();
    xy_rect_set(&rect, x0, x1, y0, y1, k, n, mat_handle, flip_normals);
    scene->hostHelper->rectangles.push_back(rect);
    rv.isvalid = 1;
    rv.isbinded = 0;
    rv.handle = n;
    rv.object_type = OBJECT_XY_RECTANGLE;
    
    if(sample){
        scene->hostHelper->samplers.push_back(rv);
    }
    
    return rv;
}

inline __host__ 
Object scene_add_rectangle_xz(Scene *scene, float x0, float x1,
                              float z0, float z1, float k,
                              material_handle mat_handle,
                              int flip_normals, int sample)
{
    Rectangle rect;
    Object rv;
    int n = scene->hostHelper->rectangles.size();
    xz_rect_set(&rect, x0, x1, z0, z1, k, n, mat_handle, flip_normals);
    scene->hostHelper->rectangles.push_back(rect);
    rv.isvalid = 1;
    rv.isbinded = 0;
    rv.handle = n;
    rv.object_type = OBJECT_XZ_RECTANGLE;
    
    if(sample){
        scene->hostHelper->samplers.push_back(rv);
    }
    
    return rv;
}

inline __host__ 
Object scene_add_rectangle_yz(Scene *scene, float y0, float y1,
                              float z0, float z1, float k,
                              material_handle mat_handle,
                              int flip_normals, int sample)
{
    Rectangle rect;
    Object rv;
    int n = scene->hostHelper->rectangles.size();
    yz_rect_set(&rect, y0, y1, z0, z1, k, n, mat_handle, flip_normals);
    scene->hostHelper->rectangles.push_back(rect);
    rv.isvalid = 1;
    rv.isbinded = 0;
    rv.handle = n;
    rv.object_type = OBJECT_YZ_RECTANGLE;
    
    if(sample){
        scene->hostHelper->samplers.push_back(rv);
    }
    return rv;
}

inline __host__ 
Object scene_add_sphere(Scene *scene, glm::vec3 center, float radius,
                        material_handle mat_handle, int sample)
{
    Sphere sphere;
    Object rv;
    sphere.center = center;
    sphere.radius = radius;
    sphere.handle = scene->hostHelper->spheres.size();
    sphere.mat_handle = mat_handle;
    scene->hostHelper->spheres.push_back(sphere);
    rv.isvalid = 1;
    rv.isbinded = 0;
    rv.handle = sphere.handle;
    rv.object_type = OBJECT_SPHERE;
    
    if(sample){
        scene->hostHelper->samplers.push_back(rv);
    }
    return rv;
}

inline __host__ 
Object scene_add_box(Scene *scene, glm::vec3 p, glm::vec3 scale,
                     glm::vec3 rotation, material_handle mat_handle)
{
    Box box;
    Object rv;
    glm::mat4 mScale = glm::scale(glm::mat4(1.0f), scale);
    glm::mat4 mRotate = glm::rotate(glm::mat4(1.0f), glm::radians(rotation.y),
                                    glm::vec3(0,1,0));
    glm::mat4 mTranslate = glm::translate(glm::mat4(1.0f), p);
    
    box.transforms.toWorld = mTranslate * mScale * mRotate;
    box.transforms.toLocal = glm::inverse(mRotate) * 
        glm::inverse(mScale) * 
        glm::inverse(mTranslate);
    
    box.transforms.normalMatrix = glm::transpose(glm::inverse(box.transforms.toWorld));
    
    //TODO: Should this be transformed?
    glm::vec3 p0 = glm::vec3(-0.5f);
    glm::vec3 p1 = glm::vec3(0.5f);
    
    int n = scene->hostHelper->boxes.size();
    box_set(&box, p0, p1, n, mat_handle);
    scene->hostHelper->boxes.push_back(box);
    rv.isvalid = 1;
    rv.isbinded = 0;
    rv.handle = n;
    rv.object_type = OBJECT_BOX;
    return rv;
}


/*
* NOTE(Felipe): Cannot add a light geometry here. Otherwise lights
* will not be sampled.
*/
inline __host__ 
Object scene_add_medium(Scene *scene, Object geometry,
                        float density, material_handle mat)
{
    Medium medium;
    Object rv;
    medium.geometry = geometry;
    medium.density = density;
    medium.phase_function = mat;
    int n = scene->hostHelper->mediums.size();
    medium.handle = n;
    scene->hostHelper->mediums.push_back(medium);
    rv.isvalid = 1;
    rv.isbinded = 0;
    rv.handle = n;
    rv.object_type = OBJECT_MEDIUM;
    return rv;
}

inline __host__ 
void scene_free(Scene *scene){
    if(scene->spheres) cudaFree(scene->spheres);
    if(scene->rectangles) cudaFree(scene->rectangles);
    if(scene->boxes) cudaFree(scene->boxes);
    if(scene->mediums) cudaFree(scene->mediums);
    if(scene->material_table) cudaFree(scene->material_table);
    if(scene->texture_table) cudaFree(scene->texture_table);
    
    //TODO: free BVH, perlin and camera
}

#endif