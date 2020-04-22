#include <image.h>
#include <scene.h>
#include <geometry.h>
#include <material.h>
#include <parser_v2.h>
#include <camera.h>
#include <mesh.h>
#include <spectrum.h>
#include <bsdf.h>

#include "fluid_cut.h"

#define OUT_FILE "result.png"
#define BUNNY "/home/felpz/Documents/Bunny-LowPoly.stl"

#define MAX2(a, b) (a) > (b) ? (a) : (b)
#define MAX3(a, b, c) MAX2(MAX2((a), (b)), (c))

inline void perlin_generate(glm::vec3 *p, int size){
    for(int i = 0; i < size; i += 1){
        float x = 2.0f*random_float() - 1.0f;
        float y = 2.0f*random_float() - 1.0f;
        float z = 2.0f*random_float() - 1.0f;
        p[i] = glm::normalize(glm::vec3(x,y,z));
    }
}

inline void permute(int *p, int size){
    for(int i = size-1; i > 0; i --){
        int target = int(random_float() * (i+1));
        int tmp = p[i];
        p[i] = p[target];
        p[target] = tmp;
    }
}

inline void perlin_generate_perm(int *p, int size){
    for(int i = 0; i < size; i += 1){
        p[i] = i;
    }
    permute(p, size);
}

inline __host__ 
void perlin_initialize(Perlin **perlin, int size){
    if(!(*perlin)){
        *perlin = (Perlin *)cudaAllocate(sizeof(Perlin));
        //TODO: Pack
        (*(perlin))->ranvec = (glm::vec3 *)cudaAllocate(size * sizeof(glm::vec3));
        (*(perlin))->permx  = (int *)cudaAllocate(size * sizeof(int));
        (*(perlin))->permy  = (int *)cudaAllocate(size * sizeof(int));
        (*(perlin))->permz  = (int *)cudaAllocate(size * sizeof(int));
    }
    
    (*perlin)->size = size;
    
    perlin_generate((*(perlin))->ranvec, size);
    perlin_generate_perm((*(perlin))->permx, size);
    perlin_generate_perm((*(perlin))->permy, size);
    perlin_generate_perm((*(perlin))->permz, size);
}


__global__ 
void init_random_states(Image *image){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int x = tid % image->width;
    int y = tid / image->width;
    if(x < image->width && y < image->height){
        curand_init(1234, tid, 0, &(image->states[tid]));
    }
}

__bidevice__ 
Spectrum get_sky(glm::vec3 rd){
    Spectrum blue = Spectrum::FromRGB(0.5f, 0.7f, 1.0f);
    glm::vec3 s = glm::normalize(rd);
    float t = 1.2f * (glm::abs(s.y) + 1.0f);
    Spectrum v1 = (1.0f - t) * Spectrum(1.0f);
    Spectrum v2 = t * blue;
    if(v1.HasNaNs()){
        printf("Sky NaN v1 (%g) (%g %g %g)!\n", 1.0f - t, rd.x, rd.y, rd.z);
    }
    
    if(v2.HasNaNs()){
        printf("Sky NaN v2 (%g) (%g %g %g)!\n", t, rd.x, rd.y, rd.z);
    }
    
    return v1 + v2;
}

__device__ Spectrum radiance(Ray source, Scene *scene, curandState *state,
                             int max_bounces)
{
    Spectrum L(0.0f);
    Spectrum beta(1.0f);
    Material *material;
    int total_bounces = max_bounces;
    for(int bounce = 0; bounce < total_bounces; bounce++){
        hit_record record;
        if(!hit_scene(scene, source, 0.0001f, FLT_MAX, &record, state)){
            //L += beta * get_sky(source.direction);
            break;
        }
        
        if(record.mat_handle < scene->material_it){
            material = &scene->material_table[record.mat_handle];
            glm::vec3 wo = -glm::normalize(source.direction);
            glm::vec3 wi;
            glm::vec2 u(random_float(state), random_float(state));
            float pdf = 1.0f;
            
            BxDFType sampled;
            
            BSDF bsdf(record.normal);
            
            material_sample(material, &record, &bsdf, scene);
            
            Spectrum s = BSDF_Sample_f(&bsdf, wo, &wi, u, &pdf, 
                                       BSDF_ALL, &sampled);
            
            if(s.IsBlack() || ABS(pdf) < 0.0001) { break; }
            
            beta *= s * glm::abs(glm::dot(wi, record.normal)) / pdf;
            if(material->has_Le){
                L += beta * material->Le;
                break;
            }
            
            source.origin = record.p;
            source.direction = glm::normalize(wi);
        }else{
            printf("Ooops, hit something without material?\n");
            break;
        }
    }
    
    return L;
}

__device__ glm::vec3 get_color(Ray source, Scene *scene, curandState *state,
                               int max_bounces)
{
    Spectrum r = radiance(source, scene, state, max_bounces);
    glm::vec3 color(0.0f);
    
    if(r.HasNaNs()){
        printf("NaN value on radiance return!\n");
    }
    
    r.ToRGB(&color);
    return color;
}

__global__ void RenderBatch(Image *image, Scene *scene, 
                            int samples, int total_samples)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int x = tid % image->width;
    int y = tid / image->width;
    
    if(x < image->width && y < image->height){
        Camera *camera = scene->camera;
        glm::vec3 color = image->pixels[tid];
        curandState *state = &image->states[tid];
        
        int max_bounces = 50;
        float invSamp = 1.0f / ((float)total_samples);
        for(int i = 0; i < samples; i += 1){
            float u1 = 2.0f * curand_uniform(state);
            float u2 = 2.0f * curand_uniform(state);
            float dx = (u1 < 1.0f) ? sqrt(u1) - 1.0f : 1.0f - sqrt(2.0f - u1);
            float dy = (u2 < 1.0f) ? sqrt(u2) - 1.0f : 1.0f - sqrt(2.0f - u2);
            
            float u = ((float)x + dx) / (float)image->width;
            float v = ((float)y + dy) / (float)image->height;
            Ray r = camera_get_ray(camera, u, v, state);
            glm::vec3 col = get_color(r, scene, state, max_bounces) * invSamp;
            color += col;
        }
        
        image->pixels[tid] = color;
    }
}

void _render_scene(Scene *scene, Image *image, int &samples, int samplesPerBatch){
    size_t threads = 64;
    size_t blocks = (image->pixels_count + threads - 1)/threads;
    
    std::cout << "Generating per pixel RNG seed" << std::endl;
    init_random_states<<<blocks, threads>>>(image);
    cudaSynchronize();
    
    std::cout << "Path tracing... 0%" << std::endl;
    
    int runs = samples / samplesPerBatch;
    int total = 0;
    for(int i = 0; i < runs; i += 1){
        RenderBatch<<<blocks, threads>>>(image, scene, samplesPerBatch,
                                         samples);
        int rv = cudaSynchronize();
        if(rv != 0){
            printf("Cuda Failure. Aborting...\n");
            image_write(image, OUT_FILE, samples);
            exit(0);
        }
        float pct = 100.0f*(float(i + 1)/float(runs));
        std::cout.precision(4);
        std::cout << "Path tracing... " << pct << "%" << std::endl;
        total += samplesPerBatch;
    }
    
    std::cout << std::endl;
    samples = total;
}

void render_scene(Scene *scene, Image *image, int samples, int samplesPerBatch){
    Timed("Rendering", _render_scene(scene, image, samples, samplesPerBatch));
}

int render_bsdf(){
    Image *image = image_new(800, 600);
    Scene *scene = scene_new();
    int samples = 1000;
    int samplesPerBatch = 100;
    
    //glm::vec3 origin = glm::vec3(-25,20,15);
    //glm::vec3 target = glm::vec3(0,5,0);
    //glm::vec3 up = glm::vec3(0.f,1.0f,0.0f);
    
    glm::vec3 origin = glm::vec3(278, 278, -700);
    glm::vec3 target = glm::vec3(278,278,0);
    glm::vec3 up = glm::vec3(0.f,1.0f,0.0f);
    
    scene->perlin = nullptr;
    perlin_initialize(&scene->perlin, 256);
    
    
    texture_handle solidRed = scene_add_texture_solid(scene, glm::vec3(0.65, 0.05, 0.05));
    texture_handle solidGray = scene_add_texture_solid(scene,glm::vec3(0.73, 0.73, 0.73));
    texture_handle solidGreen = scene_add_texture_solid(scene,glm::vec3(0.12, 0.45, 0.15));
    
    texture_handle black = scene_add_texture_solid(scene, glm::vec3(0.f));
    texture_handle white = scene_add_texture_solid(scene, glm::vec3(1.));
    texture_handle kd1 = scene_add_texture_solid(scene, glm::vec3(0.8));
    texture_handle kd2 = scene_add_texture_solid(scene, glm::vec3(0.23f,0.87f,0.0f));
    texture_handle sigma = scene_add_texture_solid(scene, glm::vec3(43.0f));
    
    texture_handle kd = scene_add_texture_solid(scene, glm::vec3(0.82, 0.25, 0.76));
    
    
    material_handle red = scene_add_matte_material(scene, solidRed, sigma);
    material_handle green = scene_add_matte_material(scene, solidGreen, sigma);
    material_handle emit = scene_add_matte_materialLe(scene, white, sigma, 
                                                      Spectrum(10.f));
    material_handle gray = scene_add_matte_material(scene, solidGray, sigma);
    
    
    material_handle bsdf1 = scene_add_matte_material(scene, white, sigma);
    material_handle bsdf2 = scene_add_matte_material(scene, kd2, sigma);
    material_handle bsdf3 = scene_add_plastic_material(scene, kd, kd1, 0.1f);
    material_handle bsdf4 = scene_add_glass_material(scene, black, kd1,
                                                     0.0, 0.0, 1.33f);
    
    
    scene_add_rectangle_yz(scene, 0, 555, 0, 555, 555, green, 1);
    scene_add_rectangle_yz(scene, 0, 555, 0, 555, 0, red);
    scene_add_rectangle_xz(scene, 213, 343, 227, 332, 554, emit, 1, 1);
    scene_add_rectangle_xz(scene, 0, 555, 0, 555, 555, gray, 1);
    scene_add_rectangle_xz(scene, 0, 555, 0, 555, 0, gray);
    scene_add_rectangle_xy(scene, 0, 555, 0, 555, 555, gray, 1);
    
    scene_add_sphere(scene, glm::vec3(190, 90, 190), 90, bsdf4);
    scene_add_box(scene, glm::vec3(357.5, 165.0, 377.5), glm::vec3(165,330,165),
                  glm::vec3(0.0f,15.0f,0.0f), gray);
    
    //scene_add_sphere(scene, glm::vec3(0.0f, 5.0f, -1.0f), 5.0f, bsdf3);
    //scene_add_sphere(scene, glm::vec3(0.f, 15.f, -1.f), 3.f, bsdf5);
    //scene_add_sphere(scene, glm::vec3(0.0f, -500.5f, 0.0f), 500.0f, bsdf2);
    
    
    glm::mat4 translate(1.0f);
    glm::mat4 scale(1.0f);
    glm::mat4 rot(1.0f);
    translate = glm::translate(translate, glm::vec3(1.0f, 4.0f, -3.0f));
    scale = glm::scale(scale, glm::vec3(0.5f));
    
    rot = glm::rotate(rot, glm::radians(0.0f),
                      glm::vec3(0.0f,0.0f,1.0f));
    
    rot = glm::rotate(rot, glm::radians(-90.0f),
                      glm::vec3(0.0f,1.0f,0.0f));
    
    Transforms transform;
    //transform_from_matrixes(&transform, translate, scale, rot);
    transform.toWorld = translate * scale * rot;
    /*
    Mesh *mesh = load_mesh_obj("/home/felpz/Documents/untitled1.obj", bsdf0, transform);
    transform.toWorld = glm::mat4(1.0f);
    scene_add_mesh(scene, mesh, transform);
    
    
    translate = glm::translate(glm::mat4(1.0f), glm::vec3(-11.0f, -0.8f, -1.0f));
    scale = glm::scale(glm::mat4(1.0f), glm::vec3(0.1f));
    rot = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f),
                      glm::vec3(0.0f,1.0f,0.0f));
    rot = glm::rotate(rot, glm::radians(-90.0f),
                      glm::vec3(1.0f,0.0f,0.0f));
                      
                      
    transform.toWorld = translate * scale * rot;
    Mesh *mesh2 = load_mesh_stl(BUNNY, bsdf0, transform);
    transform.toWorld = glm::mat4(1.0f);
    //scene_add_mesh(scene, mesh2, transform);
    */
    
    Timed("Building BVH", scene_build_done(scene));
    float aspect = (float)image->width / (float)image->height;
    float focus_dist = glm::length(origin - target);
    (void)focus_dist;
    
    scene->camera = camera_new(origin, target, up, 45, aspect);
    render_scene(scene, image, samples, samplesPerBatch);
    
    image_write(image, OUT_FILE, samples);
    image_free(image);
    return 0;
}

int main(int argc, char **argv){
    srand(time(0));
    (void)cudaInit();
    return render_bsdf();
}
