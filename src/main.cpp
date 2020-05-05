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
#include <demo_scenes.h>

#include <integrators/path.h>

#include <gr_display.hpp>
#include <gr_opengl.hpp>

#define OUT_FILE "result.png"

#define MAX2(a, b) (a) > (b) ? (a) : (b)
#define MAX3(a, b, c) MAX2(MAX2((a), (b)), (c))

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
    return Spectrum(0.f);
    Spectrum blue = Spectrum::FromRGB(0.123f, 0.34f, 0.9f);
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

__device__
bool IsOccluded(glm::vec3 p, glm::vec3 xl, Scene *scene, curandState *state){
    Ray ray;
    hit_record tmp;
    float refd = glm::distance(p, xl);
    ray.origin = p;
    ray.direction = glm::normalize(xl - p);
    return hit_scene(scene, ray, 0.0001f, refd - 0.0001f, &tmp, state);
}

__device__
Spectrum direct_light(Scene *scene, hit_record *record, Light *light,
                      glm::vec2 uLight, glm::vec2 uDir, curandState *state,
                      BSDF *bsdf, glm::vec3 wo, bool specular = false)
{
    BxDFType bsdfFlags = specular ? BSDF_ALL : BxDFType(BSDF_ALL & ~BSDF_SPECULAR);
    Spectrum Ld(0.f);
    float lightPdf = 0.f;
    //float scatteringPdf = 0.f;
    glm::vec3 xl, wi;
    
    Spectrum Li = Light_Sample_Li(light, scene, record, uLight, &wi, 
                                  &lightPdf, &xl);
    
    if(lightPdf > 0.f && !Li.IsBlack()){
        Spectrum f(0.f);
        f = BSDF_f(bsdf, wo, wi, bsdfFlags) * AbsDot(wi, record->normal);
        //scatteringPdf = BSDF_Pdf(bsdf, wo, wi, bsdfFlags);
        
        if(!f.IsBlack()){
            if(IsOccluded(record->p, xl, scene, state)){
                Li = Spectrum(0.f);
            }
            
            if(!Li.IsBlack()){
                if(IsDeltaLight(light->flags)){
                    Ld += f * Li / lightPdf;
                }else{
                    printf("?????\n");
                }
            }
        }
    }
    
    return Ld;
}

__device__
Spectrum sample_light(Scene *scene, hit_record *record, curandState *state,
                      BSDF *bsdf, glm::vec3 wo)
{
    int num_lights = scene->lights_it;
    //If there is no samplers return 0
    if(num_lights == 0) return Spectrum(0.f);
    
    //chose a single sampler for sampling with a uniform pdf
    float light_rng = random_float(state);
    int chosen_light = 0;
    float light_pdf = 0.f;
    chosen_light = MIN((int)(light_rng * num_lights), num_lights - 1);
    light_pdf = 1.f / (float)num_lights;
    
    Light *light = &scene->lights[chosen_light];
    glm::vec2 uLight(random_float(state), random_float(state));
    glm::vec2 uDir(random_float(state), random_float(state));
    return direct_light(scene, record, light, uLight, uDir,
                        state, bsdf, wo) / light_pdf;
}

#if 1
//NOTE: This one solves Path Tracing without sampling
__device__
Spectrum trace_single(Ray &source, Scene *scene, curandState *state){
    hit_record record;
    if(!hit_scene(scene, source, 0.0001f, FLT_MAX, &record, state)){
        source.alive = 0;
        return get_sky(source.direction);
    }
    
    Material *material = &scene->material_table[record.mat_handle];
    glm::vec3 wo = -glm::normalize(source.direction);
    glm::vec3 wi;
    glm::vec2 u(random_float(state), random_float(state));
    float pdf = 1.0f;
    
    BxDFType sampled;
    
    BSDF bsdf(record.normal);
    
    material_sample(material, &record, &bsdf, scene);
    
    Spectrum s = BSDF_Sample_f(&bsdf, wo, &wi, u, &pdf, 
                               BSDF_ALL, &sampled);
    
    if(s.IsBlack() || ABS(pdf) < 0.0001){
        source.energy = Spectrum(0.f);
        return Spectrum(0.f);
    }
    
    source.origin = record.p;
    source.direction = glm::normalize(wi);
    source.energy *= s * glm::abs(glm::dot(source.direction, record.normal)) / pdf;
    return material->Le;
}

#else
//NOTE: Attempting to implement light sampling, again!
static __device__
Spectrum trace_single(Ray &source, Scene *scene, curandState *state){
    hit_record record;
    
    if(!hit_scene(scene, source, 0.0001f, FLT_MAX, &record, state)){
        source.alive = 0;
        return get_sky(source.direction);
    }
    
    Material *material = &scene->material_table[record.mat_handle];
    glm::vec3 wo = -glm::normalize(source.direction);
    glm::vec3 wi;
    glm::vec2 u(random_float(state), random_float(state));
    //float pdf = 1.0f;
    
    //BxDFType sampled;
    
    BSDF bsdf(record.normal);
    material_sample(material, &record, &bsdf, scene);
    return sample_light(scene, &record, state, &bsdf, wo);
}

#endif


__device__ 
glm::vec3 get_color(Ray source, Scene *scene, curandState *state,
                    int max_bounces)
{
    Spectrum result(0.f);
    source.energy = Spectrum(1.f);
    source.alive = 1;
    for(int i = 0; i < max_bounces; i++){
        result += source.energy * trace_single(source, scene, state);
        if(source.energy.IsBlack()) break;
    }
    
    if(result.HasNaNs()){
        printf("NaN value on radiance!\n");
        result = Spectrum(1.f); //this makes the pixel visible
    }
    
    return result.ToRGB();
}

__global__ void RenderBatch(Image *image, Scene *scene, int samples, int total_samples){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int x = tid % image->width;
    int y = tid / image->width;
    
    if(x < image->width && y < image->height){
        Camera *camera = scene->camera;
        glm::vec3 color = image->pixels[tid].color;
        glm::vec3 rawColor = image->pixels[tid].progColor;
        int psmp = image->pixels[tid].samples;
        curandState *state = &image->states[tid];
        
        float inv = 1.f/(float)total_samples;
        int max_bounces = 20;
        //int max_bounces = 1;
        
        for(int i = 0; i < samples; i += 1){
            float u1 = 2.0f * curand_uniform(state);
            float u2 = 2.0f * curand_uniform(state);
            float dx = (u1 < 1.0f) ? sqrt(u1) - 1.0f : 1.0f - sqrt(2.0f - u1);
            float dy = (u2 < 1.0f) ? sqrt(u2) - 1.0f : 1.0f - sqrt(2.0f - u2);
            
            float u = ((float)x + dx) / (float)image->width;
            float v = ((float)y + dy) / (float)image->height;
            Ray r = camera_get_ray(camera, u, v, state);
            glm::vec3 col = get_color(r, scene, state, max_bounces);
            color += col * inv;
            rawColor += col;
            psmp += 1;
        }
        
        image->pixels[tid].samples = psmp;
        image->pixels[tid].color = color;
        image->pixels[tid].progColor = rawColor;
    }
}

enum ToneMapAlgorithm{
    Reinhard,
    Exponential,
    NaughtyDog
};

inline __bidevice__
glm::vec3 ReinhardMap(glm::vec3 value, float exposure){
    (void)exposure;
    return (value / (value + 1.f));
}

inline __bidevice__
glm::vec3  NaughtyDogMap(glm::vec3 value, float exposure){
    float A = 0.15f;
    float B = 0.50f;
    float C = 0.10f;
    float D = 0.20f;
    float E = 0.02f;
    float F = 0.30f;
    float W = 11.2f;
    value *= exposure;
    value = ((value * (A*value+C*B)+D*E)/(value*(A*value+B)+D*F))-E/F;
    float white = ((W*(A*W+C*B)+D*E)/(W*(A*W+B)+D*F))-E/F;
    value /= white;
    return value;
}

inline __bidevice__
glm::vec3 ExponentialMap(glm::vec3 value, float exposure){
    return (glm::vec3(1.f) - glm::exp(-value * exposure));
}

/*
 * Tone mapping is essential when doing PBR since values *will* be 
 * outside 0-1 range.
*/
__global__ void ToneMap(Image *image, int exposure, ToneMapAlgorithm algorithm){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int x = tid % image->width;
    int y = tid / image->width;
    
    if(x < image->width && y < image->height){
        glm::vec3 color = image->pixels[tid].color;
        glm::vec3 mapped = color;
        switch(algorithm){
            case ToneMapAlgorithm::Reinhard: {
                mapped = ReinhardMap(mapped, exposure);
            } break;
            
            case ToneMapAlgorithm::NaughtyDog:{
                mapped = NaughtyDogMap(mapped, exposure);
            } break;
            
            case ToneMapAlgorithm::Exponential:{
                mapped = ExponentialMap(color, exposure);
            } break;
            
            default:{
                printf("Unknown algorithm\n");
            }
        }
        
        image->pixels[tid].color = mapped;
    }
}

void _render_display_pixels(Image *image, gr_display **display, float *rgb){
    int it = 0;
    
    if(!*display){
        *display = gr_new_display(image->width, image->height);
    }
    
    for(int k = 0; k < image->pixels_count; k++){
        glm::vec3 c(0.f);
        int samples = image->pixels[k].samples;
        c = image->pixels[k].progColor;
        float iv = 1.f / (float)samples;
        rgb[it++] = c[0] * iv;
        rgb[it++] = c[1] * iv;
        rgb[it++] = c[2] * iv;
    }
    
    //NOTE: Graphy is applying the Naughty Dog algorithm
    //      for automatic Tone Mapping, no need to perform it here.
    gr_opengl_render_pixels(rgb, image->width, image->height, *display);
}

void _render_scene(Scene *scene, Image *image, int &samples, int samplesPerBatch){
    gr_display *display = nullptr;
    float *rgb = new float[image->pixels_count * 3];
    size_t threads = 64;
    size_t blocks = (image->pixels_count + threads - 1)/threads;
    
    std::cout << "Generating per pixel RNG seed" << std::endl;
    init_random_states<<<blocks, threads>>>(image);
    cudaSynchronize();
    
    std::cout << "Path tracing... 0%" << std::endl;
    int rv = 0;
    int runs = samples / samplesPerBatch;
    int total = 0;
    for(int i = 0; i < runs; i += 1){
        RenderBatch<<<blocks, threads>>>(image, scene, samplesPerBatch, samples);
        rv = cudaSynchronize();
        if(rv != 0){
            printf("Cuda Failure. Aborting...\n");
            image_write(image, OUT_FILE, samples);
            cudaSafeExit();
            exit(0);
        }
        
        _render_display_pixels(image, &display, rgb);
        
        float pct = 100.0f*(float(i + 1)/float(runs));
        std::cout.precision(4);
        std::cout << "Path tracing... " << pct << "%" << std::endl;
        total += samplesPerBatch;
    }
    
    samples = total;
    
    std::cout << std::endl;
    std::cout << "Tone Mapping..." << std::flush;
    ToneMap<<<blocks, threads>>>(image, 2.f, ToneMapAlgorithm::Exponential);
    rv = cudaSynchronize();
    if(rv != 0){
        std::cout << "Cuda Failure. Aborting..." << std::endl;
        image_write(image, OUT_FILE, samples);
        cudaSafeExit();
        exit(0);
    }
    
    _render_display_pixels(image, &display, rgb);
    
    delete[] rgb;
    std::cout << "OK" << std::endl;
}

void render_scene(Scene *scene, Image *image, int samples, int samplesPerBatch){
    Timed("Rendering", _render_scene(scene, image, samples, samplesPerBatch));
}

inline __host__
Scene *scene_cornell(float aspect){
    Scene *scene = scene_new();
    glm::vec3 origin(0.f,5.f, 25.f);
    glm::vec3 target(0.f, 5.f, 0.f);
    glm::vec3 up(0.f, 1.f, 0.f);
    
    scene->perlin = nullptr;
    perlin_initialize(&scene->perlin, 256);
    Spectrum spec_emit = Spectrum::FromRGB(10.f,10.f,9.41f)*0.1f;
    
    texture_handle whiteTex = scene_add_texture_solid(scene, glm::vec3(7.f));
    texture_handle grayTex = scene_add_texture_solid(scene, glm::vec3(0.8f));
    texture_handle redTex = scene_add_texture_solid(scene, glm::vec3(0.57f,0.025f,0.025f));
    texture_handle greenTex = scene_add_texture_solid(scene, glm::vec3(0.025f,0.236f,0.025f));
    
    material_handle emit = scene_add_matte_materialLe(scene, whiteTex, whiteTex, spec_emit);
    material_handle redMat = scene_add_matte_material(scene, redTex, redTex);
    material_handle greenMat = scene_add_matte_material(scene, greenTex, greenTex);
    material_handle ballmat = scene_add_plastic_material(scene, grayTex, grayTex, 0.03f);
    
    scene_add_sphere(scene, glm::vec3(0.f, 2.f, 0.f), 2.f, ballmat);
    scene_add_rectangle_xz(scene, -500.f, 500.f, -500.f, 500.f, 0.f, ballmat);
    scene_add_rectangle_yz(scene, -500.f, 500.f, -500.f, 500.f, -8.f, redMat);
    scene_add_rectangle_yz(scene, -500.f, 500.f, -500.f, 500.f, 8.f, greenMat);
    scene_add_rectangle_xy(scene, -500.f, 500.f, -500.f, 500.f, -6.f, ballmat);
    scene_add_rectangle_xz(scene, -500.f, 500.f, -500.f, 500.f, 12.f, ballmat);
    
    scene_add_rectangle_xz(scene, -6.f, 6.f, -3.f, 3.f, 12.f, emit);
    //scene_add_point_light(scene, glm::vec3(0.f, 10.f, 0.f), spec_emit);
    
    Timed("Building BVH", scene_build_done(scene));
    float focus_dist = glm::length(origin - target);
    (void)focus_dist;
    
    scene->camera = camera_new(origin, target, up, 45, aspect);
    return scene;
}


int main(int argc, char **argv){
    srand(time(0));
    (void)cudaInit();
    
    Image *image = image_new(800, 600);
    int samples = 100;
    int samplesPerBatch = 10;
    
    float aspect = (float)image->width / (float)image->height;
    //Scene *scene = scene_bsdf(aspect);
    //Scene *scene = scene_basic(aspect);
    Scene *scene = scene_cornell(aspect);
    render_scene(scene, image, samples, samplesPerBatch);
    
    image_write(image, OUT_FILE, samples);
    image_free(image);
    cudaSafeExit();
    return 0;
}
