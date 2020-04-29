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
Spectrum trace_shadow(Scene *scene, hit_record *record, 
                      Object *light, glm::vec2 u, curandState *state)
{
    Spectrum r(0.f);
    Ray shadow;
    shadow.origin = record->p;
    float pdf = 0.f;
    glm::vec3 xp = sample_object(scene, *light, u, &pdf);
    shadow.direction = glm::normalize(xp - record->p);
    if(pdf > 0.f){
        hit_record tmp;
        bool hit = hit_scene(scene, shadow, 0.0001f, FLT_MAX, &tmp, state);
        if(hit){
            float refd = glm::distance(xp, record->p);
            r = Spectrum(1.f);
            if(tmp.t < refd - 0.001f){
                r = Spectrum(0.f);
            }else{
                glm::vec3 lightNormal(0.f,0.f,1.f);
                float cos0 = glm::dot(record->normal, shadow.direction);
                float cos1 = glm::dot(-lightNormal, shadow.direction);
                Material *mat = &scene->material_table[tmp.mat_handle];
                r = mat->Le * cos0 * cos1/(refd * refd);
            }
        }
    }
    
    return r;
}

#if 0
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
    
    
    u = glm::vec2(random_float(state), random_float(state));
    return trace_shadow(scene, &record, &scene->samplers[0], u, state) + material->Le;
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
        Spectrum Le = trace_single(source, scene, state);
        result += source.energy * Le;
        if(source.energy.IsBlack() || !Le.IsBlack()) break;
    }
    
    if(result.HasNaNs()){
        printf("NaN value on radiance!\n");
        result = Spectrum(1.f); //this makes the pixel visible
    }
    
    return result.ToRGB();
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
        
        //int max_bounces = 20;
        int max_bounces = 1;
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

__global__ void ToneMap(Image *image, int exposure, ToneMapAlgorithm algorithm){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int x = tid % image->width;
    int y = tid / image->width;
    
    if(x < image->width && y < image->height){
        glm::vec3 color = image->pixels[tid];
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
        
        image->pixels[tid] = mapped;
    }
}

void _render_scene(Scene *scene, Image *image, int &samples, int samplesPerBatch){
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
        RenderBatch<<<blocks, threads>>>(image, scene, samplesPerBatch,
                                         samples);
        rv = cudaSynchronize();
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
    
    samples = total;
    
    std::cout << std::endl;
    std::cout << "Tone Mapping..." << std::flush;
    ToneMap<<<blocks, threads>>>(image, 2.f, ToneMapAlgorithm::Exponential);
    rv = cudaSynchronize();
    if(rv != 0){
        std::cout << "Cuda Failure. Aborting..." << std::endl;
        image_write(image, OUT_FILE, samples);
        exit(0);
    }
    
    std::cout << "OK" << std::endl;
}

void render_scene(Scene *scene, Image *image, int samples, int samplesPerBatch){
    Timed("Rendering", _render_scene(scene, image, samples, samplesPerBatch));
}

int main(int argc, char **argv){
    srand(time(0));
    (void)cudaInit();
    Image *image = image_new(800, 600);
    int samples = 100;
    int samplesPerBatch = 10;
    
    float aspect = (float)image->width / (float)image->height;
    //Scene *scene = scene_bsdf(aspect);
    Scene *scene = scene_basic(aspect);
    render_scene(scene, image, samples, samplesPerBatch);
    
    image_write(image, OUT_FILE, samples);
    image_free(image);
    return 0;
}
