#include <image.h>
#include <scene.h>
#include <geometry.h>
#include <material.h>
#include <parser_v2.h>
#include <camera.h>
#include <mesh.h>
#include <spectrum.h>
#include <bsdf.h>

#include <lbvh.h>

#include "fluid_cut.h"

#define OUT_FILE "result.png"
#define BUNNY "/home/felpz/Documents/Bunny-LowPoly.stl"
#define LOW_POLY_DRAGON "/home/felpz/Documents/untitled1.obj"
#define DEMON "/home/felpz/Documents/demon.obj"

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
        
        int max_bounces = 20;
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

int render_bsdf(){
    Image *image = image_new(800, 600);
    Scene *scene = scene_new();
    int samples = 500;
    int samplesPerBatch = 10;
    
    //glm::vec3 origin = glm::vec3(-25,20,15);
    //glm::vec3 target = glm::vec3(0,5,0);
    //glm::vec3 up = glm::vec3(0.f,1.0f,0.0f);
    
    //glm::vec3 origin = glm::vec3(0.5f,4.f,-7.f);
    glm::vec3 origin = glm::vec3(1.f);
    glm::vec3 target = glm::vec3(0,0,0.f);
//    glm::vec3 origin = glm::vec3(278, 278, -700);
    //glm::vec3 target = glm::vec3(278,278,0);
    glm::vec3 up = glm::vec3(0.f,1.0f,0.0f);
    
    scene->perlin = nullptr;
    perlin_initialize(&scene->perlin, 256);
    
    
    texture_handle solidRed = scene_add_texture_solid(scene, glm::vec3(0.65, 0.05, 0.05));
    texture_handle solidGray = scene_add_texture_solid(scene,glm::vec3(0.8f));
    texture_handle solidGreen = scene_add_texture_solid(scene,glm::vec3(0.12, 0.45, 0.15));
    
    texture_handle black = scene_add_texture_solid(scene, glm::vec3(0.f));
    texture_handle white = scene_add_texture_solid(scene, glm::vec3(1.));
    texture_handle kd1 = scene_add_texture_solid(scene, glm::vec3(0.8));
    texture_handle kd2 = scene_add_texture_solid(scene, glm::vec3(.4f));
    texture_handle sigma = scene_add_texture_solid(scene, glm::vec3(43.0f));
    
    texture_handle kd = scene_add_texture_solid(scene, glm::vec3(0.7,0.7,0.658));
    texture_handle ks = scene_add_texture_solid(scene, glm::vec3(0.7,0.7,0.658));
    texture_handle k2 = scene_add_texture_solid(scene, glm::vec3(0.4));

    texture_handle kdgreen = scene_add_texture_solid(scene, glm::vec3(0.6784313725490196, 1.f, 0.1843137254901961));
    
    
    Spectrum spec_emit = Spectrum::FromRGB(10.f,10.f,9.41f)*3.f;
    texture_handle milk = scene_add_texture_solid(scene, glm::vec3(1.f,1.f,0.941f));
    
    material_handle red = scene_add_matte_material(scene, solidRed, sigma);
    material_handle green = scene_add_matte_material(scene, solidGreen, sigma);
    material_handle emit = scene_add_matte_materialLe(scene, white, sigma, 
                                                      spec_emit);
    material_handle gray = scene_add_matte_material(scene, solidGray, sigma);
    
    
    material_handle bsdf1 = scene_add_matte_material(scene, white, sigma);
    material_handle bsdf2 = scene_add_plastic_material(scene, kd2, kd2, 0.03);
    material_handle bsdf3 = scene_add_plastic_material(scene, kd, ks, 0.1f);
    material_handle bsdf6 = scene_add_plastic_material(scene, k2, k2, 0.03f);
    material_handle bsdfPlastic = scene_add_plastic_material(scene, kdgreen,
                                                             kdgreen, 0.03f);
    material_handle bsdf4 = scene_add_glass_material(scene, black, kd1,
                                                     0.0, 0.0, 1.33f);
    
    material_handle glass2 = scene_add_glass_reflector_material(scene, milk, milk, 1.33f);
    
    material_handle mirror = scene_add_mirror_material(scene, kd1);
    
#if 0
    scene_add_rectangle_yz(scene, 0, 555, 0, 555, 555, green, 1);
    scene_add_rectangle_yz(scene, 0, 555, 0, 555, 0, red);
    scene_add_rectangle_xz(scene, 213, 343, 227, 332, 554, emit, 1, 1);
    scene_add_rectangle_xz(scene, 0, 555, 0, 555, 555, gray, 1);
    scene_add_rectangle_xz(scene, 0, 555, 0, 555, 0, gray);
    scene_add_rectangle_xy(scene, 0, 555, 0, 555, 555, gray, 1);
    
    scene_add_sphere(scene, glm::vec3(190, 90, 190), 90, glass2);
    scene_add_box(scene, glm::vec3(357.5, 165.0, 377.5), glm::vec3(165,330,165),
                  glm::vec3(0.0f,15.0f,0.0f), gray);
#endif

                                  
    glm::mat4 translate(1.0f);
    glm::mat4 scale(1.0f);
    glm::mat4 rot(1.0f);
    Transforms transform;
    
    /* Dragon mesh (reduced)
    translate = glm::translate(translate, glm::vec3(1.0f, 0.8f, -3.0f));
    scale = glm::scale(scale, glm::vec3(0.1f));
    origin = glm::vec3(1.f,4.f,-8.5f);
    rot = glm::rotate(rot, glm::radians(240.0f),
                      glm::vec3(0.0f,1.0f,0.0f));
                      
    transform.toWorld = translate * scale * rot;
    
    Mesh *mesh = load_mesh_obj(LOW_POLY_DRAGON, bsdfPlastic, transform);
    */
    
//    Mesh *mesh = load_mesh_obj(DEMON, bsdfPlastic, transform);
    
    //target = mesh->bvh->box.centroid;
    //transform.toWorld = glm::mat4(1.0f);
//    scene_add_mesh(scene, mesh, transform);
    
    
#if 0
    struct sph{
        float rad;
        glm::vec3 c;
    };
    
    std::vector<sph> pos;
    
    float boxlen = aabb_max_length(mesh->bvh->box)/2.f;
    
    auto test = [&](sph c){
        for(sph &s : pos){
            float d = glm::distance(s.c, c.c);
            float d2 = glm::distance(c.c, mesh->bvh->box.centroid);
            if(d < s.rad + c.rad || d2 < boxlen - c.rad) return false;
        }
        
        return true;
    };
    
    for(int i = 0; i < 80; i++){
        float rad = 0.f;
        glm::vec3 c(0.f);
        float g = 0.f;
        bool ok = false;
        sph pp;
        while(!ok){
            rad = 1.f + 2.f * random_float();
            c = glm::vec3(-20.f - 40.f * random_float(),
                          rad, -20.f + 40.f * random_float());
            pp.rad = rad;
            pp.c = c;
            ok = test(pp);
        }
        
        pos.push_back(pp);
        
        g = random_float();
        
        glm::vec3 c0(random_float(),random_float(),random_float());
        glm::vec3 c1(random_float(),random_float(),random_float());
        if(g < 0.5){
            float t = 0.1f * random_float();
            texture_handle hnd = scene_add_texture_solid(scene, c0);
            texture_handle hnd2 = scene_add_texture_solid(scene, c1);
            material_handle mat = scene_add_plastic_material(scene, hnd, hnd2, t);
            scene_add_sphere(scene, c, rad, mat);
        }else if(g < 0.85){
            g *= 10.f;
            texture_handle hnd = scene_add_texture_solid(scene, c0);
            texture_handle sig = scene_add_texture_solid(scene, glm::vec3(g));
            material_handle mat = scene_add_matte_material(scene, hnd, sig);
            scene_add_sphere(scene, c, rad, mat);
        }else if(g < 0.9){
            texture_handle hnd = scene_add_texture_solid(scene, c0);
            material_handle mat = scene_add_mirror_material(scene, hnd);
            scene_add_sphere(scene, c, rad, mat);
        }else{
            Spectrum Le = Spectrum::FromRGB(c0);
            texture_handle hnd = scene_add_texture_solid(scene, c0);
            material_handle emit2 = scene_add_matte_materialLe(scene, hnd, sigma, Le);
            scene_add_sphere(scene, c, rad, emit2);
        }
    }
    
    glm::vec3 v(0.f);
    for(sph &c : pos){
        v += c.c;
    }
    
    v /= (float)pos.size();
    
    target = v;
#endif
    
    /*
    samples = 100;
    samplesPerBatch = 10;
    origin = glm::vec3(5.f, 1.f, 5.f);
    target = glm::vec3(0.f,1.f,0.f);
    scene_add_sphere(scene, glm::vec3(0.0f, 1.f, 0.0f), 1.f, bsdf6);
    */
    scene_add_sphere(scene, glm::vec3(0.15f, 0.25f, 0.f), 0.25f, mirror);
    scene_add_sphere(scene, glm::vec3(-0.5f, 0.15f, 0.f), 0.15f, bsdfPlastic);
    scene_add_rectangle_xz(scene, -500, 500, -500, 500, 0.f, bsdf3);
    float w = 10.f;
    glm::vec3 v(0.f, 5.f, 0.f);
    
    //scene_add_rectangle_xz(scene, v.x-w, v.x+w, v.z-w, v.z+w, 10.f, emit);
    scene_add_sphere(scene, glm::vec3(4.f, 5.f, 2.f), 1.f, emit);
    
    
/*                          
    translate = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.8f, .0f));
    scale = glm::scale(glm::mat4(1.0f), glm::vec3(0.02f));
    rot = glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f),
                      glm::vec3(0.0f,1.0f,0.0f));
    rot = glm::rotate(rot, glm::radians(-90.0f),
                      glm::vec3(1.0f,0.0f,0.0f));
                      

    transform.toWorld = translate * scale * rot;
    Mesh *mesh2 = load_mesh_stl(BUNNY, red, transform);
    transform.toWorld = glm::mat4(1.0f);
    scene_add_mesh(scene, mesh2, transform);
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

int change_val(int &s){
    s += 1;
    return s;
}

int main(int argc, char **argv){
    srand(time(0));
    (void)cudaInit();
    return render_bsdf();
}
