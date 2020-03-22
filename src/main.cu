#include <image.h>
#include <scene.h>
#include <geometry.h>
#include <cuda_util.cuh>
#include <material.h>
#include <parser_v2.h>
#define OUT_FILE "result.png"

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

inline __host__ void perlin_initialize(Perlin **perlin, int size){
    if(!(*perlin)){
        CHECK(cudaMallocManaged(perlin, sizeof(Perlin)));
        //TODO: Pack
        CHECK(cudaMallocManaged(&(*(perlin))->ranvec, size * sizeof(glm::vec3)));
        CHECK(cudaMallocManaged(&(*(perlin))->permx, size * sizeof(int)));
        CHECK(cudaMallocManaged(&(*(perlin))->permy, size * sizeof(int)));
        CHECK(cudaMallocManaged(&(*(perlin))->permz, size * sizeof(int)));
    }
    
    (*perlin)->size = size;
    
    perlin_generate((*(perlin))->ranvec, size);
    perlin_generate_perm((*(perlin))->permx, size);
    perlin_generate_perm((*(perlin))->permy, size);
    perlin_generate_perm((*(perlin))->permz, size);
}


__global__ void init_random_states(Image *image){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int x = tid % image->width;
    int y = tid / image->width;
    if(x < image->width && y < image->height){
        curand_init(1234, tid, 0, &(image->states[tid]));
    }
}

__host__ __device__ glm::vec3 get_sky(Ray r){
    return glm::vec3(0.0f);
    glm::vec3 dir = glm::normalize(r.direction);
    float t = 0.5f * (dir.y + 1.0f);
    return (1.0f - t) * glm::vec3(1.0f) + t*glm::vec3(0.5f, 0.7f, 1.0f);
}

__device__ glm::vec3 get_color(Ray source, Scene *scene, curandState *state, 
                               int max_bounces)
{
    glm::vec3 pixel(0.0f, 0.0f, 0.0f);
    glm::vec3 mask(1.0f, 1.0f, 1.0f);
    Ray r = source;
    Ray scattered;
    LightEval eval;
    Material *material = 0;
    for(int depth = 0; depth < max_bounces; depth += 1){
        hit_record record;
        /* Watch out for self intersection (0.001f) */
        if(!hit_scene(scene, r, 0.001f, FLT_MAX, 
                      &record, state))
        {
            pixel += mask * get_sky(r);
            break;
        }
        
        material = &scene->material_table[record.mat_handle];
        ray_sample_material(r, scene, material, &record, &eval, state);
        
        bool st = scatter(r, &record, scene, &eval, &scattered, material, state);
        
        pixel += mask * eval.emitted;
        
        mask *= eval.attenuation;
        
        if(!st){ break; }
        
        r = scattered;
    }
    return pixel;
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
        
        int max_bounces = 5;
        
        for(int i = 0; i < samples; i += 1){
            float u1 = 2.0f * curand_uniform(state);
            float u2 = 2.0f * curand_uniform(state);
            float dx = (u1 < 1.0f) ? sqrt(u1) - 1.0f : 1.0f - sqrt(2.0f - u1);
            float dy = (u2 < 1.0f) ? sqrt(u2) - 1.0f : 1.0f - sqrt(2.0f - u2);
            
            float u = ((float)x + dx) / (float)image->width;
            float v = ((float)y + dy) / (float)image->height;
            Ray r = camera_get_ray(camera, u, v, state);
            color += get_color(r, scene, state, max_bounces) / (float)total_samples;
        }
        
        image->pixels[tid] = color;
    }
}

void _render_scene(Scene *scene, Image *image, int samples, int samplesPerBatch){
    size_t threads = 64;
    size_t blocks = (image->pixels_count + threads - 1)/threads;
    
    std::cout << "Generating per pixel RNG seed" << std::endl;
    init_random_states<<<blocks, threads>>>(image);
    cudaSynchronize();
    
    std::cout << "Path tracing..." << std::endl;
    
    int runs = samples / samplesPerBatch;
    for(int i = 0; i < runs; i += 1){
        RenderBatch<<<blocks, threads>>>(image, scene, samplesPerBatch,
                                         samples);
        cudaSynchronize();
        float pct = 100.0f*(float(i + 1)/float(runs));
        std::cout.precision(4);
        std::cout << "\r" << pct << "%    " << std::flush;
    }
    
    std::cout << std::endl;
}

void render_scene(Scene *scene, Image *image, int samples, int samplesPerBatch){
    Timed("Rendering", _render_scene(scene, image, samples, samplesPerBatch));
}

int render_fluid_scene(const char *path){
    Image *image = image_new(600, 400);
    Scene *scene = scene_new();
    Parser_v2 *parser = Parser_v2_new("vs");
    Timed("Reading particles", Parser_v2_load_single_file(parser, path));
    float radius = 0.012f;
    size_t n = 0;
    size_t bo = 0;
    
    glm::vec3 *particles = Parser_v2_get_raw_vector_ptr(parser, 0, 0, &n);
    float *boundary = Parser_v2_get_raw_scalar_ptr(parser, 0, 0, &n);
    
    for(size_t k = 0; k < n; k += 1){
        bo += boundary[k] ? 1 : 0;
    }
    
    std::cout << "Boundary " << bo << std::endl;
    scene->perlin = nullptr;
    perlin_initialize(&scene->perlin, 256);
    
    /* Build texture, all colors only */
    texture_handle tex_part = scene_add_texture_solid(scene, 
                                                      glm::vec3(0.98, 0.1, 0.2));
    texture_handle ground_tex = scene_add_texture_solid(scene,
                                                        glm::vec3(0.68));
    texture_handle glass_tex = scene_add_texture_solid(scene, glm::vec3(1.0f));
    
    /* Build materials */
    material_handle mat_part = scene_add_material_diffuse(scene, tex_part);
    material_handle mat_ground = scene_add_material_diffuse(scene, ground_tex);
    material_handle mat_glass = scene_add_material_dieletric(scene, 
                                                             glass_tex, 1.07f);
    
    //ground is giant sphere
    scene_add_sphere(scene, glm::vec3(0.0f, -1000.5f, -1.0f), 1000.0f, mat_ground);
    
    //add all particles
    for(size_t i = 0; i < n; i += 1){
        scene_add_sphere(scene, particles[i], radius, mat_part);
    }
    
    //container
    //scene_add_sphere(scene, glm::vec3(1.0f), 1.0f+2.0f*radius, mat_glass);
    Timed("Building BVH", scene_build_done(scene));
    
    //set camera stuff
    float aspect = (float)image->width / (float)image->height;
    glm::vec3 origin = glm::vec3(1.0f, 3.0f, 3.5f);
    glm::vec3 target = glm::vec3(1.0f);
    glm::vec3 up = glm::vec3(0.f,1.0f,0.0f);
    
    //in case you want focus, I don't use it
    float focus_dist = glm::length(origin - target);
    (void)focus_dist;
    
    //scene->camera = camera_new(origin, target, up, 45, aspect, 2.0f, focus_dist);
    scene->camera = camera_new(origin, target, up, 45, aspect);
    
    //define samples to run per pixel and per pixel per run
    int samples = 100;
    int samplesPerBatch = 10;
    
    //start path tracer
    render_scene(scene, image, samples, samplesPerBatch);
    
    image_write(image, OUT_FILE);
    image_free(image);
    return 0;
}

int render_cornell(){
    Image *image = image_new(800, 600);
    Scene *scene = scene_new();
    scene->perlin = nullptr;
    perlin_initialize(&scene->perlin, 256);
    texture_handle solidRed = scene_add_texture_solid(scene, glm::vec3(0.65, 0.05, 0.05));
    texture_handle solidGray = scene_add_texture_solid(scene,glm::vec3(0.73, 0.73, 0.73));
    texture_handle solidGreen = scene_add_texture_solid(scene,glm::vec3(0.12, 0.45, 0.15));
    texture_handle solidWhite = scene_add_texture_solid(scene,glm::vec3(10.0f));
    
    texture_handle imageTex = scene_add_texture_image(scene, "forest.png");
    texture_handle gridTex = scene_add_texture_image(scene, "forest_grid.png");
    texture_handle white = scene_add_texture_solid(scene, glm::vec3(1.0f));
    material_handle red = scene_add_material_diffuse(scene, solidRed);
    material_handle green = scene_add_material_diffuse(scene, solidGreen);
    material_handle emit = scene_add_material_emitter(scene, solidWhite);
    material_handle gray = scene_add_material_diffuse(scene, solidGray);
    
    material_handle imageMat = scene_add_material_diffuse(scene, imageTex);
    material_handle gridMat = scene_add_material_diffuse(scene, gridTex);
    material_handle glass = scene_add_material_dieletric(scene, white, 1.07f);
    
    scene_add_rectangle_yz(scene, 0, 555, 0, 555, 555, gridMat, 1);
    scene_add_rectangle_yz(scene, 0, 555, 0, 555, 0, gridMat);
    
    scene_add_rectangle_xz(scene, 213, 343, 227, 332, 554, emit);
    scene_add_rectangle_xz(scene, 0, 555, 0, 555, 555, gray, 1);
    scene_add_rectangle_xz(scene, 0, 555, 0, 555, 0, gray);
    scene_add_rectangle_xy(scene, 0, 555, 0, 555, 555, imageMat, 1);
    
    scene_add_sphere(scene, glm::vec3(277.0, 120.5, 277.0), -70.0f, glass);
    
    Timed("Building BVH", scene_build_done(scene));
    
    float aspect = (float)image->width / (float)image->height;
    glm::vec3 origin = glm::vec3(278, 278, -500);
    glm::vec3 target = glm::vec3(278,278,0);
    glm::vec3 up = glm::vec3(0.f,1.0f,0.0f);
    
    float focus_dist = glm::length(origin - target);
    (void)focus_dist;
    
    //scene->camera = camera_new(origin, target, up, 45, aspect, 2.0f, focus_dist);
    scene->camera = camera_new(origin, target, up, 43, aspect);
    int samples = 1000;
    int samplesPerBatch = 100;
    
    render_scene(scene, image, samples, samplesPerBatch);
    
    image_write(image, OUT_FILE);
    image_free(image);
    return 0;
}

int render_cornell2(){
    Image *image = image_new(800, 600);
    Scene *scene = scene_new();
    scene->perlin = nullptr;
    
    perlin_initialize(&scene->perlin, 256);
    texture_handle solidRed = scene_add_texture_solid(scene, glm::vec3(0.65, 0.05, 0.05));
    texture_handle solidGray = scene_add_texture_solid(scene,glm::vec3(0.73, 0.73, 0.73));
    texture_handle solidGreen = scene_add_texture_solid(scene,glm::vec3(0.12, 0.45, 0.15));
    texture_handle solidWhite = scene_add_texture_solid(scene,glm::vec3(10.0f));
    texture_handle white = scene_add_texture_solid(scene, glm::vec3(1.0f));
    
    material_handle red = scene_add_material_diffuse(scene, solidRed);
    material_handle green = scene_add_material_diffuse(scene, solidGreen);
    material_handle emit = scene_add_material_emitter(scene, solidWhite);
    material_handle gray = scene_add_material_diffuse(scene, solidGray);
    
    texture_handle solidMet = scene_add_texture_solid(scene, 
                                                      glm::vec3(0.8, 0.6, 0.2));
    material_handle metal1 = scene_add_material_metal(scene, solidMet, 1.0f);
    material_handle glass = scene_add_material_dieletric(scene, white, 1.5f);
    
    scene_add_rectangle_yz(scene, 0, 555, 0, 555, 555, green, 1);
    scene_add_rectangle_yz(scene, 0, 555, 0, 555, 0, red);
    scene_add_rectangle_xz(scene, 213, 343, 227, 332, 554, emit, 1);
    scene_add_rectangle_xz(scene, 0, 555, 0, 555, 555, gray, 1);
    scene_add_rectangle_xz(scene, 0, 555, 0, 555, 0, gray);
    scene_add_rectangle_xy(scene, 0, 555, 0, 555, 555, gray, 1);
    
    //scene_add_box(scene, glm::vec3(192.5, 82.5, 147.5), glm::vec3(165,165,165),
    //glm::vec3(0.0f, -18.0f,0.0f), gray);
    
    scene_add_sphere(scene, glm::vec3(190, 90, 190), 90, glass);
    scene_add_box(scene, glm::vec3(357.5, 165.0, 377.5), glm::vec3(165,330,165),
                  glm::vec3(0.0f,15.0f,0.0f), gray);
    
    Timed("Building BVH", scene_build_done(scene));
    
    float aspect = (float)image->width / (float)image->height;
    glm::vec3 origin = glm::vec3(278, 278, -700);
    glm::vec3 target = glm::vec3(278,278,0);
    glm::vec3 up = glm::vec3(0.f,1.0f,0.0f);
    
    float focus_dist = glm::length(origin - target);
    (void)focus_dist;
    
    //scene->camera = camera_new(origin, target, up, 45, aspect, 2.0f, focus_dist);
    scene->camera = camera_new(origin, target, up, 43, aspect);
    int samples = 30000;
    int samplesPerBatch = 100;
    
    render_scene(scene, image, samples, samplesPerBatch);
    
    image_write(image, OUT_FILE);
    image_free(image);
    return 0;
}


int render_scene(){
    Object ball, air;
    Image *image = image_new(1366, 720);
    Scene *scene = scene_new();
    scene->perlin = nullptr;
    perlin_initialize(&scene->perlin, 256);
    
    int nb = 20;
    texture_handle solidWhite = scene_add_texture_solid(scene, glm::vec3(0.73));
    texture_handle solidGreen = scene_add_texture_solid(scene, glm::vec3(0.48, 0.83,
                                                                         0.53));
    texture_handle brightWhite = scene_add_texture_solid(scene, glm::vec3(7.0f));
    texture_handle solidBright = scene_add_texture_solid(scene, glm::vec3(0.8,0.8,0.9));
    
    texture_handle pertext = scene_add_texture_noise(scene, NOISE_TRILINEAR,
                                                     glm::vec3(1.0f));
    
    material_handle ground = scene_add_material_diffuse(scene, solidGreen);
    material_handle brightLight = scene_add_material_emitter(scene, brightWhite);
    material_handle glass = scene_add_material_dieletric(scene, solidWhite, 1.5f);
    material_handle iso = scene_add_material_isotropic(scene, glm::vec3(0.2,0.4,
                                                                        0.9));
    material_handle iso2 = scene_add_material_isotropic(scene, glm::vec3(0.9,0.4,
                                                                         0.1));
    
    
    material_handle white = scene_add_material_diffuse(scene, scene->white_texture);
    material_handle noisemat = scene_add_material_diffuse(scene, pertext);
    material_handle wwh = scene_add_material_diffuse(scene, solidWhite);
    
    
    for(int i = 0; i < nb; i += 1){
        for(int j = 0; j < nb; j += 1){
            float w = 100.0f;
            float x0 = -1000.0f + i * w;
            float z0 = -1000.0f + j * w;
            float y0 = 0.0f;
            
            float x1 = x0 + w;
            float y1 = 100.0f * (random_float() + 0.01f);
            float z1 = z0 + w;
            
            glm::vec3 p = glm::vec3((x0+x1)/2.0f, (y0+y1)/2.0f, (z0+z1)/2.0f); //pos
            glm::vec3 s = glm::vec3((x1-x0), (y0-y1), (z1-z0)); //scale
            glm::vec3 r = glm::vec3(0.0f); //rotation
            scene_add_box(scene, p, s, r, ground);
        }
    }
    
    
    scene_add_rectangle_xz(scene, 123, 423, 147, 412, 554, brightLight);
    scene_add_sphere(scene, glm::vec3(160, 170, 80), 70, glass);
    
    scene_add_sphere(scene, glm::vec3(360, 150, 145), 70, glass);
    ball = scene_add_sphere(scene, glm::vec3(360, 150, 145), 70, glass);
    scene_add_medium(scene, ball, 0.2f, iso);
    
    air = scene_add_sphere(scene, glm::vec3(0.0f), 5000.0f, glass);
    scene_add_medium(scene, air, 0.0001f, white);
    
    
    Object data;
    glm::vec3 boxp = glm::vec3(-70, 170, 175);
    glm::vec3 boxsize = glm::vec3(100.0f) * 0.5f ;
    
    scene_add_sphere(scene, boxp, boxsize.x, glass);
    data = scene_add_sphere(scene, boxp, boxsize.x, glass);
    //scene_add_box(scene, boxp, boxsize, glm::vec3(0.0f), glass);
    //data = scene_add_box(scene, boxp, boxsize, glm::vec3(0.0f), glass);
    
    scene_add_medium(scene, data, 0.2f, iso2);
    
    scene_add_sphere(scene, glm::vec3(250,280,300), 80, noisemat);
    
    int ns = 1000;
    for(int j = 0; j < ns; j += 1){
        float x = 165.0f * random_float();
        float y = 165.0f * random_float();
        float z = 165.0f * random_float();
        glm::vec3 c(x, y, z);
        scene_add_sphere(scene, c + glm::vec3(-100.0f, 270.0f, 395.0f), 10.0f, wwh);
    }
    
    Timed("Building BVH", scene_build_done(scene));
    
    float aspect = (float)image->width / (float)image->height;
    glm::vec3 origin = glm::vec3(478, 278, -600);
    glm::vec3 target = glm::vec3(278,278,0);
    glm::vec3 up = glm::vec3(0.f,1.0f,0.0f);
    
    scene->camera = camera_new(origin, target, up, 40, aspect);
    int samples = 30000;
    int samplesPerBatch = 100;
    
    render_scene(scene, image, samples, samplesPerBatch);
    
    image_write(image, OUT_FILE);
    image_free(image);
    return 0;
}


int main(int argc, char **argv){
    (void)cudaInit();
    
    //return render_scene();
    return render_cornell2();
    //return render_fluid_scene("/home/felpz/OUT_PART_SimplexSphere2_60.txt");
    //return render_fluid_scene("/home/felpz/OUT_PART_3DRun_10.txt");
    
    
    Image *image = image_new(800, 600);
    Scene *scene = scene_new();
    scene->perlin = nullptr;
    perlin_initialize(&scene->perlin, 256);
    
    texture_handle imageTexture = scene_add_texture_image(scene, "earthmap.jpg");
    
    texture_handle noiseTexture = scene_add_texture_noise(scene, NOISE_TRILINEAR,
                                                          glm::vec3(1.0f));
    
    texture_handle solidGreen = scene_add_texture_solid(scene,
                                                        glm::vec3(0.1,0.6,0.1));
    texture_handle solidRed = scene_add_texture_solid(scene, glm::vec3(0.9,0.1,0.1));
    
    texture_handle solidBlue = scene_add_texture_solid(scene, 
                                                       glm::vec3(0.1, 0.2, 0.7));
    
    texture_handle solidYellow = scene_add_texture_solid(scene,
                                                         glm::vec3(0.8, 0.8, 0.0));
    
    texture_handle solidMet = scene_add_texture_solid(scene, 
                                                      glm::vec3(0.8, 0.6, 0.2));
    
    texture_handle solidMet2 = scene_add_texture_solid(scene, 
                                                       glm::vec3(0.8, 0.8, 0.8));
    
    texture_handle solidWhite = scene_add_texture_solid(scene, glm::vec3(1.0f));
    
    texture_handle checker = scene_add_texture_checker(scene, 
                                                       solidWhite, solidGreen);
    
    material_handle diffuse1 = scene_add_material_diffuse(scene, solidBlue);
    
    material_handle diffuse2 = scene_add_material_diffuse(scene, checker);
    //material_handle diffuse2 = scene_add_material_diffuse(scene, noiseTexture);
    
    material_handle imageMat = scene_add_material_diffuse(scene, imageTexture);
    
    material_handle metal1 = scene_add_material_metal(scene, solidMet, 1.0f);
    
    material_handle metal2 = scene_add_material_metal(scene, solidMet2, 0.3f);
    
    material_handle glass = scene_add_material_dieletric(scene, solidWhite, 1.07f);
    
    material_handle emit = scene_add_material_emitter(scene, solidGreen);
    material_handle emit2 = scene_add_material_emitter(scene, solidRed);
    material_handle emit3 = scene_add_material_emitter(scene, solidBlue);
    material_handle emit4 = scene_add_material_emitter(scene, scene->white_texture);
    
    scene_add_sphere(scene, glm::vec3(0.0f, 0.0f, -1.0f), 0.5f, diffuse1);
    scene_add_sphere(scene, glm::vec3(0.0f, -1000.5f, -1.0f), 1000.0f, diffuse2);
    
    scene_add_sphere(scene, glm::vec3(1.0f+0.001f,0.0f,-1.0f), -0.45f, glass);
    scene_add_sphere(scene, glm::vec3(-1.0f-0.001f,0.0f,-1.0f), 0.5f, imageMat);
    
    scene_add_rectangle_yz(scene, -0.2f, 1.5f, -2.2f, 1.2f, -2.1f, emit);
    scene_add_rectangle_yz(scene, -0.2f, 1.5f, -2.2f, 1.2f, 2.1f, emit2);
    scene_add_rectangle_xy(scene, -1.7f, 1.7f, -0.2f, 1.5f, 1.2f, emit4);
    //scene_add_sphere(scene, glm::vec3(0.0f,2.3f,-1.0f), 0.35f, emit3);
    
    Timed("Building BVH", scene_build_done(scene));
    
    float aspect = (float)image->width / (float)image->height;
    glm::vec3 origin = glm::vec3(0.f,3.f,-7.0f);
    glm::vec3 target = glm::vec3(0.0f,0.0f,-1.0f);
    glm::vec3 up = glm::vec3(0.f,1.0f,0.0f);
    
    float focus_dist = glm::length(origin - target);
    (void)focus_dist;
    
    //scene->camera = camera_new(origin, target, up, 45, aspect, 2.0f, focus_dist);
    scene->camera = camera_new(origin, target, up, 45, aspect);
    int samples = 1000;
    int samplesPerBatch = 100;
    
    render_scene(scene, image, samples, samplesPerBatch);
    
    image_write(image, OUT_FILE);
    image_free(image);
    return 0;
}
