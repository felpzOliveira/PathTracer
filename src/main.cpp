#include <iostream>
#include <geometry.h>
#include <transform.h>
#include <camera.h>
#include <shape.h>
#include <primitive.h>
#include <graphy.h>
#include <reflection.h>
#include <material.h>
#include <util.h>

__device__ Float rand_float(curandState *state){
    return curand_uniform(state);
}

__device__ vec3f rand_vec(curandState *state){
    return vec3f(rand_float(state),
                 rand_float(state),
                 rand_float(state));
}

__device__ Point2f rand_point2(curandState *state){
    return Point2f(rand_float(state), rand_float(state));
}

__bidevice__ void MakeSceneTable(Aggregator *scene, int id){
    if(id < scene->head){
        Primitive *pri = scene->primitives[id];
        scene->handles[id].bound = pri->worldBound;
        scene->handles[id].handle = id;
    }
}

__global__ void BuildSceneTable(Aggregator *scene){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(tid < scene->head){
        MakeSceneTable(scene, tid);
    }
}

__device__ void MakeScene(Aggregator *scene, curandState *state){
#if 0
    Texture ZeroTex(Spectrum(0.f));
    Sphere *sphere = new Sphere(Translate(vec3f(0.f,0.f,-1.f)), 0.5);
    
    Material *matRed = new Material();
    Texture kdRed(Spectrum(0.7f, 0.3f, 0.3f));
    matRed->Init_Matte(kdRed, ZeroTex);
    
    GeometricPrimitive *geo0 = new GeometricPrimitive(sphere, matRed);
    
    Sphere *sphere2 = new Sphere(Translate(vec3f(0.f,-100.5f,-1.f)), 100);
    Material *matYellow = new Material();
    Texture kdYellow(Spectrum(0.7f, 0.7f, 0.0f));
    matYellow->Init_Matte(kdYellow, ZeroTex);
    
    GeometricPrimitive *geo1 = new GeometricPrimitive(sphere2, matYellow);
    
    Sphere *sphere3 = new Sphere(Translate(vec3f(1.f,0.f,-1.f)), 0.5);
    Material *matBlue = new Material();
    Texture kdBlue(Spectrum(0.15f,0.34f,0.9f));
    Texture sigma(Spectrum(30.f));
    //matBlue->Init_Mirror(kdBlue);
    matBlue->Init_Matte(kdRed, sigma);
    
    GeometricPrimitive *geo2 = new GeometricPrimitive(sphere3, matBlue);
    
    Sphere *sphere4 = new Sphere(Translate(vec3f(-1.f, 0.f, -1.f)), 0.5);
    Material *matGlass = new Material();
    matGlass->Init_Glass(Spectrum(0.9f), Spectrum(0.9f), 0, 0, 2.f);
    
    GeometricPrimitive *geo3 = new GeometricPrimitive(sphere4, matGlass);
    
    scene->Reserve(4);
    scene->Insert(geo0);
    scene->Insert(geo1);
    scene->Insert(geo2);
    scene->Insert(geo3);
#else
    int count = 22 * 22 + 10;
    scene->Reserve(count);
    Sphere *spherePtr = nullptr;
    Material *material = nullptr;
    GeometricPrimitive *gPri = nullptr;
    
    spherePtr = new Sphere(Translate(0,1,0),1);
    material = new Material();
    material->Init_Glass(Spectrum(0.9f), Spectrum(0.9f), 0, 0, 2.f);
    gPri = new GeometricPrimitive(spherePtr, material);
    scene->Insert(gPri);
    
    spherePtr = new Sphere(Translate(0,-1000,0),1000);
    material = new Material();
    material->Init_Matte(Texture(Spectrum(0.5)), Texture(Spectrum(30.f)));
    gPri = new GeometricPrimitive(spherePtr, material);
    scene->Insert(gPri);
    
#if 1
    
    spherePtr = new Sphere(Translate(-4,1,0),1);
    material = new Material();
    material->Init_Matte(Texture(Spectrum(.4,.2,.1)), Texture(Spectrum(0.f)));
    gPri = new GeometricPrimitive(spherePtr, material);
    scene->Insert(gPri);
    
    spherePtr = new Sphere(Translate(4,1,0),1);
    material = new Material();
    material->Init_Mirror(Texture(Spectrum(.7,.6,.5)));
    gPri = new GeometricPrimitive(spherePtr, material);
    scene->Insert(gPri);
    
    for(int a = -11; a < 11; a++){
        for(int b = -11; b < 11; b++){
            vec3f center(a + 0.9*rand_float(state), 0.2, b + 0.9*rand_float(state));
            if((center - vec3f(4, 0.2, 0)).Length() > 0.9){
                Float mat_rng = rand_float(state);
                material = new Material();
                if(mat_rng < 0.8){
                    Spectrum R = rand_vec(state);
                    Float ang = rand_float(state) * 60.f;
                    material->Init_Matte(Texture(R), Texture(Spectrum(ang)));
                }else if(mat_rng < 0.95){
                    Spectrum R = rand_vec(state) * 0.5 + vec3f(0.5);
                    material->Init_Mirror(Texture(R));
                }else{
                    material->Init_Glass(Spectrum(0.9f), Spectrum(0.9f), 0, 0, 3.5f);
                }
                
                spherePtr = new Sphere(Translate(center), 0.2);
                gPri = new GeometricPrimitive(spherePtr, material);
                scene->Insert(gPri);
            }
        }
    }
#endif
#endif
}

__global__ void BuildScene(Aggregator *scene, Image *image){
    if (threadIdx.x == 0 && blockIdx.x == 0){
        curandState *state = &image->pixels[0].state;
        MakeScene(scene, state);
    }
}

__global__ void SetupPixels(Image *image, unsigned long long seed){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    int width = image->width;
    int height = image->height;
    
    if(i < width && j < height){
        int pixel_index = j * width + i;
        image->pixels[pixel_index].we = Spectrum(0.f);
        image->pixels[pixel_index].misses = 0;
        image->pixels[pixel_index].hits = 0;
        image->pixels[pixel_index].samples = 0;
        image->pixels[pixel_index].max_transverse_tests = 0;
        curand_init(seed, pixel_index, 0, &image->pixels[pixel_index].state);
    }
}

__global__ void ReleaseScene(Aggregator *scene){
    if (threadIdx.x == 0 && blockIdx.x == 0){
        scene->Release();
    }
}

__bidevice__ Spectrum GetSky(vec3f dir){
    vec3f unit = Normalize(dir);
    Float t = 0.5*(dir.y + 1.0);
    return (1.0-t)*Spectrum(1.0, 1.0, 1.0) + t*Spectrum(0.5, 0.7, 1.0);
}

__device__ Spectrum Li(Ray ray, Aggregator *scene, Pixel *pixel){
    Spectrum out(0.f);
    Spectrum curr(1.f);
    
    int maxi = 50;
    int i = 0;
    for(i = 0; i < maxi; i++){
        SurfaceInteraction isect;
        if(scene->Intersect(ray, &isect, pixel)){
            BSDF bsdf(isect);
            
            Float pdf = 1.f;
            Point2f u(rand_float(&pixel->state), rand_float(&pixel->state));
            vec3f wi, wo = -ray.d;
            
            isect.ComputeScatteringFunctions(&bsdf, ray, TransportMode::Radiance, true);
            
            Spectrum f = bsdf.Sample_f(wo, &wi, u, &pdf);
            
            if(IsZero(pdf)) break;
            
            curr = curr * f * AbsDot(wi, ToVec3(isect.n)) / pdf;
            ray.d = Normalize(wi);
            ray.o = isect.p + 3. * ShadowEpsilon * ray.d;
            
            pixel->hits += 1;
        }else{
            out = curr * GetSky(ray.d);
            pixel->misses += 1;
            break;
        }
    }
    
    if(i == maxi-1) return Spectrum(0.f);
    return out;
}

__global__ void Render(Image *image, Aggregator *scene, int ns){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    int width = image->width;
    int height = image->height;
    
    if(i < width && j < height){
        Float aspect = ((Float)width)/((Float)height);
        
        //Camera camera(Point3f(0.f,0.5f, -4.f), Point3f(0.0f,0.f,-1.f), 
        //vec3f(0.f,1.f,0.f), 33.f, aspect);
        
        Camera camera(Point3f(13,2,3), Point3f(0.0f,0.f,-1.f), 
                      vec3f(0.f,1.f,0.f), 20.f, aspect, 0.3);
        
        int pixel_index = j * width + i;
        Pixel *pixel = &image->pixels[pixel_index];
        curandState state = pixel->state;
        
        Spectrum out = image->pixels[pixel_index].we;
        for(int n = 0; n < ns; n++){
            Float u = ((Float)i + rand_float(&state)) / (Float)width;
            Float v = ((Float)j + rand_float(&state)) / (Float)height;
            
            Point2f sample = ConcentricSampleDisk(rand_point2(&state));
            
            Ray ray = camera.SpawnRay(u, v, sample);
            out += Li(ray, scene, pixel);
            pixel->samples ++;
        }
        
        image->pixels[pixel_index].we = out;
    }
}

void launch_render_kernel(Image *image){
    int tx = 8;
    int ty = 8;
    int nx = image->width;
    int ny = image->height;
    int it = 100;
    unsigned long long seed = time(0);
    
    dim3 blocks(nx/tx+1, ny/ty+1);
    dim3 threads(tx,ty);
    
    Aggregator *scene = cudaAllocateVx(Aggregator, 1);
    std::cout << "Initializing image..." << std::flush;
    SetupPixels<<<blocks, threads>>>(image, seed);
    cudaDeviceAssert();
    std::cout << "OK" << std::endl;
    
    std::cout << "Building scene..." << std::flush;
    BuildScene<<<1, 1>>>(scene, image);
    cudaDeviceAssert();
    
    size_t pThreads = 64;
    size_t pBlocks = (scene->head + pThreads - 1)/pThreads;
    scene->handles = cudaAllocateVx(PrimitiveHandle, scene->head);
    BuildSceneTable<<<pBlocks,pThreads>>>(scene);
    cudaDeviceAssert();
    
    std::cout << "OK" << std::endl;
    
    std::cout << "Packing scene..." << std::flush;
    scene->Wrap();
    std::cout << "OK [ Build BVH with " << scene->totalNodes << " nodes ]" << std::endl;
    
    std::cout << "Rendering..." << std::endl;
    for(int i = 0; i < it; i++){
        Render<<<blocks, threads>>>(image, scene, 1);
        cudaDeviceAssert();
        graphy_display_pixels(image);
        //if(i == 0) getchar();
        std::cout << "\rIteration: " << i << std::flush;
    }
    
    std::cout << std::endl;
    
    ReleaseScene<<<1, 1>>>(scene); //guess we can contiue even if this crashes
    if(cudaSynchronize()){ printf("Failed to free scene\n"); }
    
    cudaFree(scene->handles);//TODO: This needs to transverse the tree
    cudaFree(scene);
}

int main(int argc, char **argv){
    cudaInitEx();
    Float aspect_ratio = 16.0 / 9.0;
    const int image_width = 800;
    const int image_height = (int)((Float)image_width / aspect_ratio);
    
    Image *image = CreateImage(image_width, image_height);
    
    launch_render_kernel(image);
    
    ImageWrite(image, "output.png", 1.f, ToneMapAlgorithm::Exponential);
    ImageFree(image);
    cudaDeviceReset();
    return 0;
}