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

__device__ void MakeScene(Aggregator *scene, curandState *state,
                          ParsedMesh **pMeshes, int nMeshes)
{
    Texture ZeroTex(Spectrum(0.f));
    Sphere *sphere = new Sphere(Translate(vec3f(0.f,0.f,-1.f)), 0.5);
    
    Material *matRed = new Material();
    Texture kdGreen(Spectrum(0.2f, 0.7f, 0.3f));
    Texture kdRed(Spectrum(0.8f, 0.2f, 0.3f));
    //matRed->Init_Matte(kdGreen, ZeroTex);
    matRed->Init_Metal(Spectrum(0.7, 0.8, 1.0), 0.5f, 1.0f, 0.5f);
    
    GeometricPrimitive *geo0 = new GeometricPrimitive(sphere, matRed);
    
    int indices[] = {0, 1, 2, 2, 3, 0};
    //int indices[] = {0, 1, 2};
    Point3f p[4];
    //p[0] = Point3f(-1,0,0); p[1] = Point3f(1,0,0);
    //p[2] = Point3f(0,1,0);
    p[0] = Point3f(-1000, -0.1, -100); p[1] = Point3f(1000, -0.1, -100);
    p[2] = Point3f(1000, -0.1, 100); p[3] = Point3f(-1000, -0.1, 100);
    
    Mesh *ground = scene->AddMesh(Translate(vec3f(10,0,-10)), 
                                  2, indices, 4, p, 0, 0, 0);
    
    Material *matYellow = new Material();
    Texture kdYellow(Spectrum(0.7f, 0.7f, 0.0f));
    matYellow->Init_Matte(kdYellow, ZeroTex);
    
    GeometricPrimitive *geo1 = new GeometricPrimitive(ground, matYellow);
    
    Sphere *sphere3 = new Sphere(Translate(vec3f(9.5f,5.f,-1.f)), 5);
    Material *matBlue = new Material();
    Texture kdBlue(Spectrum(0.15f,0.34f,0.9f));
    Texture sigma(Spectrum(30.f));
    matBlue->Init_Plastic(Spectrum(.1f, .1f, .4f), Spectrum(0.6f), 0.03);
    
    
    GeometricPrimitive *geo2 = new GeometricPrimitive(sphere3, matRed);
    
    Sphere *sphere4 = new Sphere(Translate(vec3f(-9.f, 5.f, -1.f)), 5);
    Material *matGlass = new Material();
    //Spectrum glassSpec(0.7f,0.96f,0.75f);
    Spectrum glassSpec(1.f);
    matGlass->Init_Glass(glassSpec, glassSpec, 2.f);
    
    GeometricPrimitive *geo3 = new GeometricPrimitive(sphere4, matGlass);
    
    Material *matGlass2 = new Material();
    Spectrum R(0.31, 0.64, 0.32);
    //Spectrum R(1.f);
    Spectrum T(1.f);
    //matGlass2->Init_Glass(R, T, .05, .05, 1.5f);
    //matGlass2->Init_Matte(Spectrum(.7), 40);
    matGlass2->Init_Uber(Spectrum(.05), Spectrum(.8), Spectrum(0), 
                         Spectrum(0), 0.001, 0.001, 1.f, 2.5f);
    
    scene->Reserve(4+nMeshes);
    for(int i = 0; i < nMeshes; i++){
        Mesh *mptr = scene->AddMesh(pMeshes[i]->toWorld, pMeshes[i], 0);
        GeometricPrimitive *mGeo = new GeometricPrimitive(mptr, matGlass2);
        scene->Insert(mGeo);
    }
    
    
    //scene->Insert(geo0);
    scene->Insert(geo3);
    scene->Insert(geo2);
    
    scene->Insert(geo1);
}

__global__ void BuildScene(Aggregator *scene, Image *image, 
                           ParsedMesh **pMeshes, int nMeshes)
{
    if(threadIdx.x == 0 && blockIdx.x == 0){
        curandState *state = &image->pixels[0].state;
        MakeScene(scene, state, pMeshes, nMeshes);
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
            
            Float pdf = 0.f;
            Point2f u(rand_float(&pixel->state), rand_float(&pixel->state));
            vec3f wi, wo = -ray.d;
            
            isect.ComputeScatteringFunctions(&bsdf, ray, TransportMode::Radiance, true);
            
            Spectrum f = bsdf.Sample_f(wo, &wi, u, &pdf, BSDF_ALL);
            if(IsZero(pdf)) break;
            
            curr = curr * f * AbsDot(wi, ToVec3(isect.n)) / pdf;
            ray.d = Normalize(wi);
            ray.o = isect.p + 3.0 * ShadowEpsilon * ray.d;
            
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
        
        Camera camera(Point3f(0.f, 20.f, -40.f), Point3f(0.0f,10.f,0.f), 
                      vec3f(0.f,1.f,0.f), 30.f, aspect);
        
        //Camera camera(Point3f(13,2,3), Point3f(0.0f,0.f,-1.f), 
        //vec3f(0.f,1.f,0.f), 20.f, aspect, 0.3);
        
        int pixel_index = j * width + i;
        Pixel *pixel = &image->pixels[pixel_index];
        curandState state = pixel->state;
        
        Spectrum out = image->pixels[pixel_index].we;
        for(int n = 0; n < ns; n++){
            Float u = ((Float)i + rand_float(&state)) / (Float)width;
            Float v = ((Float)j + rand_float(&state)) / (Float)height;
            
            Point2f sample = ConcentricSampleDisk(rand_point2(&state));
            
            Ray ray = camera.SpawnRay(u, v, sample);
            //vec3f d = Normalize(Point3f(-0.01, 0.1, 0) - Point3f(0,0,5));
            //Ray ray(Point3f(0,0,5), d);
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
    int it = 1000;
    unsigned long long seed = time(0);
    int nMeshes = 1;
    int mIt = 0;
    
    ParsedMesh **pMeshes = nullptr;
    if(nMeshes > 0){
        pMeshes = cudaAllocateVx(ParsedMesh*, nMeshes);
    }
    
    LoadObjData("/home/felpz/Documents/budda.obj", &pMeshes[mIt]);
    pMeshes[mIt]->toWorld = Scale(20,20,20); mIt++;
    //LoadObjData("/home/felpz/Documents/dragon_aligned.obj", &pMeshes[mIt++]);
    
    dim3 blocks(nx/tx+1, ny/ty+1);
    dim3 threads(tx,ty);
    
    Aggregator *scene = cudaAllocateVx(Aggregator, 1);
    scene->ReserveMeshes(1 + nMeshes);
    
    std::cout << "Initializing image..." << std::flush;
    SetupPixels<<<blocks, threads>>>(image, seed);
    cudaDeviceAssert();
    std::cout << "OK" << std::endl;
    
    std::cout << "Building scene..." << std::flush;
    BuildScene<<<1, 1>>>(scene, image, pMeshes, nMeshes);
    cudaDeviceAssert();
    std::cout << "OK" << std::endl;
    
    scene->Wrap();
    
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
    const int image_width = 1366;
    const int image_height = (int)((Float)image_width / aspect_ratio);
    
    Image *image = CreateImage(image_width, image_height);
    
    launch_render_kernel(image);
    
    ImageWrite(image, "output.png", 1.f, ToneMapAlgorithm::Exponential);
    ImageFree(image);
    cudaDeviceReset();
    return 0;
}