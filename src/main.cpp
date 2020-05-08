#include <iostream>
#include <geometry.h>
#include <transform.h>
#include <camera.h>
#include <shape.h>
#include <primitive.h>
#include <graphy.h>
#include <reflection.h>

__device__ Float rand_float(curandState *state){
    return curand_uniform(state);
}

__bidevice__ AggregateList *MakeScene(){
    AggregateList *scene  = new AggregateList(2);
    Sphere *sphere = new Sphere(Translate(vec3f(0.f,0.f,-1.f)), 0.5);
    GeometricPrimitive *geo0 = new GeometricPrimitive(sphere);
    
    Sphere *sphere2 = new Sphere(Translate(vec3f(0.f,-100.5f,-1.f)), 100);
    GeometricPrimitive *geo1 = new GeometricPrimitive(sphere2);
    
    scene->Insert(geo1);
    scene->Insert(geo0);
    return scene;
}

__global__ void BuildScene(AggregateList **scene){
    if (threadIdx.x == 0 && blockIdx.x == 0){
        *scene = MakeScene();
    }
}

__global__ void SetupPixels(Image *image){
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
        curand_init(1984, pixel_index, 0, &image->pixels[pixel_index].state);
    }
}

__global__ void ReleaseScene(AggregateList **scene){
    if (threadIdx.x == 0 && blockIdx.x == 0){
        (*scene)->Release();
        delete *scene;
    }
}

__bidevice__ vec3f random_in_unit_sphere(Float u[2]){
    Float a = u[0]  * 2 * Pi;
    Float z = u[1]  * 2  - 1;
    Float r = sqrt(1 - z*z);
    return vec3f(r * std::cos(a), r * std::sin(a), z);
}

__bidevice__ Spectrum GetSky(vec3f dir){
    vec3f unit = Normalize(dir);
    Float t = 0.5*(dir.y + 1.0);
    return (1.0-t)*Spectrum(1.0, 1.0, 1.0) + t*Spectrum(0.5, 0.7, 1.0);
}

__device__ Spectrum Li(Ray ray, AggregateList **scene, Pixel *pixel){
    Spectrum out(0.f);
    Spectrum curr(1.f);
    
    int maxi = 50;
    int i = 0;
    for(i = 0; i < maxi; i++){
        SurfaceInteraction isect;
        if((*scene)->Intersect(ray, &isect)){
#if 0
            Float u[2] = {rand_float(&pixel->state), rand_float(&pixel->state)};
            vec3f target = ToVec3(isect.p) + ToVec3(isect.n) + random_in_unit_sphere(u);
            curr *= 0.5f;
            ray.o = isect.p;
            ray.d = Normalize(target - vec3f(isect.p));
#else
            Float pdf = 1.f;
            Point2f u(rand_float(&pixel->state), rand_float(&pixel->state));
            vec3f wi;
            vec3f wo = -ray.d;
            LambertianReflection ref(Spectrum(0.5f), isect);
            
            Spectrum f = ref.Sample_f(wo, &wi, u, &pdf);
            if(IsZero(pdf)) break;
            
            curr = curr * f * AbsDot(wi, ToVec3(isect.n)) / pdf;
            ray.o = isect.p;
            ray.d = Normalize(wi);
#endif
            pixel->hits += 1;
        }else{
            out = curr * GetSky(ray.d);
            break;
        }
    }
    
    if(i == maxi-1) return Spectrum(0.f);
    return out;
}

__global__ void Render(Image *image, AggregateList **scene, int ns){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    int width = image->width;
    int height = image->height;
    
    if(i < width && j < height){
        Float aspect = ((Float)width)/((Float)height);
        Camera camera(Point3f(0.f,0.5f, -5.f), Point3f(0.0f,0.f,-1.f), 
                      vec3f(0.f,1.f,0.f), 33.f, aspect);
        
        int pixel_index = j * width + i;
        Pixel *pixel = &image->pixels[pixel_index];
        curandState state = pixel->state;
        
        Spectrum out = image->pixels[pixel_index].we;
        for(int n = 0; n < ns; n++){
            Float u = ((Float)i + rand_float(&state)) / (Float)width;
            Float v = ((Float)j + rand_float(&state)) / (Float)height;
            
            Ray ray = camera.SpawnRay(u, v);
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
    
    dim3 blocks(nx/tx+1, ny/ty+1);
    dim3 threads(tx,ty);
    
    AggregateList **scene = cudaAllocateVx(AggregateList*, 1);
    
    BuildScene<<<1, 1>>>(scene);
    cudaDeviceAssert();
    
    SetupPixels<<<blocks, threads>>>(image);
    cudaDeviceAssert();
    for(int i = 0; i < it; i++){
        Render<<<blocks, threads>>>(image, scene, 10);
        cudaDeviceAssert();
        graphy_display_pixels(image);
        std::cout << "\rIteration: " << i << std::flush;
    }
    
    std::cout << std::endl;
    
    ReleaseScene<<<1, 1>>>(scene); //guess we can contiue even if this crashes
    if(cudaSynchronize()){ printf("Failed to free scene\n"); }
    
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