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
#include <scene.h>
#include <ppm.h>
#include <light.h>

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
    return Spectrum(0);
    vec3f unit = Normalize(dir);
    Float t = 0.5*(dir.y + 1.0);
    return ((1.0-t)*Spectrum(1.0, 1.0, 1.0) + t*Spectrum(0.5, 0.7, 1.0))*1;
}

__device__ Spectrum DirectLi(Ray ray, Aggregator *scene, Pixel *pixel){
    Spectrum L(0.f);
    SurfaceInteraction isect;
    
    if(!scene->Intersect(ray, &isect, pixel)) {
        for(int i = 0; i < scene->lightCounter; i++){
            DiffuseAreaLight *light = scene->lights[i];
            L += light->Le(ray);
        }
    }else{
        BSDF bsdf(isect);
        isect.ComputeScatteringFunctions(&bsdf, ray, TransportMode::Radiance, true);
        vec3f wo = isect.wo;
        L += isect.Le(wo);
        
        if(scene->lightCounter > 0){
            Point2f u2(rand_float(&pixel->state), rand_float(&pixel->state));
            Point3f u3(rand_float(&pixel->state), rand_float(&pixel->state), 
                       rand_float(&pixel->state));
            
            L += scene->UniformSampleOneLight(isect, &bsdf, u2, u3);
        }
    }
    
    return L;
}

__device__ Spectrum SampledLi(Ray r, Aggregator *scene, Pixel *pixel){
    Spectrum L(0.f), beta(1.f);
    RayDifferential ray(r);
    bool specularBounce = false;
    int bounces;
    int max_bounces = 5;
    Float rrThreshold = 1;
    
    for(bounces = 0; ; bounces++){
        SurfaceInteraction isect;
        bool foundIntersection = scene->Intersect(ray, &isect, pixel);
        if(bounces == 0 || specularBounce){
            if(foundIntersection){
                L += beta * isect.Le(-ray.d);
            }else{
                for(int i = 0; i < scene->lightCounter; i++){
                    DiffuseAreaLight *light = scene->lights[i];
                    L += light->Le(ray);
                }
            }
        }
        
        if(!foundIntersection || bounces >= max_bounces) break;
        BSDF bsdf(isect);
        
        isect.ComputeScatteringFunctions(&bsdf, ray, TransportMode::Radiance, true);
        if(bsdf.nBxDFs == 0){
            ray = isect.SpawnRay(ray.d);
            bounces--;
            continue;
        }
        
        if(bsdf.NumComponents(BxDFType(BSDF_ALL & ~BSDF_SPECULAR)) > 0){
            Point2f u2(rand_float(&pixel->state), rand_float(&pixel->state));
            Point3f u3(rand_float(&pixel->state), rand_float(&pixel->state), 
                       rand_float(&pixel->state));
            Spectrum Ld = beta * scene->UniformSampleOneLight(isect, &bsdf, u2, u3);
            L += Ld;
        }
        
        Float pdf = 0.f;
        Point2f u(rand_float(&pixel->state), rand_float(&pixel->state));
        vec3f wi, wo = -ray.d;
        BxDFType flags;
        
        Spectrum f = bsdf.Sample_f(wo, &wi, u, &pdf, BSDF_ALL, &flags);
        
        if(f.IsBlack() || IsZero(pdf)) break;
        
        beta *= f * AbsDot(wi, ToVec3(isect.n)) / pdf;
        specularBounce = (flags & BSDF_SPECULAR) != 0;
        
        ray = isect.SpawnRay(wi);
        
        Spectrum rrBeta = beta;
        if(MaxComponent(rrBeta) < rrThreshold && bounces > 3) {
            Float q = Max((Float).05, 1 - MaxComponent(rrBeta));
            Float u = rand_float(&pixel->state);
            if (u < q) break;
            beta = beta / (1 - q);
        }
    }
    
    return L;
}

__device__ Spectrum Li(Ray ray, Aggregator *scene, Pixel *pixel){
    Spectrum out(0.f);
    Spectrum curr(1.f);
    
    int maxi = 10;
    int i = 0;
    for(i = 0; i < maxi; i++){
        SurfaceInteraction isect;
        
        if(scene->Intersect(ray, &isect, pixel)){
            BSDF bsdf(isect);
            
            Float pdf = 0.f;
            Point2f u(rand_float(&pixel->state), rand_float(&pixel->state));
            vec3f wi, wo = -ray.d;
            
            isect.ComputeScatteringFunctions(&bsdf, ray, TransportMode::Radiance, true);
            Spectrum L = isect.primitive->Le();
            out += curr * L;
            
            Spectrum f = bsdf.Sample_f(wo, &wi, u, &pdf, BSDF_ALL);
            if(IsZero(pdf)) break;
            
            curr = curr * f * AbsDot(wi, ToVec3(isect.n)) / pdf;
            ray = isect.SpawnRay(wi);
            pixel->hits += 1;
        }else{
            out += curr * GetSky(ray.d);
            pixel->misses += 1;
            break;
        }
    }
    
    if(i == maxi-1) return Spectrum(0.f);
    return out;
}

__global__ void Render(Image *image, Aggregator *scene, Camera *camera, int ns){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    int width = image->width;
    int height = image->height;
    
    if(i < width && j < height){
        int pixel_index = j * width + i;
        Pixel *pixel = &image->pixels[pixel_index];
        curandState state = pixel->state;
        
        Spectrum out = image->pixels[pixel_index].we;
        for(int n = 0; n < ns; n++){
            Float u = ((Float)i + rand_float(&state)) / (Float)width;
            Float v = ((Float)j + rand_float(&state)) / (Float)height;
            
            Point2f sample = ConcentricSampleDisk(rand_point2(&state));
            
            Ray ray = camera->SpawnRay(u, v, sample);
            //out += Li(ray, scene, pixel);
            //out += DirectLi(ray, scene, pixel);
            out += SampledLi(ray, scene, pixel);
            pixel->samples ++;
        }
        
        image->pixels[pixel_index].we = out;
    }
}

void DragonScene(Camera *camera, Float aspect){
    AssertA(!!camera, "Invalid camera pointer");
    
    camera->Config(Point3f(53.f, 48.f, -43.f), 
                   Point3f(-10.12833, 5.15341, 5.16229),
                   vec3f(0.f,1.f,0.f), 40.f, aspect);
    
    MaterialDescriptor gray = MakePlasticMaterial(Spectrum(.1,.1, .1), 
                                                  Spectrum(.7, .7, .7), 0.1);
    MaterialDescriptor yellow = MakeMatteMaterial(Spectrum(.7,.7,.0));
    MaterialDescriptor white  = MakeMatteMaterial(Spectrum(0.87, 0.87, 0.87));
    MaterialDescriptor orange = MakeMatteMaterial(Spectrum(0.98, 0.56, 0.0));
    MaterialDescriptor red    = MakeMatteMaterial(Spectrum(0.86, 0.2, 0.1));
    MaterialDescriptor blue   = MakePlasticMaterial(Spectrum(.1f, .1f, .4f), 
                                                    Spectrum(0.6f), 0.03);
    MaterialDescriptor greenGlass = MakeGlassMaterial(Spectrum(1), Spectrum(1),
                                                      1.5, 0.05, 0.05);
    
    Transform er = Translate(0, -1.29, 0) * RotateX(90);
    RectDescriptor bottomWall = MakeRectangle(er, 1000, 1000);
    InsertPrimitive(bottomWall, yellow);
    
    Transform rl = Translate(0, 60, 0) * RotateX(90);
    RectDescriptor rect = MakeRectangle(rl, 200, 200);
    MaterialDescriptor matEm = MakeEmissive(Spectrum(0.992, 0.964, 0.890));
    //InsertPrimitive(rect, matEm);
    
    SphereDescriptor lightSphere = MakeSphere(Translate(0, 160, 0), 100);
    InsertPrimitive(lightSphere, matEm);
    
    ParsedMesh *dragonMesh;
    LoadObjData("/home/felpz/Documents/dragon_aligned.obj", &dragonMesh);
    dragonMesh->toWorld = Translate(0, 13,0) * Scale(15) * RotateZ(-15) * RotateY(70);
    MeshDescriptor dragon = MakeMesh(dragonMesh);
    InsertPrimitive(dragon, greenGlass);
}

void CornellBoxScene(Camera *camera, Float aspect){
    AssertA(!!camera, "Invalid camera pointer");
    
    camera->Config(Point3f(0.f, 18.f, -103.f), Point3f(0.0f,15.f,0.f), 
                   vec3f(0.f,1.f,0.f), 40.f, aspect);
    
    MaterialDescriptor matUber = MakeUberMaterial(Spectrum(.05), Spectrum(.8), 
                                                  Spectrum(0), Spectrum(0), 0.001, 
                                                  0.001, Spectrum(1), 1.5f);
    
    MaterialDescriptor red = MakeMatteMaterial(Spectrum(.65, .05, .05));
    MaterialDescriptor green = MakeMatteMaterial(Spectrum(.12, .45, .15));
    MaterialDescriptor white = MakeMatteMaterial(Spectrum(.73, .73, .73));
    
    Transform lr = Translate(30, 25, 0) * RotateY(-90);
    RectDescriptor leftWall = MakeRectangle(lr, 200, 50);
    InsertPrimitive(leftWall, green);
    
    Transform rr = Translate(-30, 25, 0) * RotateY(90);
    RectDescriptor rightWall = MakeRectangle(rr, 200, 50);
    InsertPrimitive(rightWall, red);
    
    Transform br = Translate(0, 25, 20);
    RectDescriptor backWall = MakeRectangle(br, 200, 50);
    InsertPrimitive(backWall, white);
    
    Transform tr = Translate(0, 50, 0) * RotateX(90);
    RectDescriptor topWall = MakeRectangle(tr, 60, 200);
    InsertPrimitive(topWall, white);
    
    Transform er = Translate(0, 0, 0) * RotateX(90);
    RectDescriptor bottomWall = MakeRectangle(er, 400, 400);
    InsertPrimitive(bottomWall, white);
    
    MaterialDescriptor mirror = MakeMirrorMaterial(Spectrum(0.98));
    
    SphereDescriptor glassSphere = MakeSphere(Translate(-13, 8.f, -25), 8);
    MaterialDescriptor matGlass = MakeGlassMaterial(Spectrum(0.9), Spectrum(0.9), 1.5);
    InsertPrimitive(glassSphere, matGlass);
    
    Transform sbt = Translate(-13,4,-25) * RotateY(-30);
    BoxDescriptor box = MakeBox(sbt, 8,8,8); //cornell is 14,14,14
    //InsertPrimitive(box, white);
    
    Transform bbt = Translate(10,18,0) * RotateY(25);
    BoxDescriptor bigBox = MakeBox(bbt, 18,36,18);
    InsertPrimitive(bigBox, white);
    
    Transform r = Translate(0, 49, -10) * RotateX(90);
    RectDescriptor rect = MakeRectangle(r, 30, 30);
    MaterialDescriptor matEm = MakeEmissive(Spectrum(0.992, 0.964, 0.390) * 5);
    //InsertPrimitive(rect, matEm);
    
    //Test for sampling
    SphereDescriptor lightSphere = MakeSphere(Translate(0,45,0), 5);
    InsertPrimitive(lightSphere, matEm);
    
    DiskDescriptor disk = MakeDisk(r, 0, 10, 0, 360);
    //InsertPrimitive(disk, matEm);
    
#if 0
    ParsedMesh *buddaMesh;
    LoadObjData("/home/felpz/Documents/budda.obj", &buddaMesh);
    Float s = 40;
    buddaMesh->toWorld = Translate(10,0,-5) * Scale(s,s,s);
    
    MeshDescriptor budda = MakeMesh(buddaMesh);
    MaterialDescriptor matGlass2 = MakeGlassMaterial(Spectrum(1),
                                                     Spectrum(0.31, 0.64, 0.32), 
                                                     0.02, 0.02, 1.5);
    InsertPrimitive(budda, matGlass2);
#endif
}

void render(Image *image){
    int tx = 8;
    int ty = 8;
    int nx = image->width;
    int ny = image->height;
    int it = 10000;
    unsigned long long seed = time(0);
    Float aspect = (Float)nx / (Float)ny;
    dim3 blocks(nx/tx+1, ny/ty+1);
    dim3 threads(tx,ty);
    
    Aggregator *scene = cudaAllocateVx(Aggregator, 1);
    
    std::cout << "Initializing image..." << std::flush;
    SetupPixels<<<blocks, threads>>>(image, seed);
    cudaDeviceAssert();
    std::cout << "OK" << std::endl;
    
    Camera *camera = cudaAllocateVx(Camera, 1);
    BeginScene(scene);
    
    //NOTE: Use this function to perform scene setup
    //CornellBoxScene(camera, aspect);
    DragonScene(camera, aspect);
    ////////////////////////////////////////////////
    
    std::cout << "Building scene\n" << std::flush;
    PrepareSceneForRendering(scene);
    std::cout << "Done" << std::endl;
    
    std::cout << "Rendering..." << std::endl;
    for(int i = 0; i < it; i++){
        Render<<<blocks, threads>>>(image, scene, camera, 1);
        cudaDeviceAssert();
        graphy_display_pixels(image, i);
        //if(i == 0) getchar();
        std::cout << "\rIteration: " << i << std::flush;
    }
    
    std::cout << std::endl;
    
    ReleaseScene<<<1, 1>>>(scene); //guess we can contiue even if this crashes
    if(cudaSynchronize()){ printf("Failed to free scene\n"); }
    
    cudaFree(scene->handles);//TODO: This needs to transverse the tree
    cudaFree(scene);
    cudaFree(camera);
}

int main(int argc, char **argv){
    if(argc > 1){
        if(argc != 3){
            printf("Converter is: %s <INPUT_PPM_FILE> <OUTPUT_PNG_FILE>\n", argv[0]);
        }else{
            ConvertPPMtoPNG(argv[1], argv[2]);
        }
        return 0;
    }else{
        cudaInitEx();
        
        Float aspect_ratio = 16.0 / 9.0;
        const int image_width = 1366;
        const int image_height = (int)((Float)image_width / aspect_ratio);
        
        Image *image = CreateImage(image_width, image_height);
        
        render(image);
        
        ImageWrite(image, "output.png", 1.f, ToneMapAlgorithm::Exponential);
        ImageFree(image);
        cudaDeviceReset();
        return 0;
    }
}