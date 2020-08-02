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
#include <image_util.h>
#include <mtl.h>
#include <obj_loader.h>
#include <ctime>
#include <sstream>

#define MESH_FOLDER "/home/felpz/Documents/models/"
#define TEXTURE_FOLDER "/home/felpz/Documents/models/Textures/"

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
        image->pixels[pixel_index].accWeight = 0;
        image->pixels[pixel_index].samples = 0;
        memset(&image->pixels[pixel_index].stats, 0, sizeof(PixelStats));
        curand_init(seed, pixel_index, 0, &image->pixels[pixel_index].state);
    }
}

__global__ void ReleaseScene(Aggregator *scene){
    if (threadIdx.x == 0 && blockIdx.x == 0){
        scene->Release();
    }
}

__bidevice__ Spectrum GetSky(vec3f dir){
    //return Spectrum(0);
    vec3f unit = Normalize(dir);
    Float t = 0.5*(dir.y + 1.0);
    return ((1.0-t)*Spectrum(1.0, 1.0, 1.0) + t*Spectrum(0.5, 0.7, 1.0));
}

/*
* 1 bounce Direct lighting. This is mostly debug code used for checking
* if direct lighting is working.
*/
__device__ Spectrum Li_Direct(Ray ray, Aggregator *scene, Pixel *pixel){
    Spectrum L(0.f);
    SurfaceInteraction isect;
    curandState *state = &pixel->state;
    if(!scene->Intersect(ray, &isect, pixel)){
        for(int i = 0; i < scene->lightCounter; i++){
            Light *light = scene->lights[i];
            L += light->Le(ray);
        }
    }else{
        BSDF bsdf(isect);
        
        isect.ComputeScatteringFunctions(&bsdf, ray, TransportMode::Radiance, true);
        vec3f wo = isect.wo;
        L += isect.Le(wo);
        
        if(scene->lightCounter > 0){
            Point2f u2(rand_float(state), rand_float(state));
            Point3f u3(rand_float(state), rand_float(state), rand_float(state));
            Spectrum Ld = scene->UniformSampleOneLight(isect, &bsdf, u2, u3);
            //printf(v3fA(Ld) "\n", v3aA(Ld));
            L += Ld;
        }
    }
    
    return L;
}

/*
* Volumetric Path Tracer. The only code for now that can capture mediums,
* several rays are traced for each medium interaction so be aware that things
* might get complicated. 
*/
__device__ Spectrum Li_VolPath(Ray r, Aggregator *scene, Pixel *pixel){
    Spectrum L(0.f), beta(1.f);
    RayDifferential ray(r);
    bool specularBounce = false;
    bool rayFromMedium = scene->viewMedium ? true : false;
    int bounces;
    int max_bounces = 15;
    Float rrThreshold = 1;
    curandState *state = &pixel->state;
    
    for(bounces = 0; ; bounces++){
        SurfaceInteraction isect;
        bool foundIntersection = scene->Intersect(ray, &isect, pixel);
        
        pixel->stats.totalPaths++;
        pixel->stats.misses += !foundIntersection ? 1 : 0;
        
        MediumInteraction mi;
        if(ray.medium){
            Point2f u(rand_float(state), rand_float(state));
            beta *= ray.medium->Sample(ray, u, &mi);
        }
        
        if(beta.IsBlack()) break;
        if(mi.IsValid()){
            if(bounces >= max_bounces) break;
            
            Point2f u2(rand_float(state), rand_float(state));
            Point3f u3(rand_float(state), rand_float(state), rand_float(state));
            
            Spectrum Ld = beta * scene->UniformSampleOneLight(mi, nullptr, u2, u3, true);
            if(Ld.IsBlack()){
                pixel->stats.zeroRadiancePaths ++;
            }
            
            L += Ld;
            
            vec3f wo = -ray.d, wi;
            u2 = Point2f(rand_float(state), rand_float(state));
            mi.phase.Sample_p(wo, &wi, u2);
            ray = mi.SpawnRay(wi);
            specularBounce = false;
            rayFromMedium = true;
            pixel->stats.mediumHits++;
        }else{
            if(bounces == 0 || specularBounce){
                if(foundIntersection){
                    L += beta * isect.Le(-ray.d);
                }else{
                    for(int i = 0; i < scene->lightCounter; i++){
                        Light *light = scene->lights[i];
                        L += beta * light->Le(ray);
                    }
                }
            }
            
            if(!foundIntersection || bounces >= max_bounces){ break; }
            
            AssertA(isect.primitive, "Intersection without primitive");
            
            if(isect.primitive->IsEmissive()){
                pixel->stats.lightHits++;
                if(rayFromMedium)
                    pixel->stats.lightHitFromMedium ++;
            }
            
            rayFromMedium = false;
            
            BSDF bsdf(isect);
            isect.ComputeScatteringFunctions(&bsdf, ray, TransportMode::Radiance, true);
            
            if(bsdf.nBxDFs == 0){
                ray = isect.SpawnRay(ray.d);
                bounces--;
                continue;
            }
            
            if(bsdf.NumComponents(BxDFType(BSDF_ALL & ~BSDF_SPECULAR)) > 0){
                Point2f u2(rand_float(state), rand_float(state));
                Point3f u3(rand_float(state), rand_float(state), rand_float(state));
                Spectrum Ld = beta * scene->UniformSampleOneLight(isect, &bsdf, u2, 
                                                                  u3, true);
                if(Ld.IsBlack()){
                    pixel->stats.zeroRadiancePaths ++;
                }
                
                L += Ld;
            }
            
            Float pdf = 0.f;
            Point2f u(rand_float(state), rand_float(state));
            vec3f wi, wo = -ray.d;
            BxDFType flags;
            
            Spectrum f = bsdf.Sample_f(wo, &wi, u, &pdf, BSDF_ALL, &flags);
            if(f.IsBlack() || IsZero(pdf)) break;
            
            beta *= f * AbsDot(wi, ToVec3(isect.n)) / pdf;
            specularBounce = (flags & BSDF_SPECULAR) != 0;
            
            ray = isect.SpawnRay(wi);
            
            if(bsdf.has_bssrdf && (flags & BSDF_TRANSMISSION)){
                SurfaceInteraction pi;
                bsdf.nBxDFs = 0;
                Float u1 = rand_float(state);
                Point2f u2(rand_float(state), rand_float(state));
                Spectrum S = bsdf.bssrdf.Sample_S(scene, &bsdf, u1, u2, &pi, &pdf, &isect);
                
                if(S.IsBlack() || IsZero(pdf)) break;
                beta *= S / pdf;
                u2 = Point2f(rand_float(state), rand_float(state));
                Point3f u3(rand_float(state), rand_float(state), rand_float(state));
                Spectrum uL = scene->UniformSampleOneLight(pi, &bsdf, u2, u3, true);
                L += beta * uL;
                
                u2 = Point2f(rand_float(state), rand_float(state));
                Spectrum f = bsdf.Sample_f(pi.wo, &wi, u2, &pdf, BSDF_ALL, &flags);
                
                if(f.IsBlack() || IsZero(pdf)) break;
                beta *= f * AbsDot(wi, ToVec3(pi.n)) / pdf;
                specularBounce = (flags & BSDF_SPECULAR) != 0;
                ray = pi.SpawnRay(wi);
            }
            
            pixel->stats.hits++;
        }
        
        Spectrum rrBeta = beta;
        if(MaxComponent(rrBeta) < rrThreshold && bounces > 3){
            Float q = Max((Float).05, 1 - MaxComponent(rrBeta));
            Float u = rand_float(state);
            if (u < q) break;
            beta = beta / (1 - q);
        }
    }
    
    return L;
}

/*
* Sampled Path Tracer. Performs light sampling at each intersection that is
* not specular, for large scenes with small lights this is the one you want,
* each ray needs to hit the scene 3 times so it might be slower than Brute-Force
* for some scenes, but overall converges faster.
*/
__device__ Spectrum Li_PathSampled(Ray r, Aggregator *scene, Pixel *pixel){
    Spectrum L(0.f), beta(1.f);
    RayDifferential ray(r);
    bool specularBounce = false;
    int bounces;
    int max_bounces = 4;
    Float rrThreshold = 1;
    
    curandState *state = &pixel->state;
    for(bounces = 0; ; bounces++){
        SurfaceInteraction isect;
        bool foundIntersection = scene->Intersect(ray, &isect, pixel);
        
        pixel->stats.totalPaths++;
        pixel->stats.misses += !foundIntersection ? 1 : 0;
        
        if(bounces == 0 || specularBounce){
            if(foundIntersection){
                L += beta * isect.Le(-ray.d);
            }else{
                for(int i = 0; i < scene->lightCounter; i++){
                    Light *light = scene->lights[i];
                    L += beta * light->Le(ray);
                }
            }
        }
        
        if(!foundIntersection || bounces >= max_bounces){ break; }
        
        AssertA(isect.primitive, "Intersection without primitive");
        
        if(isect.primitive->IsEmissive()){
            pixel->stats.lightHits++;
        }
        
        BSDF bsdf(isect);
        
        isect.ComputeScatteringFunctions(&bsdf, ray, TransportMode::Radiance, true);
        if(bsdf.nBxDFs == 0){
            ray = isect.SpawnRay(ray.d);
            bounces--;
            continue;
        }
        
        if(bsdf.NumComponents(BxDFType(BSDF_ALL & ~BSDF_SPECULAR)) > 0){
            Point2f u2(rand_float(state), rand_float(state));
            Point3f u3(rand_float(state), rand_float(state), rand_float(state));
            Spectrum Ld = beta * scene->UniformSampleOneLight(isect, &bsdf, u2, u3);
            if(Ld.IsBlack()){
                pixel->stats.zeroRadiancePaths++;
            }
            L += Ld;
        }
        
        Float pdf = 0.f;
        Point2f u(rand_float(state), rand_float(state));
        vec3f wi, wo = -ray.d;
        BxDFType flags;
        
        Spectrum f = bsdf.Sample_f(wo, &wi, u, &pdf, BSDF_ALL, &flags);
        
        if(f.IsBlack() || IsZero(pdf)) break;
        
        beta *= f * AbsDot(wi, ToVec3(isect.n)) / pdf;
        specularBounce = (flags & BSDF_SPECULAR) != 0;
        
        ray = isect.SpawnRay(wi);
        pixel->stats.hits++;
        
        if(bsdf.has_bssrdf && (flags & BSDF_TRANSMISSION)){
            SurfaceInteraction pi;
            bsdf.nBxDFs = 0;
            Float u1 = rand_float(state);
            Point2f u2(rand_float(state), rand_float(state));
            Spectrum S = bsdf.bssrdf.Sample_S(scene, &bsdf, u1, u2, &pi, &pdf, &isect);
            if(S.IsBlack() || IsZero(pdf)) break;
            beta *= S / pdf;
            
            u2 = Point2f(rand_float(state), rand_float(state));
            Point3f u3(rand_float(state), rand_float(state), rand_float(state));
            Spectrum uL = scene->UniformSampleOneLight(pi, &bsdf, u2, u3);
            L += beta * uL;
            
            u2 = Point2f(rand_float(state), rand_float(state));
            Spectrum f = bsdf.Sample_f(pi.wo, &wi, u2, &pdf, BSDF_ALL, &flags);
            
            if(f.IsBlack() || IsZero(pdf)) break;
            
            beta *= f * AbsDot(wi, ToVec3(pi.n)) / pdf;
            specularBounce = (flags & BSDF_SPECULAR) != 0;
            ray = pi.SpawnRay(wi);
        }
        
        Spectrum rrBeta = beta;
        if(MaxComponent(rrBeta) < rrThreshold && bounces > 3) {
            Float q = Max((Float).05, 1 - MaxComponent(rrBeta));
            Float u = rand_float(state);
            if (u < q) break;
            beta = beta / (1 - q);
        }
    }
    
    return L;
}


/*
* Brute force Path Tracer. It is actually faster than performing light sampling
* if your scene has many secondary light effects and a decent light source such
* as sky or big lights. 
*/
__device__ Spectrum Li_Path(Ray ray, Aggregator *scene, Pixel *pixel){
    Spectrum L(0.f);
    Spectrum beta(1.f);
    Float rrThreshold = 1;
    int max_bounces = 10;
    int bounces = 0;
    curandState *state = &pixel->state;
    for(bounces = 0; bounces < max_bounces; bounces++){
        SurfaceInteraction isect;
        
        if(scene->Intersect(ray, &isect, pixel)){
            BSDF bsdf(isect);
            
            Float pdf = 0.f;
            Point2f u(rand_float(state), rand_float(state));
            vec3f wi, wo = -ray.d;
            
            isect.ComputeScatteringFunctions(&bsdf, ray, TransportMode::Radiance, true);
            Spectrum Le = isect.primitive->Le();
            L += beta * Le;
            
            Spectrum f = bsdf.Sample_f(wo, &wi, u, &pdf, BSDF_ALL);
            if(IsZero(pdf)) break;
            
            beta *= f * AbsDot(wi, ToVec3(isect.n)) / pdf;
            ray = isect.SpawnRay(wi);
            pixel->stats.hits += 1;
            
            Spectrum rrBeta = beta;
            if(MaxComponent(rrBeta) < rrThreshold && bounces > 3) {
                Float q = Max((Float).05, 1 - MaxComponent(rrBeta));
                Float u = rand_float(state);
                if (u < q) break;
                beta = beta / (1 - q);
            }
        }else{
            
            for(int i = 0; i < scene->lightCounter; i++){
                Light *light = scene->lights[i];
                L += beta * light->Le(ray);
            }
            
            pixel->stats.misses += 1;
            break;
        }
    }
    
    if(bounces == max_bounces-1) return Spectrum(0.f);
    return L;
}

__bidevice__ Float GetFilterWeight(Float u0, Float u1, Float u, Float v){
    Float x = Absf(u - u0);
    Float y = Absf(v - u1);
    //return Max(0.f, 1.f - Absf(x)) * Max(0.f, 1.f - Absf(y)); //triangle
    Float alpha = 2.f;
    Float e = Exp(-alpha);
    return Max(0.f, Float(Exp(-alpha * x * x) - e)) * Max(0.f, Float(Exp(-alpha * y * y) - e));
}

__global__ void Render(Image *image, Aggregator *scene, Camera *camera, int ns){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    int width = image->width;
    int height = image->height;
    
    camera->SetMedium(scene->viewMedium);
    
    if(i < width && j < height){
        int pixel_index = j * width + i;
        Pixel *pixel = &image->pixels[pixel_index];
        curandState state = pixel->state;
        
        Spectrum out = image->pixels[pixel_index].we;
        Float u0 = (Float)i / (Float)width;
        Float u1 = (Float)j / (Float)height;
        for(int n = 0; n < ns; n++){
            Float u = ((Float)i + rand_float(&state)) / (Float)width;
            Float v = ((Float)j + rand_float(&state)) / (Float)height;
            
            Point2f sample = ConcentricSampleDisk(rand_point2(&state));
            Ray ray = camera->SpawnRay(u, v, sample);
            //Spectrum L = Li_Direct(ray, scene, pixel);
            //Spectrum L = Li_Path(ray, scene, pixel);
            //Spectrum L = Li_PathSampled(ray, scene, pixel);
            Spectrum L = Li_VolPath(ray, scene, pixel);
            
            Float w = GetFilterWeight(u0, u1, u, v);
            out += w * L;
            pixel->accWeight += w;
            pixel->samples ++;
        }
        
        image->pixels[pixel_index].we = out;
    }
}

void DragonScene(Camera *camera, Float aspect){
    AssertA(camera, "Invalid camera pointer");
    
    camera->Config(Point3f(60.f, 69.f, -43.f), 
                   Point3f(-12.12833, 0., 3.16229),
                   //camera->Config(Point3f(0.f, 48.f, -70.f), 
                   //Point3f(0, 10.15341, 10.16229),
                   vec3f(0.f,1.f,0.f), 38.f, aspect);
    
    MaterialDescriptor gray = MakePlasticMaterial(Spectrum(.1,.1, .1), 
                                                  Spectrum(.7, .7, .7), 0.1);
    
    MaterialDescriptor blue   = MakePlasticMaterial(Spectrum(.1f, .1f, .4f), 
                                                    Spectrum(0.6f), 0.03);
    
    Transform er = Translate(0, 0, 0) * RotateX(90);
    //RectDescriptor bottomWall = MakeRectangle(er, 500, 500);
    SphereDescriptor bottomWall = MakeSphere(Translate(0, -1500, 0), 1500);
    
    
    InsertEXRLightMap(TEXTURE_FOLDER "skylight-sunset.exr",  
                      RotateX(-95) * RotateZ(50), Spectrum(2.5));
    
    ParsedMesh *dragonMesh = LoadObjOnly(MESH_FOLDER "dragon_aligned.obj");
    dragonMesh->toWorld = Scale(15) * RotateZ(-1) * RotateY(70);
    MeshDescriptor dragon = MakeMesh(dragonMesh);
    InsertPrimitive(dragon, blue);
    InsertPrimitive(bottomWall, gray);
}

void BoxesScene(Camera *camera, Float aspect){
    AssertA(camera, "Invalid camera pointer");
    camera->Config(Point3f(478, 278, -600), Point3f(-70, 298, 0),
                   //camera->Config(Point3f(1000, 278, 0), Point3f(-70, 298, 0),
                   vec3f(0, 1, 0), 40.f, aspect);
    
    MaterialDescriptor matGreenFloor = MakeMatteMaterial(Spectrum(0.48, 0.83, 0.53));
    const int boxes_per_side = 20;
    for(int i = 0; i < boxes_per_side; i++){
        for(int j = 0; j < boxes_per_side; j++){
            Float w = 100.0;
            Float x0 = -1000.0 + i*w;
            Float z0 = -1000.0 + j*w;
            Float y0 = 0.0;
            Float x1 = x0 + w;
            Float y1 = 1.f + 101 * rand_float();
            Float z1 = z0 + w;
            Point3f c((x0+x1)/2,(y0+y1)/2,(z0+z1)/2);
            
            Transform model = Translate(ToVec3(c));
            BoxDescriptor box = MakeBox(model, Absf(x0-x1), Absf(y0-y1), Absf(z0-z1));
            InsertPrimitive(box, matGreenFloor);
        }
    }
    
    MaterialDescriptor matWhite = MakeMatteMaterial(Spectrum(.73));
    int ns = 1000;
    for(int j = 0; j < ns; j++){
        vec3f p(165*rand_float()-120, 165*rand_float() + 250, 165*rand_float()-100);
        SphereDescriptor sph = MakeSphere(Translate(p), 10);
        InsertPrimitive(sph, matWhite);
    }
    
    MaterialDescriptor mirror = MakeMirrorMaterial(Spectrum(0.98));
    MaterialDescriptor red = MakePlasticMaterial(Spectrum(0.87,0.23,0.16), Spectrum(0.9), 0.3);
    Transform tBox = Translate(300,300,0);
    BoxDescriptor box = MakeBox(tBox, 100, 100, 100);
    //InsertPrimitive(box, red);
    
    Transform tDisk = Translate(200, 300, 0) * RotateZ(20) * RotateX(-23);
    Float diskRadius = 90;
    DiskDescriptor disk = MakeDisk(tDisk, 0, diskRadius, 0, 360);
    InsertPrimitive(disk, mirror);
    DiskDescriptor diskBorder = MakeDisk(tDisk, 0, diskRadius+10, diskRadius, 360);
    InsertPrimitive(diskBorder, red);
    
    MaterialDescriptor matBlue = MakeMatteMaterial(Spectrum(0.1,0.1,0.4));
    MaterialDescriptor matGlass = MakeGlassMaterial(Spectrum(1), Spectrum(1), 1.5);
    MaterialDescriptor matLucy = MakePlasticMaterial(Spectrum(0.98, 0.56, 0.), 
                                                     Spectrum(0.9), 0.1);
    
    vec3f center(120,160,-35);
    Float baseRadius = 70;
    SphereDescriptor glassSphere = MakeSphere(Translate(center), baseRadius+0.1);
    InsertPrimitive(glassSphere, matGlass);
    SphereDescriptor blueSphere = MakeSphere(Translate(center), baseRadius);
    InsertPrimitive(blueSphere, matBlue);
    
    SphereDescriptor cSphere = MakeSphere(Translate(50, 150, -150), 50);
    InsertPrimitive(cSphere, matGlass);
    
    ParsedMesh *winged;
    LoadObjData(MESH_FOLDER "winged.obj", &winged);
    winged->toWorld = Translate(-100,100,-200) * RotateY(65);
    MeshDescriptor wingedDesc = MakeMesh(winged);
    InsertPrimitive(wingedDesc, matLucy);
    
    ParsedMesh *budda;
    LoadObjData(MESH_FOLDER "budda.obj", &budda);
    budda->toWorld = Translate(160, 20, -270) * Scale(100) * RotateY(180);
    MeshDescriptor buddaDesc = MakeMesh(budda);
    InsertPrimitive(buddaDesc, red);
    
    MaterialDescriptor matEm = MakeEmissive(Spectrum(7));
    Transform lightModel = Translate(0,554,0) * RotateX(90);
    RectDescriptor light = MakeRectangle(lightModel, 300, 265);
    InsertPrimitive(light, matEm);
    
    MediumDescriptor medium = MakeMedium(Spectrum(0.0003), Spectrum(0.0005), -0.7);
    InsertCameraMedium(medium);
}


void Helmet(Camera *camera, Float aspect){
    camera->Config(Point3f(-1.401275, -1.344, -10), Point3f(-0.7501275, -0.7f, 0.0f),
                   vec3f(0,1,0), 22, aspect);
    
    std::vector<MeshMtl> mtls;
    std::vector<ParsedMesh *> *meshes = LoadObj(MESH_FOLDER "helmet.obj", &mtls, true);
    Transform r = RotateY(180);
    
    
#if 1
    MaterialDescriptor kdSub = MakeKdSubsurfaceMaterial(Spectrum(.9),
                                                        Spectrum(1), Spectrum(1),
                                                        Spectrum(.1), 0, 0, 
                                                        1.33, 1.);
    for(int i = 0; i < meshes->size(); i++){
        ParsedMesh *m = meshes->at(i);
        if(m->nTriangles > 0){
            m->toWorld = r;
            // get object mtl
            if(mtls[i].name == "lamps"){
                continue;
            }
            
            MeshDescriptor d = MakeMesh(m);
            if(mtls[i].name == "Material.001"){
                MaterialDescriptor ub = MakeUberMaterial(Spectrum(.02), Spectrum(.0), 
                                                         Spectrum(0), Spectrum(0), 0.01, 
                                                         0.01, Spectrum(1), 1.5f);
                InsertPrimitive(d, ub);
            }else if(mtls[i].name == "Material.002"){
                //MaterialDescriptor pl = MakePlasticMaterial(Spectrum(.1), Spectrum(.7), .05);
                //InsertPrimitive(d, pl);
            }else if(mtls[i].name == "white"){
                InsertPrimitive(d, kdSub);
            }else{
                printf(" ** [ERROR] : Unhandled material\n");
                cudaSafeExit();
            }
            
        }
    }
#endif
    MaterialDescriptor gray = MakeMatteMaterial(Spectrum(.1, .1, .1));
    
    Transform er = Translate(0, -6, 0) * RotateX(90);
    RectDescriptor bottomWall = MakeRectangle(er, 1000, 1000);
    InsertPrimitive(bottomWall, gray);
    
    InsertEXRLightMap(TEXTURE_FOLDER "skylight-sunset.exr",
                      RotateX(-90) * RotateZ(90), Spectrum(2.5));
}


void CornellRoom(Camera *camera, Float aspect){
    camera->Config(Point3f(-1, 1, -30), Point3f(0, 8.0f, 0.0f),
                   vec3f(0,1,0), 40, aspect);
    
    MaterialDescriptor white = MakeMatteMaterial(Spectrum(.73, .73, .73));
    MaterialDescriptor orange = MakeMatteMaterial(Spectrum(.73, .53, .33));
    MaterialDescriptor red = MakeMatteMaterial(Spectrum(.65, .05, .05));
    ImageData *desert = LoadTextureImageData(TEXTURE_FOLDER "greek.jpg");
    ImageData *grid = LoadTextureImageData(TEXTURE_FOLDER "orange.png");
    ImageData *texView = LoadTextureImageData(MESH_FOLDER "Shabti.jpg");
    MaterialDescriptor matUber = MakeUberMaterial(Spectrum(.05), Spectrum(.8), 
                                                  Spectrum(0), Spectrum(0), 0.001, 
                                                  0.001, Spectrum(1), 1.5f);
    
    TextureDescriptor deserttex = MakeTexture(desert);
    MaterialDescriptor desertMat = MakeMatteMaterial(deserttex);
    TextureDescriptor gridtex = MakeTexture(grid);
    MaterialDescriptor gridMat = MakeMatteMaterial(gridtex);
    TextureDescriptor texViewtex = MakeTexture(texView);
    MaterialDescriptor texMat = MakeMatteMaterial(texViewtex);
    
    Float height = 17;
    Float width = height * desert->GetAspectRatio();
    
    Transform backT = Translate(0, height/2, -15) * RotateY(0);
    RectDescriptor backWall = MakeRectangle(backT, width, height);
    InsertPrimitive(backWall, desertMat);
    
    Transform rightT = Translate(-width/2, height/2, 0) * RotateY(90);
    RectDescriptor rightWall = MakeRectangle(rightT, width+20, height);
    InsertPrimitive(rightWall, gridMat);
    
    Transform leftT = Translate(width/2, height/2, 0) * RotateY(90);
    RectDescriptor leftWall = MakeRectangle(leftT, width+20, height);
    InsertPrimitive(leftWall, gridMat);
    
    Transform bottomT = Translate(0, 0, -height/2) * RotateX(90);
    RectDescriptor bottomWall = MakeRectangle(bottomT, width, height+100);
    InsertPrimitive(bottomWall, white);
    
    Transform topT = Translate(0, height, -height/2) * RotateX(90);
    RectDescriptor topWall = MakeRectangle(topT, width, height+100);
    InsertPrimitive(topWall, white);
    
    std::vector<MeshMtl> mtls;
    std::vector<MTL *> mMaterials;
    std::vector<ParsedMesh *> *meshes = LoadObj(MESH_FOLDER "afrodite.obj", &mtls, true);
    bool rv = MTLParseAll(&mMaterials, &mtls, MESH_FOLDER);
    
    Transform ss = Translate(0,0,-20) * Scale(0.175) * RotateY(180);
    for(int i = 0; i < meshes->size(); i++){
        ParsedMesh *m = meshes->at(i);
        if(m->nTriangles > 0){
            m->toWorld = ss;
            // get object mtl
            MTL *mPtr = MTLFindMaterial(mtls[i].name.c_str(), &mMaterials);
            MeshDescriptor d = MakeMesh(m);
            MaterialDescriptor mat = MakeMTLMaterial(mPtr);
            InsertPrimitive(d, mat);
        }
    }
    
    std::vector<MeshMtl> mtlsShield;
    std::vector<MTL *> matShield;
    std::vector<ParsedMesh *> *mShield = LoadObj(MESH_FOLDER "shield.obj", &mtlsShield, true);
    rv = MTLParseAll(&matShield, &mtlsShield, MESH_FOLDER);
    
    ss = Translate(0, 0.7,-21) * Scale(0.08) * RotateZ(30) * RotateX(-60);
    for(int i = 0; i < mShield->size(); i++){
        ParsedMesh *m = mShield->at(i);
        if(m->nTriangles > 0){
            m->toWorld = ss;
            // get object mtl
            MTL *mPtr = MTLFindMaterial(mtlsShield[i].name.c_str(), &matShield);
            MeshDescriptor d = MakeMesh(m);
            MaterialDescriptor mat = MakeMTLMaterial(mPtr);
            InsertPrimitive(d, mat);
        }
    }
    
    MaterialDescriptor matEm = MakeEmissive(Spectrum(0.992, 0.964, 0.390) * 10);
    
    Transform r = Translate(0, height - 0.001, -15) * RotateX(90);
    RectDescriptor rect = MakeRectangle(r, height*0.8, width*0.5);
    InsertPrimitive(rect, matEm);
}


void Vader(Camera *camera, Float aspect){
    AssertA(camera, "Invalid camera pointer");
    
    camera->Config(Point3f(0.f, 62.f, 90.f),
                   Point3f(-10, 0, -20), //fov = 42
                   vec3f(0.f,1.f,0.f), 42.f, aspect);
    
    MaterialDescriptor gray = MakeMatteMaterial(Spectrum(.1, .1, .1));
    MaterialDescriptor glass = MakeGlassMaterial(Spectrum(1), Spectrum(1), 1.5);
    
    Transform er = Translate(0, 0, 0) * RotateX(90);
    RectDescriptor bottomWall = MakeRectangle(er, 1000, 1000);
    InsertPrimitive(bottomWall, gray);
    
    InsertEXRLightMap(TEXTURE_FOLDER "skylight-sunset.exr",  
                      RotateX(-90) * RotateZ(90), Spectrum(2.5));
    
    MaterialDescriptor kdsub = MakeKdSubsurfaceMaterial(Spectrum(.2, .5, .7),
                                                        Spectrum(1), Spectrum(1),
                                                        Spectrum(.1), 0, 0, 1.33, 1);
    
    ParsedMesh *darth = LoadObjOnly(MESH_FOLDER "darth.obj");
    darth->toWorld = Translate(-10, 0, 33) * Scale(6.7) * RotateZ(4);
    MeshDescriptor darthDesc = MakeMesh(darth);
    InsertPrimitive(darthDesc, kdsub);
}


void CornellRoomBasic(Camera *camera, Float aspect){
    camera->Config(Point3f(0, 4, -25), Point3f(0, 4, 0.0f),
                   vec3f(0,1,0), 40, aspect);
    
    MaterialDescriptor white  = MakeMatteMaterial(Spectrum(.73, .73, .73));
    MaterialDescriptor green  = MakeMatteMaterial(Spectrum(.05, .65, .05));
    MaterialDescriptor red    = MakeMatteMaterial(Spectrum(.65, .05, .05));
    MaterialDescriptor mirror = MakeMirrorMaterial(Spectrum(.9));
    MaterialDescriptor blue   = MakeGlassMaterial(Spectrum(1), Spectrum(1), 1.34);
    ImageData *image = LoadTextureImageData(TEXTURE_FOLDER "grid.png");
    TextureDescriptor gridTex = MakeTexture(image);
    
    MaterialDescriptor uvMat = MakeMatteMaterial(gridTex);
    
    Float height = 14;
    Float width = height * 1.2;
    Float extend = 30;
    Transform backT = Translate(0, height/2, 0) * RotateY(0);
    RectDescriptor backWall = MakeRectangle(backT, width, height);
    InsertPrimitive(backWall, white);
    
    Transform rightT = Translate(-width/2, height/2, -height/2) * RotateY(90);
    RectDescriptor rightWall = MakeRectangle(rightT, width+extend, height);
    InsertPrimitive(rightWall, red);
    
    Transform leftT = Translate(width/2, height/2, -height/2) * RotateY(90);
    RectDescriptor leftWall = MakeRectangle(leftT, width+extend, height);
    InsertPrimitive(leftWall, green);
    
    Float bottomExt = 5;
    Transform bottomT = Translate(0, 0, -(height+bottomExt)/2) * RotateX(90);
    RectDescriptor bottomWall = MakeRectangle(bottomT, width, height+bottomExt);
    InsertPrimitive(bottomWall, uvMat);
    
    Transform frontT = Translate(0,height/2,-28);
    RectDescriptor frontWall = MakeRectangle(frontT, width, height);
    InsertPrimitive(frontWall, white);
    
    Transform topT = Translate(0, height, -height/2) * RotateX(90);
    RectDescriptor topWall = MakeRectangle(topT, width, height);
    InsertPrimitive(topWall, white);
    
    Transform boxT = Translate(-3.5, 2,-width*0.5) * RotateY(-25);
    BoxDescriptor box = MakeBox(boxT, 4, 4, 4);
    InsertPrimitive(box, white);
    
    Transform boxBT = Translate(3.0, 4.25, -width*0.25) * RotateY(30);
    BoxDescriptor bbox = MakeBox(boxBT, 4.5, 8.5, 4.5);
    InsertPrimitive(bbox, mirror);
    
    MaterialDescriptor matEm = MakeEmissive(Spectrum(0.992, 0.964, 0.390) * 6);
    Transform r = Translate(0, height-0.01, -height/2) * RotateX(90);
    DiskDescriptor disk = MakeDisk(r, 0, width*0.15, 0.f, 360);
    InsertPrimitive(disk, matEm);
}

void CornellRoomSkull(Camera *camera, Float aspect){
    camera->Config(Point3f(0, 4, -25), Point3f(0, 4, 0.0f),
                   vec3f(0,1,0), 40, aspect);
    
    MaterialDescriptor white  = MakeMatteMaterial(Spectrum(.73, .73, .73));
    MaterialDescriptor green  = MakeMatteMaterial(Spectrum(.05, .65, .05));
    MaterialDescriptor red    = MakeMatteMaterial(Spectrum(.65, .05, .05));
    MaterialDescriptor wPlast = MakePlasticMaterial(Spectrum(.53), Spectrum(1.f), 30);
    
    Float height = 14;
    Float width = height * 1.2;
    Float extend = 30;
    Transform backT = Translate(0, height/2, 0) * RotateY(0);
    RectDescriptor backWall = MakeRectangle(backT, width, height);
    InsertPrimitive(backWall, white);
    
    Transform rightT = Translate(-width/2, height/2, -height/2) * RotateY(90);
    RectDescriptor rightWall = MakeRectangle(rightT, width+extend, height);
    InsertPrimitive(rightWall, red);
    
    Transform leftT = Translate(width/2, height/2, -height/2) * RotateY(90);
    RectDescriptor leftWall = MakeRectangle(leftT, width+extend, height);
    InsertPrimitive(leftWall, green);
    
    Float bottomExt = 5;
    Transform bottomT = Translate(0, 0, -(height+bottomExt)/2) * RotateX(90);
    RectDescriptor bottomWall = MakeRectangle(bottomT, width, height+bottomExt);
    InsertPrimitive(bottomWall, white);
    
    Transform frontT = Translate(0,height/2,-28);
    RectDescriptor frontWall = MakeRectangle(frontT, width, height);
    InsertPrimitive(frontWall, white);
    
    Transform topT = Translate(0, height, -height/2) * RotateX(90);
    RectDescriptor topWall = MakeRectangle(topT, width, height);
    InsertPrimitive(topWall, white);
    
    ParsedMesh *skull = LoadObjOnly(MESH_FOLDER "skull2.obj");
    skull->toWorld = Translate(0.0, 3.33, -width*0.45) * Scale(0.05);
    MeshDescriptor skDesc = MakeMesh(skull);
    InsertPrimitive(skDesc, wPlast);
    
    MaterialDescriptor matEm = MakeEmissive(Spectrum(0.992, 0.964, 0.390) * 6);
    Transform r = Translate(0, height-0.01, -height/2) * RotateX(90);
    DiskDescriptor disk = MakeDisk(r, 0, width*0.15, 0.f, 360);
    InsertPrimitive(disk, matEm);
    
    MediumDescriptor medium = MakeMedium(Spectrum(.008), Spectrum(.01), 0);
    InsertCameraMedium(medium);
}

void OrigamiScene(Camera *camera, Float aspect){
    AssertA(camera, "Invalid camera pointer");
    Point3f origin = Point3f(0.f, 62.f, 70.f);
    Point3f target = Point3f(-10, 23, 0);
    camera->Config(origin, target, vec3f(0.f,1.f,0.f), 50.f, aspect);//35
    
    std::vector<Spectrum> boatSpectrums({ SpectrumFromURGB(139,69,19), Spectrum(.87, .12, .27),
                                        SpectrumFromURGB(244,164,96) });
    
    std::vector<Spectrum> whaleSpectrums({ Spectrum(.02, .21, .54), Spectrum(.1, .32, .76),
                                         Spectrum(.86, .86, .98) });
    
    MaterialDescriptor gray = MakeMatteMaterial(Spectrum(0.9, 0.9, 0.7));
    MaterialDescriptor dragonMat = MakePlayGroundMaterial(Spectrum(.85, .12, .02));
    MaterialDescriptor boatMat = MakePlayGroundMaterial(boatSpectrums);
    MaterialDescriptor whaleMat = MakePlayGroundMaterial(whaleSpectrums);
    MaterialDescriptor birdMat = MakePlayGroundMaterial(Spectrum(.85, .42, .52));
    
    Transform er = Translate(0, 0, 0) * RotateX(90);
    RectDescriptor bottomWall = MakeRectangle(er, 2000, 2000);
    InsertPrimitive(bottomWall, gray);
    
    InsertEXRLightMap(TEXTURE_FOLDER "skylight-sunset.exr", 
                      RotateX(-90) * RotateZ(90) * RotateY(180), Spectrum(2.5));
    
    Float len = 15;
    Float h = 1.5;
    Float h2 = h * 0.5 * len;
    
    Transform transf = Translate(-len * 0.5, h2 * 0.93, 0) * RotateX(-20);
    Float depth = 0.5;
    Bounds3f bound(Point3f(-len, -h2, -depth), Point3f(len, h2, depth));
    ProcToyDescriptor dragon = MakeProceduralToy(transf, bound, 1);
    
    transf = Translate(-len * 2.5, h2 * 0.93, len);
    ProcToyDescriptor whale = MakeProceduralToy(transf, bound, 2);
    
    len = 19;
    h = 2.0;
    h2 = h * 0.5 * len;
    transf = Translate(-len * 0.5, h2, -len) * RotateY(60);
    bound = Bounds3f(Point3f(-len, -h2, -len), Point3f(len, h2, len));
    ProcToyDescriptor boat = MakeProceduralToy(transf, bound, 0);
    
    InsertPrimitive(dragon, dragonMat);
    InsertPrimitive(boat, boatMat);
    InsertPrimitive(whale, whaleMat);
    
    Bounds3f bo2(Point3f(-len), Point3f(len));
    transf = Translate(-3*len, 2.0 * len, -len) * RotateY(-45);
    ProcToyDescriptor tt = MakeProceduralToy(transf, bo2, 3);
    InsertPrimitive(tt, birdMat);
    
    transf = Translate(len, 1.6 * len, -len * 0.5) * RotateY(200);
    ProcToyDescriptor box = MakeProceduralToy(transf, bo2, 3);
    InsertPrimitive(box, dragonMat);
}


void SubsurfaceSpheres(Camera *camera, Float aspect){
    AssertA(camera, "Invalid camera pointer");
    
    camera->Config(Point3f(0.f, 62.f, 90.f),
                   Point3f(-10, 0, -20),
                   vec3f(0.f,1.f,0.f), 42.f, aspect);
    
    MaterialDescriptor gray = MakePlasticMaterial(Spectrum(.1), Spectrum(.7), 0.1);
    
    Transform er = Translate(0, 0, 0) * RotateX(90);
    RectDescriptor bottomWall = MakeRectangle(er, 1000, 1000);
    InsertPrimitive(bottomWall, gray);
    
    MaterialDescriptor ket = MakeSubsurfaceMaterial("Ketchup", 1, 
                                                    1.33, 0, 0.05, 0.05);
    
    MaterialDescriptor spec = MakeKdSubsurfaceMaterial(Spectrum(.2, .5, .7),
                                                       Spectrum(1), Spectrum(1),
                                                       Spectrum(.1), 0.05, 0.05, 
                                                       1.33, 1);
    
    MaterialDescriptor coke = MakeSubsurfaceMaterial("Coke", 1, 
                                                     1.33, 0, 0.05, 0.05);
    MaterialDescriptor apple = MakeSubsurfaceMaterial("Apple", 1, 
                                                      1.33, 0, 0.05, 0.05);
    
    InsertEXRLightMap(TEXTURE_FOLDER "20060807_wells6_hd.exr",  
                      RotateX(-90) * RotateZ(90), Spectrum(2.5));
    
    SphereDescriptor desc0 = MakeSphere(Translate(20, 20, 10), 18);
    SphereDescriptor desc1 = MakeSphere(Translate(-30, 20, 10), 18);
    SphereDescriptor desc2 = MakeSphere(Translate(6, 20, -20), 18);
    SphereDescriptor desc3 = MakeSphere(Translate(-60, 20, -20), 18);
    
    InsertPrimitive(desc0, ket);
    InsertPrimitive(desc1, apple);
    InsertPrimitive(desc2, spec);
    InsertPrimitive(desc3, coke);
}

void TwoDragonsScene(Camera *camera, Float aspect){
    AssertA(camera, "Invalid camera pointer");
    
    camera->Config(Point3f(0.f, 62.f, 90.f),
                   Point3f(-10, 0, -10),
                   vec3f(0.f,1.f,0.f), 42.f, aspect);
    
    MaterialDescriptor gray = MakeMatteMaterial(Spectrum(.1, .1, .1));
    
    Transform er = Translate(0, 0, 0) * RotateX(90);
    RectDescriptor bottomWall = MakeRectangle(er, 1000, 1000);
    InsertPrimitive(bottomWall, gray);
    
    MaterialDescriptor skin = MakeSubsurfaceMaterial("Ketchup", 10, 
                                                     1.33, 0, 0.05, 0.05);
    MaterialDescriptor milk = MakeSubsurfaceMaterial("Skin1", 10, 
                                                     1.33, 0, 0.05, 0.05);
    InsertEXRLightMap(TEXTURE_FOLDER "skylight-sunset.exr",  
                      RotateX(-90) * RotateZ(90), Spectrum(2.5));
    
    ParsedMesh *sssDragon = LoadObjOnly(MESH_FOLDER "sssDragonAligned.obj");
    MeshProperties props; props.flip_x = 0;
    ParsedMesh *sssDragonCopy = DuplicateMesh(sssDragon, &props);
    
    sssDragonCopy->toWorld = Translate(15, 0, 10) * Scale(0.4) * RotateY(280) * RotateZ(4);
    sssDragon->toWorld = Translate(-25, 0, 15) * Scale(0.4) * RotateY(266) * RotateZ(4);
    MeshDescriptor sssDragonDesc = MakeMesh(sssDragon);
    MeshDescriptor sssDragonDescCopy = MakeMesh(sssDragonCopy);
    
    InsertPrimitive(sssDragonDesc, milk);
    InsertPrimitive(sssDragonDescCopy, skin);
}

void VolumetricCausticsScene(Camera *camera, Float aspect){
    AssertA(camera, "Invalid camera pointer");
    
    camera->Config(Point3f(0.f, 18.f, 70.f),
                   Point3f(0, 21.15341, 0),
                   vec3f(0.f,1.f,0.f), 40.f, aspect);
    
    MaterialDescriptor gray = MakePlasticMaterial(Spectrum(.1,.1, .1), 
                                                  Spectrum(.7, .7, .7), 0.1);
    MaterialDescriptor yellow = MakeMatteMaterial(Spectrum(.7,.7,.0));
    MaterialDescriptor white  = MakeMatteMaterial(Spectrum(0.87, 0.87, 0.87));
    MaterialDescriptor orange = MakeMatteMaterial(Spectrum(0.98, 0.56, 0.0));
    MaterialDescriptor red    = MakeMatteMaterial(Spectrum(0.86, 0.2, 0.1));
    MaterialDescriptor blue   = MakePlasticMaterial(Spectrum(.1f, .1f, .4f), 
                                                    Spectrum(0.6f), 0.03);
    MaterialDescriptor greenGlass = MakeGlassMaterial(Spectrum(1), Spectrum(1), 1.5);
    
    Transform er = Translate(0, -1.29, 0) * RotateX(90);
    RectDescriptor bottomWall = MakeRectangle(er, 1000, 1000);
    InsertPrimitive(bottomWall, yellow);
    
    Transform rl = Translate(0, 60, 0) * RotateX(90);
    RectDescriptor rect = MakeRectangle(rl, 200, 200);
    MaterialDescriptor matEm = MakeEmissive(Spectrum(0.992, 0.964, 0.890) * 35);
    SphereDescriptor lightSphere = MakeSphere(Translate(15, 60, 3), 4);
    SphereDescriptor lightSphere2 = MakeSphere(Translate(-15, 60, 3), 4);
    InsertPrimitive(lightSphere, matEm);
    InsertPrimitive(lightSphere2, matEm);
    
    MediumDescriptor medium = MakeMedium(Spectrum(.0007), Spectrum(0.005), 0);
    Transform glassT = Translate(0, 26, 0) * RotateY(45);
    SphereDescriptor spehre = MakeSphere(glassT, 13);
    InsertPrimitive(spehre, greenGlass);
    InsertCameraMedium(medium);
}

void render(Image *image){
    int tx = 8;
    int ty = 8;
    int nx = image->width;
    int ny = image->height;
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
    
    //NOTE: Use this place to set the scene
    ////////////////////////////////////////////////
    //CornellRoomBasic(camera, aspect);
    CornellRoomSkull(camera, aspect);
    //OrigamiScene(camera, aspect);
    //Helmet(camera, aspect);
    //Vader(camera, aspect);
    //VolumetricCausticsScene(camera, aspect);
    //TwoDragonsScene(camera, aspect);
    //SubsurfaceSpheres(camera, aspect);
    //BoxesScene(camera, aspect);
    ////////////////////////////////////////////////
    
    std::cout << "Building scene\n" << std::flush;
    PrepareSceneForRendering(scene);
    std::cout << "Done" << std::endl;
    
    int it = 8000;
    clock_t start = clock();
    std::cout << "Rendering..." << std::endl;
    for(int i = 0; i < it; i++){
        Render<<<blocks, threads>>>(image, scene, camera, 1);
        cudaDeviceAssert();
        clock_t end = clock();
        graphy_display_pixels(image, i);
        //if(i == 0) getchar();
        std::string ctime = get_time_string(start, end, i, it);
        std::cout << "\rSamples: " << (i+1) << " " << ctime << std::flush;
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
            printf("Converter is: %s <PATH_TO_PPM> <PATH_TO_PNG>\n", argv[0]);
        }else{
            ConvertPPMtoPNG(argv[1], argv[2]);
        }
        
        return 0;
    }else{
        cudaInitEx();
        
        Float aspect_ratio = 16.0 / 9.0;
        //Float aspect_ratio = 4.0 / 3.0;
        //int image_width = 1600;
        int image_width = 800;
        int image_height = (int)((Float)image_width / aspect_ratio);
#if 0
        image_width = 1200;
        image_height = 900;
#endif
        aspect_ratio = ((Float)image_width / (Float)image_height);
        
        Image *image = CreateImage(image_width, image_height);
        
        render(image);
        
        ImageWrite(image, "output.png", 2.f, ToneMapAlgorithm::Exponential);
        
        ImageStats(image);
        
        ImageFree(image);
        cudaDeviceReset();
        return 0;
    }
}
