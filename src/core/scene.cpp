#include <scene.h>
#include <cutil.h>
#include <vector>

std::vector<PrimitiveDescriptor> hPrimitives;
int spheresIds = 0;
int meshesIds = 0;
int matId = 0;
int rectsIds = 0;

typedef struct{
    SphereDescriptor *spheres;
    int nspheres;
    
    MeshDescriptor *meshes;
    int nmeshes;
    
    PrimitiveDescriptor *primitives;
    int nprimitives;
}SceneDescription;

SceneDescription *hostScene = nullptr;

__host__ void BeginScene(Aggregator *scene){
    if(hostScene){
        if(hostScene->primitives) cudaFree(hostScene->primitives);
        if(hostScene->meshes) cudaFree(hostScene->meshes);
        if(hostScene->spheres) cudaFree(hostScene->spheres);
    }else{
        hostScene = cudaAllocateVx(SceneDescription, 1);
    }
    
    hPrimitives.clear();
    hostScene->nprimitives = 0;
    hostScene->nspheres = 0;
    hostScene->nmeshes = 0;
    spheresIds = 0;
    meshesIds = 0;
}

__host__ SphereDescriptor MakeSphere(Transform toWorld, Float radius){
    SphereDescriptor desc;
    desc.toWorld = toWorld;
    desc.radius = radius;
    desc.id = spheresIds;
    spheresIds++;
    return desc;
}

__host__ BoxDescriptor MakeBox(Transform toWorld, Float sizex, Float sizey, Float sizez){
    BoxDescriptor desc;
    desc.toWorld = toWorld;
    desc.sizez = sizez;
    desc.sizex = sizex;
    desc.sizey = sizey;
    return desc;
}

__host__ RectDescriptor MakeRectangle(Transform toWorld, Float sizex, Float sizey){
    RectDescriptor desc;
    desc.toWorld = toWorld;
    desc.sizex = sizex;
    desc.sizey = sizey;
    desc.id = rectsIds;
    rectsIds++;
    return desc;
}

__host__ MeshDescriptor MakeMesh(ParsedMesh *mesh){
    MeshDescriptor desc;
    desc.mesh = mesh;
    desc.id = meshesIds;
    meshesIds++;
    return desc;
}

__host__ MaterialDescriptor MakeMatteMaterial(Spectrum kd, Float sigma){
    MaterialDescriptor desc;
    desc.type = MaterialType::Matte;
    desc.is_emissive = 0;
    desc.svals[0] = kd;
    desc.fvals[0] = sigma;
    desc.id = matId;
    matId++;
    return desc;
}

__host__ MaterialDescriptor MakeMirrorMaterial(Spectrum kr){
    MaterialDescriptor desc;
    desc.type = MaterialType::Mirror;
    desc.is_emissive = 0;
    desc.svals[0] = kr;
    desc.id = matId;
    matId++;
    return desc;
}

__host__ MaterialDescriptor MakeMetalMaterial(Spectrum kr, Float etaI, Float etaT, Float k){
    MaterialDescriptor desc;
    desc.type = MaterialType::Metal;
    desc.is_emissive = 0;
    desc.svals[0] = kr;
    desc.fvals[0] = etaI; desc.fvals[1] = etaT; desc.fvals[2] = k;
    desc.id = matId;
    matId++;
    return desc;
}

__host__ MaterialDescriptor MakeGlassMaterial(Spectrum kr, Spectrum kt, Float index,
                                              Float uRough, Float vRough)
{
    MaterialDescriptor desc;
    desc.type = MaterialType::Glass;
    desc.is_emissive = 0;
    desc.svals[0] = kr; desc.svals[1] = kt;
    desc.fvals[0] = uRough; desc.fvals[1] = vRough; desc.fvals[2] = index;
    desc.id = matId;
    matId++;
    return desc;
}

__host__ MaterialDescriptor MakePlasticMaterial(Spectrum kd, Spectrum ks, Float rough){
    MaterialDescriptor desc;
    desc.type = MaterialType::Plastic;
    desc.is_emissive = 0;
    desc.svals[0] = kd; desc.svals[1] = ks;
    desc.fvals[0] = rough;
    desc.id = matId;
    matId++;
    return desc;
}

__host__ MaterialDescriptor MakeUberMaterial(Spectrum kd, Spectrum ks, Spectrum kr, 
                                             Spectrum kt, Float uRough, Float vRough,
                                             Spectrum op, Float eta)
{
    MaterialDescriptor desc;
    desc.type = MaterialType::Uber;
    desc.is_emissive = 0;
    desc.svals[0] = kd; desc.svals[1] = ks; desc.svals[2] = kr; 
    desc.svals[3] = kt; desc.svals[4] = op;
    
    desc.fvals[0] = uRough; desc.fvals[1] = vRough; desc.fvals[2] = eta;
    desc.id = matId;
    matId++;
    return desc;
}

__host__ MaterialDescriptor MakeEmissive(Spectrum L){
    MaterialDescriptor desc;
    desc.is_emissive = 1;
    desc.svals[0] = L;
    desc.id = matId;
    matId++;
    return desc;
}

__host__ void InsertPrimitive(SphereDescriptor shape, MaterialDescriptor mat){
    PrimitiveDescriptor desc;
    desc.shapeType = ShapeType::SPHERE;
    desc.mat = mat;
    desc.sphereDesc = shape;
    hPrimitives.push_back(desc);
}

__host__ void InsertPrimitive(BoxDescriptor shape, MaterialDescriptor mat){
    PrimitiveDescriptor desc;
    desc.shapeType = ShapeType::BOX;
    desc.mat = mat;
    desc.boxDesc = shape;
    hPrimitives.push_back(desc);
}

__host__ void InsertPrimitive(RectDescriptor shape, MaterialDescriptor mat){
    PrimitiveDescriptor desc;
    desc.shapeType = ShapeType::RECTANGLE;
    desc.mat = mat;
    desc.rectDesc = shape;
    hPrimitives.push_back(desc);
}

__host__ void InsertPrimitive(MeshDescriptor shape, MaterialDescriptor mat){
    PrimitiveDescriptor desc;
    desc.shapeType = ShapeType::MESH;
    desc.mat = mat;
    desc.meshDesc = shape;
    hPrimitives.push_back(desc);
}

__bidevice__ Shape *MakeShape(Aggregator *scene, PrimitiveDescriptor *pri){
    Shape *shape = nullptr;
    
    if(pri->shapeType == ShapeType::SPHERE){
        shape = new Sphere(pri->sphereDesc.toWorld, pri->sphereDesc.radius);
    }else if(pri->shapeType == ShapeType::MESH){
        shape = scene->AddMesh(pri->meshDesc.mesh->toWorld, pri->meshDesc.mesh);
    }else if(pri->shapeType == ShapeType::RECTANGLE){
        shape = new Rectangle(pri->rectDesc.toWorld, 
                              pri->rectDesc.sizex, pri->rectDesc.sizey);
    }else if(pri->shapeType == ShapeType::BOX){
        shape = new Box(pri->boxDesc.toWorld, pri->boxDesc.sizex,
                        pri->boxDesc.sizey, pri->boxDesc.sizez);
    }
    
    return shape;
}

__bidevice__ Material *MakeMaterial(PrimitiveDescriptor *pri){
    Material *material = new Material();
    MaterialDescriptor *mat = &pri->mat;
    switch(mat->type){
        case MaterialType::Matte:{
            material->Init_Matte(mat->svals[0], mat->fvals[0]);
        } break;
        
        case MaterialType::Mirror:{
            material->Init_Mirror(mat->svals[0]);
        } break;
        
        case MaterialType::Glass:{
            material->Init_Glass(mat->svals[0], mat->svals[1], mat->fvals[0],
                                 mat->fvals[1], mat->fvals[2]);
        } break;
        
        case MaterialType::Metal:{
            material->Init_Metal(mat->svals[0], mat->fvals[0], mat->fvals[1], mat->fvals[2]);
        } break;
        
        case MaterialType::Plastic:{
            material->Init_Plastic(mat->svals[0], mat->svals[1], mat->fvals[0]);
        } break;
        
        case MaterialType::Uber:{
            material->Init_Uber(mat->svals[0], mat->svals[1], mat->svals[2], mat->svals[3],
                                mat->fvals[0], mat->fvals[1], mat->svals[4], mat->fvals[2]);
        } break;
        
        default:{
            delete material;
            material = nullptr;
        }
    }
    
    return material;
}

__global__ void MakeSceneGPU(Aggregator *scene, SceneDescription *description){
    if(threadIdx.x == 0 && blockIdx.x == 0){
        Assert(description->nprimitives > 0);
        
        scene->Reserve(description->nprimitives);
        for(int i = 0; i < description->nprimitives; i++){
            Primitive *primitive = nullptr;
            
            PrimitiveDescriptor *pri = &description->primitives[i];
            Shape *shape = MakeShape(scene, pri);
            if(!shape){
                printf("Skipping unknown shape type id %d [ at: %d ]\n", 
                       (int)pri->shapeType, i);
                continue;
            }
            
            if(pri->mat.is_emissive){
                primitive = new GeometricEmitterPrimitive(shape, pri->mat.svals[0]);
            }else{
                Material *mat = MakeMaterial(pri);
                if(!mat){
                    printf("Got nullptr for material given [ at: %d ]\n", i);
                }
                
                primitive = new GeometricPrimitive(shape, mat);
            }
            
            scene->Insert(primitive, pri->mat.is_emissive);
        }
    }
}

__global__ void MakeDiffuseLights(Aggregator *scene){
    if(threadIdx.x == 0 && blockIdx.x == 0){
        Assert(scene);
        scene->SetLights();
    }
}

__host__ void PrepareSceneForRendering(Aggregator *scene){
    if(hostScene){
        size_t size = sizeof(PrimitiveDescriptor) * hPrimitives.size();
        hostScene->primitives = cudaAllocateVx(PrimitiveDescriptor, hPrimitives.size());
        memcpy(hostScene->primitives, hPrimitives.data(), size);
        hostScene->nprimitives = hPrimitives.size();
        if(meshesIds > 0)
            scene->ReserveMeshes(meshesIds);
        
        MakeSceneGPU<<<1,1>>>(scene, hostScene);
        cudaDeviceAssert();
        
        scene->Wrap();
        
        printf(" * Scene contains %d light(s)\n", scene->lightCounter);
        
        MakeDiffuseLights<<<1,1>>>(scene);
        cudaDeviceAssert();
        
    }else{
        printf("Invalid scene, you need to call BeginScene once\n");
    }
}