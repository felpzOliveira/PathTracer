#include <scene.h>
#include <cutil.h>
#include <vector>

std::vector<PrimitiveDescriptor> hPrimitives;

int any_mesh = 0;
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
}

__host__ TextureDescriptor MakeTexture(Spectrum value){
    TextureDescriptor desc;
    desc.type = 0;
    desc.spec_val = value;
    return desc;
}

__host__ TextureDescriptor MakeTexture(Float value){
    TextureDescriptor desc;
    desc.type = 1;
    desc.fval = value;
    return desc;
}

__host__ TextureDescriptor MakeTexture(ImageData *data){
    TextureDescriptor desc;
    desc.type = 2;
    desc.image = data;
    return desc;
}

__host__ SphereDescriptor MakeSphere(Transform toWorld, Float radius,
                                     bool reverseOrientation)
{
    SphereDescriptor desc;
    desc.toWorld = toWorld;
    desc.radius = radius;
    desc.reverseOrientation = reverseOrientation;
    return desc;
}

__host__ BoxDescriptor MakeBox(Transform toWorld, Float sizex, Float sizey, Float sizez,
                               bool reverseOrientation)
{
    BoxDescriptor desc;
    desc.toWorld = toWorld;
    desc.sizez = sizez;
    desc.sizex = sizex;
    desc.sizey = sizey;
    desc.reverseOrientation = reverseOrientation;
    return desc;
}

__host__ RectDescriptor MakeRectangle(Transform toWorld, Float sizex, Float sizey,
                                      bool reverseOrientation)
{
    RectDescriptor desc;
    desc.toWorld = toWorld;
    desc.sizex = sizex;
    desc.sizey = sizey;
    desc.reverseOrientation = reverseOrientation;
    return desc;
}

__host__ DiskDescriptor MakeDisk(Transform toWorld, Float height, Float radius, 
                                 Float innerRadius, Float phiMax,
                                 bool reverseOrientation)
{
    DiskDescriptor desc;
    desc.toWorld = toWorld;
    desc.height = height;
    desc.radius = radius;
    desc.innerRadius = innerRadius;
    desc.phiMax = phiMax;
    desc.reverseOrientation = reverseOrientation;
    return desc;
}

__host__ MeshDescriptor MakeMesh(ParsedMesh *mesh){
    MeshDescriptor desc;
    desc.mesh = mesh;
    any_mesh += 1;
    return desc;
}

__host__ MaterialDescriptor MakeMatteMaterial(Spectrum kd, Float sigma){
    MaterialDescriptor desc;
    TextureDescriptor tkd, ts;
    desc.type = MaterialType::Matte;
    desc.is_emissive = 0;
    desc.textures[0] = MakeTexture(kd);
    desc.textures[1] = MakeTexture(sigma);
    return desc;
}

__host__ MaterialDescriptor MakeMatteMaterial(TextureDescriptor kd, Float sigma){
    MaterialDescriptor desc;
    desc.type = MaterialType::Matte;
    desc.is_emissive = 0;
    desc.textures[0] = kd;
    desc.textures[1] = MakeTexture(sigma);
    return desc;
}

__host__ MaterialDescriptor MakeMirrorMaterial(Spectrum kr){
    MaterialDescriptor desc;
    desc.type = MaterialType::Mirror;
    desc.is_emissive = 0;
    desc.textures[0] = MakeTexture(kr);
    return desc;
}

__host__ MaterialDescriptor MakeMetalMaterial(Spectrum kr, Spectrum k, 
                                              Float etaI, Float etaT)
{
    MaterialDescriptor desc;
    int i = 0;
    desc.type = MaterialType::Metal;
    desc.is_emissive = 0;
    desc.textures[i++] = MakeTexture(kr);
    desc.textures[i++] = MakeTexture(k);
    desc.textures[i++] = MakeTexture(etaI);
    desc.textures[i++] = MakeTexture(etaT);
    return desc;
}

__host__ MaterialDescriptor MakeGlassMaterial(Spectrum kr, Spectrum kt, Float index,
                                              Float uRough, Float vRough)
{
    MaterialDescriptor desc;
    int i = 0;
    desc.type = MaterialType::Glass;
    desc.is_emissive = 0;
    desc.textures[i++] = MakeTexture(kr);
    desc.textures[i++] = MakeTexture(kt);
    desc.textures[i++] = MakeTexture(uRough);
    desc.textures[i++] = MakeTexture(vRough);
    desc.textures[i++] = MakeTexture(index);
    return desc;
}

__host__ MaterialDescriptor MakePlasticMaterial(Spectrum kd, Spectrum ks, Float rough){
    MaterialDescriptor desc;
    int i = 0;
    desc.type = MaterialType::Plastic;
    desc.is_emissive = 0;
    desc.textures[i++] = MakeTexture(kd);
    desc.textures[i++] = MakeTexture(ks);
    desc.textures[i++] = MakeTexture(rough);
    return desc;
}

__host__ MaterialDescriptor MakeUberMaterial(Spectrum kd, Spectrum ks, Spectrum kr, 
                                             Spectrum kt, Float uRough, Float vRough,
                                             Spectrum op, Float eta)
{
    MaterialDescriptor desc;
    int i = 0;
    desc.type = MaterialType::Uber;
    desc.is_emissive = 0;
    desc.textures[i++] = MakeTexture(kd);
    desc.textures[i++] = MakeTexture(ks);
    desc.textures[i++] = MakeTexture(kr);
    desc.textures[i++] = MakeTexture(kt);
    desc.textures[i++] = MakeTexture(uRough);
    desc.textures[i++] = MakeTexture(vRough);
    desc.textures[i++] = MakeTexture(op);
    desc.textures[i++] = MakeTexture(eta);
    return desc;
}

__host__ MaterialDescriptor MakeEmissive(Spectrum L){
    MaterialDescriptor desc;
    int i = 0;
    desc.is_emissive = 1;
    desc.textures[i++] = MakeTexture(L);
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

__host__ void InsertPrimitive(DiskDescriptor shape, MaterialDescriptor mat){
    PrimitiveDescriptor desc;
    desc.shapeType = ShapeType::DISK;
    desc.mat = mat;
    desc.diskDesc = shape;
    hPrimitives.push_back(desc);
}

__bidevice__ Shape *MakeShape(Aggregator *scene, PrimitiveDescriptor *pri){
    Shape *shape = nullptr;
    
    if(pri->shapeType == ShapeType::SPHERE){
        shape = new Sphere(pri->sphereDesc.toWorld, pri->sphereDesc.radius,
                           pri->sphereDesc.reverseOrientation);
    }else if(pri->shapeType == ShapeType::MESH){
        shape = scene->AddMesh(pri->meshDesc.mesh->toWorld, pri->meshDesc.mesh);
    }else if(pri->shapeType == ShapeType::RECTANGLE){
        shape = new Rectangle(pri->rectDesc.toWorld, 
                              pri->rectDesc.sizex, pri->rectDesc.sizey,
                              pri->rectDesc.reverseOrientation);
    }else if(pri->shapeType == ShapeType::BOX){
        shape = new Box(pri->boxDesc.toWorld, pri->boxDesc.sizex,
                        pri->boxDesc.sizey, pri->boxDesc.sizez,
                        pri->boxDesc.reverseOrientation);
    }else if(pri->shapeType == ShapeType::DISK){
        shape = new Disk(pri->diskDesc.toWorld, pri->diskDesc.height,
                         pri->diskDesc.radius, pri->diskDesc.innerRadius,
                         pri->diskDesc.phiMax, pri->diskDesc.reverseOrientation);
    }
    
    return shape;
}

__bidevice__ Texture<Float> *FloatTexture(TextureDescriptor *desc){
    if(desc->type == 0){
        printf("Warning: Clamp of Spectrum for texture generation\n");
        return ConstantTexture<Float>(desc->spec_val[0]);
    }else if(desc->type == 1){
        return ConstantTexture<Float>(desc->fval);
    }else{
        return ImageTexture<Float>(desc->image);
    }
}

__bidevice__ Texture<Spectrum> *SpectrumTexture(TextureDescriptor *desc){
    if(desc->type == 0){
        return ConstantTexture<Spectrum>(desc->spec_val);
    }else if(desc->type == 1){
        return ConstantTexture<Spectrum>(desc->fval);
    }else{
        return ImageTexture<Spectrum>(desc->image);
    }
}

__bidevice__ Material *MakeMaterial(PrimitiveDescriptor *pri){
    MaterialDescriptor *mat = &pri->mat;
    Material *material = new Material(mat->type);
    switch(mat->type){
        case MaterialType::Matte:{
            material->Set(new MatteMaterial(SpectrumTexture(&mat->textures[0]),
                                            FloatTexture(&mat->textures[1])));
        } break;
        
        case MaterialType::Mirror:{
            material->Set(new MirrorMaterial(SpectrumTexture(&mat->textures[0])));
        } break;
        
        case MaterialType::Glass:{
            material->Set(new GlassMaterial(SpectrumTexture(&mat->textures[0]),
                                            SpectrumTexture(&mat->textures[1]),
                                            FloatTexture(&mat->textures[2]),
                                            FloatTexture(&mat->textures[3]),
                                            FloatTexture(&mat->textures[4])));
        } break;
        
        case MaterialType::Metal:{
            material->Set(new MetalMaterial(SpectrumTexture(&mat->textures[0]),
                                            SpectrumTexture(&mat->textures[1]),
                                            FloatTexture(&mat->textures[2]),
                                            FloatTexture(&mat->textures[3]),
                                            FloatTexture(&mat->textures[4])));
        } break;
        
        case MaterialType::Plastic:{
            material->Set(new PlasticMaterial(SpectrumTexture(&mat->textures[0]),
                                              SpectrumTexture(&mat->textures[1]),
                                              FloatTexture(&mat->textures[2])));
        } break;
        
        case MaterialType::Uber:{
            material->Set(new UberMaterial(SpectrumTexture(&mat->textures[0]),
                                           SpectrumTexture(&mat->textures[1]),
                                           SpectrumTexture(&mat->textures[2]),
                                           SpectrumTexture(&mat->textures[3]),
                                           FloatTexture(&mat->textures[4]),
                                           FloatTexture(&mat->textures[5]),
                                           SpectrumTexture(&mat->textures[6]),
                                           FloatTexture(&mat->textures[7])));
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
                primitive = new GeometricEmitterPrimitive(shape, 
                                                          pri->mat.textures[0].spec_val);
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
        if(any_mesh > 0)
            scene->ReserveMeshes(any_mesh);
        
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