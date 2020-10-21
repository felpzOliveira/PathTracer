#include <scene.h>
#include <cutil.h>
#include <vector>
#include <procedural.h>
#include <image_util.h>

struct MeasuredSS{
    const char *name;
    Float sigma_prime_s[3], sigma_a[3];
};

static MeasuredSS SubsurfaceParameterTable[] = {
    // From "A Practical Model for Subsurface Light Transport"
    // Jensen, Marschner, Levoy, Hanrahan
    // Proc SIGGRAPH 2001
    {"Apple", {2.29, 2.39, 1.97}, {0.0030, 0.0034, 0.046},},
    {"Chicken1", {0.15, 0.21, 0.38}, {0.015, 0.077, 0.19},},
    {"Chicken2", {0.19, 0.25, 0.32}, {0.018, 0.088, 0.20},},
    {"Cream", {7.38, 5.47, 3.15}, {0.0002, 0.0028, 0.0163},},
    {"Ketchup", {0.18, 0.07, 0.03}, {0.061, 0.97, 1.45},},
    {"Marble", {2.19, 2.62, 3.00}, {0.0021, 0.0041, 0.0071},},
    {"Potato", {0.68, 0.70, 0.55}, {0.0024, 0.0090, 0.12},},
    {"Skimmilk", {0.70, 1.22, 1.90}, {0.0014, 0.0025, 0.0142},},
    {"Skin1", {0.74, 0.88, 1.01}, {0.032, 0.17, 0.48},},
    {"Skin2", {1.09, 1.59, 1.79}, {0.013, 0.070, 0.145},},
    {"Spectralon", {11.6, 20.4, 14.9}, {0.00, 0.00, 0.00},},
    {"Wholemilk", {2.55, 3.21, 3.77}, {0.0011, 0.0024, 0.014},},
    
    // From "Acquiring Scattering Properties of Participating Media by
    // Dilution",
    // Narasimhan, Gupta, Donner, Ramamoorthi, Nayar, Jensen
    // Proc SIGGRAPH 2006
    {"Lowfat Milk", {0.89187, 1.5136, 2.532}, {0.002875, 0.00575, 0.0115}},
    {"Reduced Milk",{2.4858, 3.1669, 4.5214},{0.0025556, 0.0051111, 0.012778}},
    {"Regular Milk", {4.5513, 5.8294, 7.136}, {0.0015333, 0.0046, 0.019933}},
    {"Espresso", {0.72378, 0.84557, 1.0247}, {4.7984, 6.5751, 8.8493}},
    {"Mint Mocha Coffee", {0.31602, 0.38538, 0.48131}, {3.772, 5.8228, 7.82}},
    {"Lowfat Soy Milk",{0.30576, 0.34233, 0.61664},{0.0014375, 0.0071875, 0.035937}},
    {"Regular Soy Milk",{0.59223, 0.73866, 1.4693},{0.0019167, 0.0095833, 0.065167}},
    {"Lowfat Chocolate Milk",{0.64925, 0.83916, 1.1057},{0.0115, 0.0368, 0.1564}},
    {"Regular Chocolate Milk",{1.4585, 2.1289, 2.9527},{0.010063, 0.043125, 0.14375}},
    {"Coke", {8.9053e-05, 8.372e-05, 0}, {0.10014, 0.16503, 0.2468}},
    {"Pepsi", {6.1697e-05, 4.2564e-05, 0}, {0.091641, 0.14158, 0.20729}},
    {"Sprite",{6.0306e-06, 6.4139e-06, 6.5504e-06},{0.001886, 0.0018308, 0.0020025}},
    {"Gatorade",{0.0024574, 0.003007, 0.0037325},{0.024794, 0.019289, 0.008878}},
    {"Chardonnay",{1.7982e-05, 1.3758e-05, 1.2023e-05},{0.010782, 0.011855, 0.023997}},
    {"White Zinfandel",{1.7501e-05, 1.9069e-05, 1.288e-05},{0.012072, 0.016184, 0.019843}},
    {"Merlot", {2.1129e-05, 0, 0}, {0.11632, 0.25191, 0.29434}},
    {"Budweiser Beer",{2.4356e-05, 2.4079e-05, 1.0564e-05},{0.011492, 0.024911, 0.057786}},
    {"Coors Light Beer",{5.0922e-05, 4.301e-05, 0},{0.006164, 0.013984, 0.034983}},
    {"Clorox",{0.0024035, 0.0031373, 0.003991},{0.0033542, 0.014892, 0.026297}},
    {"Apple Juice",{0.00013612, 0.00015836, 0.000227},{0.012957, 0.023741, 0.052184}},
    {"Cranberry Juice",{0.00010402, 0.00011646, 7.8139e-05},{0.039437, 0.094223, 0.12426}},
    {"Grape Juice", {5.382e-05, 0, 0}, {0.10404, 0.23958, 0.29325}},
    {"Ruby Grapefruit Juice",{0.011002, 0.010927, 0.011036},{0.085867, 0.18314, 0.25262}},
    {"White Grapefruit Juice",{0.22826, 0.23998, 0.32748},{0.0138, 0.018831, 0.056781}},
    {"Shampoo",{0.0007176, 0.0008303, 0.0009016},{0.014107, 0.045693, 0.061717}},
    {"Strawberry Shampoo",{0.00015671, 0.00015947, 1.518e-05},{0.01449, 0.05796, 0.075823}},
    {"Head & Shoulders Shampoo",{0.023805, 0.028804, 0.034306},{0.084621, 0.15688, 0.20365}},
    {"Lemon Tea Powder",{0.040224, 0.045264, 0.051081},{2.4288, 4.5757, 7.2127}},
    {"Orange Powder",{0.00015617, 0.00017482, 0.0001762},{0.001449, 0.003441, 0.007863}},
    {"Pink Lemonade Powder",{0.00012103, 0.00013073, 0.00012528},{0.001165, 0.002366, 0.003195}},
    {"Cappuccino Powder", {1.8436, 2.5851, 2.1662}, {35.844, 49.547, 61.084}},
    {"Salt Powder", {0.027333, 0.032451, 0.031979}, {0.28415, 0.3257, 0.34148}},
    {"Sugar Powder",{0.00022272, 0.00025513, 0.000271},{0.012638, 0.031051, 0.050124}},
    {"Suisse Mocha Powder", {2.7979, 3.5452, 4.3365}, {17.502, 27.004, 35.433}},
    {"Pacific Ocean Surface Water",{0.0001764, 0.00032095, 0.00019617},{0.031845, 0.031324, 0.030147}}
};

std::vector<PrimitiveDescriptor> hPrimitives;

int any_mesh = 0;
int any_cloud = 0;
int mplayground_ids = 0;

typedef struct{
    MipMap<Spectrum> *lightMap;
    Distribution2D *lightMapDist;
    Transform toWorld;
}LightMap;

typedef struct{
    SphereDescriptor *spheres;
    int nspheres;
    
    MeshDescriptor *meshes;
    int nmeshes;
    
    PrimitiveDescriptor *primitives;
    int nprimitives;
    MediumDescriptor cameraMedium;
    LightMap *lightMap;
}SceneDescription;

SceneDescription *hostScene = nullptr;


__host__ bool GetMediumScatteringProperties(const std::string &name, Spectrum *sigma_a,
                                            Spectrum *sigma_prime_s)
{
    for(MeasuredSS &mss : SubsurfaceParameterTable){
        if(name == mss.name){
            *sigma_a = Spectrum(mss.sigma_a[0], mss.sigma_a[1], mss.sigma_a[2]);
            *sigma_prime_s = Spectrum(mss.sigma_prime_s[0], mss.sigma_prime_s[1],
                                      mss.sigma_prime_s[2]);
            return true;
        }
    }
    
    return false;
}

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
    hostScene->cameraMedium.is_valid = 0;
    hostScene->lightMap = nullptr;
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

__host__ SphereDescriptor MakeSphereProcedural(Transform toWorld, Float radius){
    SphereDescriptor desc;
    desc.toWorld = toWorld;
    desc.radius = radius;
    desc.procedural = true;
    return desc;
}

__host__ SphereDescriptor MakeSphere(Transform toWorld, Float radius,
                                     bool reverseOrientation)
{
    SphereDescriptor desc;
    desc.toWorld = toWorld;
    desc.radius = radius;
    desc.reverseOrientation = reverseOrientation;
    desc.procedural = false;
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

__host__ ParticleCloudDescripor MakeParticleCloud(vec3f *positions, int n, Float size){
    ParticleCloudDescripor desc;
    desc.pos = positions;
    desc.npos = n;
    desc.scale = size;
    any_cloud += 1;
    return desc;
}

__host__ ProcToyDescriptor MakeProceduralToy(Transform toWorld, Bounds3f bound, int id){
    ProcToyDescriptor desc;
    desc.toWorld = toWorld;
    desc.bound = bound;
    desc.id = id;
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

__host__ MaterialDescriptor MakePlayGroundMaterial(Spectrum L, int id){
    MaterialDescriptor desc;
    desc.type = MaterialType::PlayGround;
    desc.textures[0] = MakeTexture(L);
    desc.is_emissive = 0;
    desc.specs_taken = 1;
    desc.id = id > -1 ? id : mplayground_ids;
    mplayground_ids++;
    return desc;
}

__host__ MaterialDescriptor MakePlayGroundMaterial(std::vector<Spectrum> Ls, int id){
    MaterialDescriptor desc;
    int maxc = sizeof(desc.textures) / sizeof(TextureDescriptor);
    AssertA(Ls.size() < maxc, "Too many Spectrums for material");
    desc.type = MaterialType::PlayGround;
    desc.is_emissive = 0;
    for(int i = 0; i < Ls.size(); i++){
        desc.textures[i] = MakeTexture(Ls.at(i));
    }
    
    desc.id = id > -1 ? id : mplayground_ids;
    desc.specs_taken = Ls.size();
    mplayground_ids++;
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

__host__ MaterialDescriptor MakeGlassMaterial(TextureDescriptor kr, TextureDescriptor kt,
                                              Float index, TextureDescriptor rough)
{
    MaterialDescriptor desc;
    desc.is_emissive = 0;
    desc.type = MaterialType::Glass;
    int i = 0;
    desc.textures[i++] = kr;
    desc.textures[i++] = kt;
    desc.textures[i++] = rough;
    desc.textures[i++] = rough;
    desc.textures[i++] = MakeTexture(index);
    return desc;
}

__host__ MaterialDescriptor MakeGlassMaterial(Spectrum kr, Spectrum kt, Float index,
                                              TextureDescriptor rough)
{
    MaterialDescriptor desc;
    desc.is_emissive = 0;
    desc.type = MaterialType::Glass;
    int i = 0;
    desc.textures[i++] = MakeTexture(kr);
    desc.textures[i++] = MakeTexture(kt);
    desc.textures[i++] = rough;
    desc.textures[i++] = rough;
    desc.textures[i++] = MakeTexture(index);
    return desc;
}

__host__ MaterialDescriptor MakeTranslucentMaterial(Spectrum kd, Spectrum ks,
                                                    Spectrum reflect, Spectrum transmit,
                                                    Float rough)
{
    MaterialDescriptor desc;
    desc.type = MaterialType::Translucent;
    desc.is_emissive = 0;
    desc.textures[0] = MakeTexture(kd);
    desc.textures[1] = MakeTexture(ks);
    desc.textures[2] = MakeTexture(reflect);
    desc.textures[3] = MakeTexture(transmit);
    desc.textures[4] = MakeTexture(rough);
    return desc;
}

__host__ MaterialDescriptor MakeKdSubsurfaceMaterial(Spectrum kd, Spectrum kt, Spectrum kr,
                                                     Spectrum mfp, Float urough, Float vrough,
                                                     Float eta, Float scale)
{
    MaterialDescriptor desc;
    desc.type = MaterialType::KdSubsurface;
    desc.is_emissive = 0;
    desc.textures[0] = MakeTexture(kd);
    desc.textures[1] = MakeTexture(kt);
    desc.textures[2] = MakeTexture(kr);
    desc.textures[3] = MakeTexture(mfp);
    desc.textures[4] = MakeTexture(urough);
    desc.textures[5] = MakeTexture(vrough);
    desc.vals[0] = eta;
    desc.vals[1] = scale;
    return desc;
}

__host__ MaterialDescriptor MakeSubsurfaceMaterial(const char *name, Float scale, Float eta, 
                                                   Float g, Float uRough, Float vRough)
{
    MaterialDescriptor desc;
    desc.type = MaterialType::Subsurface;
    Spectrum sig_a(.0011f, .0024f, .014f); 
    Spectrum sig_s(2.55f, 3.21f, 3.77f);
    desc.is_emissive = 0;
    if(!GetMediumScatteringProperties(name, &sig_a, &sig_s)){
        printf("Warning: Could not find scattering properties for %s, using default\n", name);
    }
    
    desc.textures[0] = MakeTexture(Spectrum(1));
    desc.textures[1] = MakeTexture(Spectrum(1));
    desc.textures[2] = MakeTexture(sig_a);
    desc.textures[3] = MakeTexture(sig_s);
    desc.textures[4] = MakeTexture(uRough);
    desc.textures[5] = MakeTexture(vRough);
    desc.vals[0] = scale;
    desc.vals[1] = eta;
    desc.vals[2] = g;
    return desc;
}

__host__ MaterialDescriptor MakeSubsurfaceMaterial(const char *name, Spectrum kr, 
                                                   Spectrum kt, Float scale, Float eta, 
                                                   Float g, Float uRough, Float vRough)
{
    MaterialDescriptor desc;
    desc.type = MaterialType::Subsurface;
    Spectrum sig_a(.0011f, .0024f, .014f); 
    Spectrum sig_s(2.55f, 3.21f, 3.77f);
    desc.is_emissive = 0;
    if(!GetMediumScatteringProperties(name, &sig_a, &sig_s)){
        printf("Warning: Could not find scattering properties for %s, using default\n", name);
    }
    
    desc.textures[0] = MakeTexture(kr);
    desc.textures[1] = MakeTexture(kt);
    desc.textures[2] = MakeTexture(sig_a);
    desc.textures[3] = MakeTexture(sig_s);
    desc.textures[4] = MakeTexture(uRough);
    desc.textures[5] = MakeTexture(vRough);
    desc.vals[0] = scale;
    desc.vals[1] = eta;
    desc.vals[2] = g;
    return desc;
}

__host__ MaterialDescriptor  MakeSubsurfaceMaterial(Spectrum kr, Spectrum kt,
                                                    Spectrum sa, Spectrum ss,
                                                    Float scale, Float eta, Float g,
                                                    Float uRough, Float vRough)
{
    MaterialDescriptor desc;
    desc.type = MaterialType::Subsurface;
    desc.is_emissive = 0;
    desc.textures[0] = MakeTexture(kr);
    desc.textures[1] = MakeTexture(kt);
    desc.textures[2] = MakeTexture(sa);
    desc.textures[3] = MakeTexture(ss);
    desc.textures[4] = MakeTexture(uRough);
    desc.textures[5] = MakeTexture(vRough);
    desc.vals[0] = scale;
    desc.vals[1] = eta;
    desc.vals[2] = g;
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

__host__ MaterialDescriptor MakeMTLMaterial(MTL *mtl){
    MaterialDescriptor desc;
    bool found = false;
    int i = 0;
    bool hasKd = false, hasKdMap = false;
    bool hasKs = false, hasKsMap = false;
    // 1 - check if is emissive
    Spectrum L = MTLGetSpectrum("Ke", mtl, found);
    if(!L.IsBlack() && found){ // light
        desc.is_emissive = 1;
        desc.textures[0] = MakeTexture(L);
        return desc;
    }
    
    // not emissive
    desc.is_emissive = 0;
    /*
    * I don't think we can mix Spectrum and Texture component
    * at the moment so we either accept Kd term or texture if present
    * because usually textures are better we get that instead.
    * This calls for a TextureMixture class but CUDA is very
    * sensitive for self invocation on derived types.
*/
    
    // 2 - get phong based diffuse term
    Spectrum kd = MTLGetSpectrum("Kd", mtl, hasKd);
    std::string kdMap = MTLGetValue("map_Kd", mtl, hasKdMap);
    
    // 3 - get specular based term
    Spectrum ks = MTLGetSpectrum("Ks", mtl, hasKs);
    std::string ksMap = MTLGetValue("map_Ks", mtl, hasKsMap);
    
    // 4 - insert a new texture component for diffuse term
    bool useSpectrumKd = hasKd;
    if(hasKdMap){ // go with map if exists
        std::string path(mtl->basedir);
        path += kdMap;
        ImageData *data = LoadTextureImageData(path.c_str());
        if(data){
            desc.textures[i++] = MakeTexture(data);
            useSpectrumKd = false;
            printf(" * Got Kd map\n");
        }else{
            hasKdMap = false;
        }
    }
    
    if(useSpectrumKd){
        desc.textures[i++] = MakeTexture(kd);
    }
    
    // 5 - insert a new texture component for specular term
    bool useSpectrumKs = hasKs;
    if(hasKsMap){
        std::string path(mtl->basedir);
        path += ksMap;
        ImageData *data = LoadTextureImageData(path.c_str());
        if(data){
            desc.textures[i++] = MakeTexture(data);
            useSpectrumKs = false;
            printf(" * Got Ks map\n");
        }else{
            hasKsMap = false;
        }
    }
    
    if(useSpectrumKs){
        desc.textures[i++] = MakeTexture(ks);
    }
    
    // 6 - define material type
    if(i == 2){ // if we added both this is a plastic material
        Float rough = 0.03;
        desc.textures[i++] = MakeTexture(rough);
        desc.type = MaterialType::Plastic;
    }else if(i == 1){ // we either added ks or kd, if kd than this is matte
        if(hasKs || hasKsMap){ // matte material
            Float sigma = 0;
            desc.textures[i++] = MakeTexture(sigma);
            desc.type = MaterialType::Matte;
        }else{ // uber material
            printf("Warning: broken uber material\n");
            desc = MakeUberMaterial(Spectrum(0.001), ks, Spectrum(0),
                                    Spectrum(0), 0, 0, 1, 1.5f);
        }
    }else{
        printf("Warning: Could not make material for MTL\n");
        desc.textures[i++] = MakeTexture(0.45);
        desc.textures[i++] = MakeTexture(0.);
        desc.type = MaterialType::Matte;
    }
    
    return desc;
}

__host__ MediumDescriptor MakeMedium(Spectrum sigma_a, Spectrum sigma_s, Float g){
    MediumDescriptor desc;
    desc.sa = sigma_a;
    desc.ss = sigma_s;
    desc.g = g;
    desc.is_valid = 1;
    return desc;
}

__host__ void InsertPrimitive(ProcToyDescriptor shape, MaterialDescriptor mat){
    PrimitiveDescriptor desc;
    MediumDescriptor md;
    md.is_valid = 0;
    desc.mediumDesc = md;
    desc.no_mat = 0;
    desc.shapeType = ShapeType::COMPONENT_PROCEDURAL;
    desc.mat = mat;
    desc.toyDesc = shape;
    hPrimitives.push_back(desc);
}

__host__ void InsertPrimitive(ParticleCloudDescripor shape, 
                              MaterialDescriptor mat)
{
    PrimitiveDescriptor desc;
    MediumDescriptor md;
    md.is_valid = 0;
    desc.mediumDesc = md;
    desc.no_mat = 0;
    desc.shapeType = ShapeType::PARTICLECLOUD;
    desc.partDesc = shape;
    desc.mat = mat;
    hPrimitives.push_back(desc);
}

__host__ void InsertPrimitive(SphereDescriptor shape, MaterialDescriptor mat,
                              MediumDescriptor medium)
{
    PrimitiveDescriptor desc;
    desc.no_mat = 0;
    desc.shapeType = ShapeType::SPHERE;
    desc.mat = mat;
    desc.mediumDesc = medium;
    desc.sphereDesc = shape;
    hPrimitives.push_back(desc);
}

__host__ void InsertPrimitive(SphereDescriptor shape, MaterialDescriptor mat){
    MediumDescriptor desc;
    desc.is_valid = 0;
    InsertPrimitive(shape, mat, desc);
}

__host__ void InsertPrimitive(BoxDescriptor shape, MaterialDescriptor mat){
    MediumDescriptor md;
    md.is_valid = 0;
    PrimitiveDescriptor desc;
    desc.mediumDesc = md;
    desc.no_mat = 0;
    desc.shapeType = ShapeType::BOX_PROCEDURAL;
    desc.mat = mat;
    desc.boxDesc = shape;
    hPrimitives.push_back(desc);
}

__host__ void InsertPrimitive(RectDescriptor shape, MaterialDescriptor mat){
    MediumDescriptor md;
    md.is_valid = 0;
    PrimitiveDescriptor desc;
    desc.mediumDesc = md;
    desc.no_mat = 0;
    desc.shapeType = ShapeType::RECTANGLE;
    desc.mat = mat;
    desc.rectDesc = shape;
    hPrimitives.push_back(desc);
}

__host__ void InsertPrimitive(MeshDescriptor shape, MaterialDescriptor mat,
                              MediumDescriptor medium)
{
    PrimitiveDescriptor desc;
    desc.shapeType = ShapeType::MESH;
    desc.mat = mat;
    desc.no_mat = 0;
    desc.mediumDesc = medium;
    desc.meshDesc = shape;
    hPrimitives.push_back(desc);
}

__host__ void InsertPrimitive(MeshDescriptor shape, MaterialDescriptor mat){
    MediumDescriptor desc;
    desc.is_valid = 0;
    InsertPrimitive(shape, mat, desc);
}

__host__ void InsertPrimitive(DiskDescriptor shape, MaterialDescriptor mat){
    MediumDescriptor md;
    md.is_valid = 0;
    PrimitiveDescriptor desc;
    desc.mediumDesc = md;
    desc.no_mat = 0;
    desc.shapeType = ShapeType::DISK;
    desc.mat = mat;
    desc.diskDesc = shape;
    hPrimitives.push_back(desc);
}

__host__ void InsertPrimitive(SphereDescriptor shape, MediumDescriptor medium){
    PrimitiveDescriptor desc;
    desc.no_mat = 1;
    desc.shapeType = ShapeType::SPHERE;
    desc.sphereDesc = shape;
    desc.mediumDesc = medium;
    hPrimitives.push_back(desc);
}
__host__ void InsertPrimitive(MeshDescriptor shape, MediumDescriptor medium){
    PrimitiveDescriptor desc;
    desc.no_mat = 1;
    desc.shapeType = ShapeType::MESH;
    desc.meshDesc = shape;
    desc.mediumDesc = medium;
    hPrimitives.push_back(desc);
}

__host__ void InsertEXRLightMap(const char *path, const Transform &toWorld, 
                                const Spectrum &scale)
{
    if(hostScene){
        if(!hostScene->lightMap){
            LightMap *map = cudaAllocateVx(LightMap, 1);
            printf("Generating environment map\n");
            map->lightMap = BuildSpectrumMipMap(path, &map->lightMapDist, scale);
            map->toWorld = toWorld;
            hostScene->lightMap = map;
        }else{
            printf("Warning: Multiple calls to InsertLightMap\n");
        }
    }else{
        printf("Warning: No call to BeginScene\n");
    }
}

__host__ void InsertCameraMedium(MediumDescriptor medium){
    if(!hostScene){
        printf("Warning: No invocation of BeginScene before camera medium insertion\n");
    }else{
        hostScene->cameraMedium = medium;
    }
}

__bidevice__ Shape *MakeShape(Aggregator *scene, PrimitiveDescriptor *pri){
    Shape *shape = nullptr;
    
    if(pri->shapeType == ShapeType::SPHERE){
        if(pri->sphereDesc.procedural){
            shape = new ProceduralSphere(pri->sphereDesc.toWorld, pri->sphereDesc.radius);
            printf(" * Created procedural sphere\n");
        }else{
            shape = new Sphere(pri->sphereDesc.toWorld, pri->sphereDesc.radius,
                               pri->sphereDesc.reverseOrientation);
        }
    }else if(pri->shapeType == ShapeType::MESH){
        shape = scene->AddMesh(pri->meshDesc.mesh->toWorld, pri->meshDesc.mesh);
    }else if(pri->shapeType == ShapeType::RECTANGLE){
        shape = new Rectangle(pri->rectDesc.toWorld, 
                              pri->rectDesc.sizex, pri->rectDesc.sizey,
                              pri->rectDesc.reverseOrientation);
    }else if(pri->shapeType == ShapeType::BOX_PROCEDURAL){
        shape = new ProceduralBox(pri->boxDesc.toWorld, 
                                  vec3f(pri->boxDesc.sizex, 
                                        pri->boxDesc.sizey,
                                        pri->boxDesc.sizez));
        //shape = new Box(pri->boxDesc.toWorld, pri->boxDesc.sizex,
        //pri->boxDesc.sizey, pri->boxDesc.sizez,
        //pri->boxDesc.reverseOrientation);
    }else if(pri->shapeType == ShapeType::DISK){
        shape = new Disk(pri->diskDesc.toWorld, pri->diskDesc.height,
                         pri->diskDesc.radius, pri->diskDesc.innerRadius,
                         pri->diskDesc.phiMax, pri->diskDesc.reverseOrientation);
    }else if(pri->shapeType == ShapeType::COMPONENT_PROCEDURAL){
        shape = new ProceduralComponent(pri->toyDesc.toWorld, 
                                        pri->toyDesc.bound, pri->toyDesc.id);
    }else if(pri->shapeType == ShapeType::PARTICLECLOUD){
        shape = scene->AddParticleCloud(pri->partDesc.pos, pri->partDesc.npos, 
                                        pri->partDesc.scale);
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

__bidevice__ int SpectrumTextureList(MaterialDescriptor *mat, Texture<Spectrum> ***dlist){
    AssertA(dlist && mat->specs_taken > 0, "Invalid pointer for texture list allocation");
    Texture<Spectrum> **list = new Texture<Spectrum>*[mat->specs_taken];
    for(int i = 0; i < mat->specs_taken; i++){
        list[i] = SpectrumTexture(&mat->textures[i]);
    }
    
    *dlist = list;
    return mat->specs_taken;
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
        
        case MaterialType::Subsurface:{
            material->Set(new SubsurfaceMaterial(SpectrumTexture(&mat->textures[0]),
                                                 SpectrumTexture(&mat->textures[1]),
                                                 SpectrumTexture(&mat->textures[2]),
                                                 SpectrumTexture(&mat->textures[3]),
                                                 FloatTexture(&mat->textures[4]),
                                                 FloatTexture(&mat->textures[5]),
                                                 mat->vals[2], mat->vals[1],
                                                 mat->vals[0]));
        } break;
        
        case MaterialType::KdSubsurface:{
            material->Set(new KdSubsurfaceMaterial(SpectrumTexture(&mat->textures[0]),
                                                   SpectrumTexture(&mat->textures[1]),
                                                   SpectrumTexture(&mat->textures[2]),
                                                   SpectrumTexture(&mat->textures[3]),
                                                   FloatTexture(&mat->textures[4]),
                                                   FloatTexture(&mat->textures[5]),
                                                   mat->vals[0], mat->vals[1]));
        } break;
        
        case MaterialType::Translucent:{
            material->Set(new TranslucentMaterial(SpectrumTexture(&mat->textures[0]),
                                                  SpectrumTexture(&mat->textures[1]),
                                                  SpectrumTexture(&mat->textures[2]),
                                                  SpectrumTexture(&mat->textures[3]),
                                                  FloatTexture(&mat->textures[4])));
        } break;
        
        case MaterialType::PlayGround:{
            Texture<Spectrum> **list = nullptr;
            int size = SpectrumTextureList(mat, &list);
            material->Set(new PlayGroundMaterial(list, size, mat->id));
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
        scene->viewMedium = nullptr;
        scene->Reserve(description->nprimitives);
        
        if(description->cameraMedium.is_valid){
            MediumDescriptor desc = description->cameraMedium;
            scene->viewMedium = new Medium(desc.sa, desc.ss, desc.g);
            printf("Added view medium\n");
        }
        
        for(int i = 0; i < description->nprimitives; i++){
            Primitive *primitive = nullptr;
            Medium *medium = nullptr;
            int is_emissive = 0;
            
            PrimitiveDescriptor *pri = &description->primitives[i];
            Shape *shape = MakeShape(scene, pri);
            
            if(!shape){
                printf("Skipping unknown shape type id %d [ at: %d ]\n", 
                       (int)pri->shapeType, i);
                continue;
            }
            
            if(pri->mat.is_emissive && !pri->no_mat){
                primitive = new GeometricEmitterPrimitive(shape, 
                                                          pri->mat.textures[0].spec_val);
                is_emissive = 1;
            }else{
                Material *mat = nullptr;
                
                if(!pri->no_mat){
                    mat = MakeMaterial(pri);
                    if(!mat){
                        printf("Got nullptr for material given [ at: %d ]\n", i);
                    }
                }
                
                if(pri->mediumDesc.is_valid){
                    medium = new Medium(pri->mediumDesc.sa, pri->mediumDesc.ss,
                                        pri->mediumDesc.g);
                    if(!medium){
                        printf("Got nullptr for medium given [ at: %d ]\n", i);
                    }
                }
                
                primitive = new GeometricPrimitive(shape, mat);
            }
            
            primitive->mediumInterface = MediumInterface(medium, nullptr);
            scene->Insert(primitive, is_emissive);
        }
    }
}

__global__ void MakeLights(Aggregator *scene, SceneDescription *hScene){
    if(threadIdx.x == 0 && blockIdx.x == 0){
        Assert(scene && hScene);
        LightMap *map = hScene->lightMap;
        if(map){
            scene->InsertInfiniteLight(map->lightMap, map->lightMapDist, map->toWorld);
        }
        
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
        if(any_cloud)
            scene->ReserveParticleClouds(any_cloud);
        
        MakeSceneGPU<<<1,1>>>(scene, hostScene);
        cudaDeviceAssert();
        
        scene->Wrap();
        
        Bounds3f sceneSize = scene->root->bound;
        Point3f pMin = sceneSize.pMin;
        Point3f pMax = sceneSize.pMax;
        printf(" * Scene bounds " v3fA(pMin) ", " v3fA(pMax) "\n", 
               v3aA(pMin), v3aA(pMax));
        
        MakeLights<<<1,1>>>(scene, hostScene);
        cudaDeviceAssert();
    }else{
        printf("Invalid scene, you need to call BeginScene once\n");
    }
}

__host__ Spectrum SpectrumFromURGB(Float r, Float g, Float b){
    return Spectrum(r/255.f, g/255.f, b/255.f);
}