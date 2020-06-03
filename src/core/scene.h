#pragma once

#include <material.h>
#include <geometry.h>
#include <iostream>
#include <shape.h>
#include <primitive.h>
#include <texture.h>
#include <mtl.h>

typedef struct{
    Transform toWorld;
    Float radius;
    bool reverseOrientation;
    bool procedural;
}SphereDescriptor;

typedef struct{
    ParsedMesh *mesh;
}MeshDescriptor;

typedef struct{
    Float sizex, sizey;
    Transform toWorld;
    bool reverseOrientation;
}RectDescriptor;

typedef struct{
    Transform toWorld;
    Float sizex, sizey, sizez;
    bool reverseOrientation;
}BoxDescriptor;

typedef struct{
    Transform toWorld;
    Float height, radius, innerRadius, phiMax;
    bool reverseOrientation;
}DiskDescriptor;

typedef struct{
    Spectrum spec_val; //0
    Float fval; //1
    ImageData *image; //2
    int type;
}TextureDescriptor;

typedef struct{
    int is_emissive;
    MaterialType type;
    TextureDescriptor textures[16];
    Float vals[16];
}MaterialDescriptor;

typedef struct{
    Spectrum sa, ss;
    Float g;
    int is_valid;
}MediumDescriptor;

typedef struct{
    ShapeType shapeType;
    SphereDescriptor sphereDesc;
    MeshDescriptor meshDesc;
    RectDescriptor rectDesc;
    BoxDescriptor boxDesc;
    DiskDescriptor diskDesc;
    MaterialDescriptor mat;
    MediumDescriptor mediumDesc;
    int no_mat;
}PrimitiveDescriptor;

__host__ void                BeginScene(Aggregator *scene);
__host__ SphereDescriptor    MakeSphere(Transform toWorld, Float radius, 
                                        bool reverseOrientation=false);
__host__ SphereDescriptor    MakeSphereProcedural(Transform toWorld, Float radius);
__host__ RectDescriptor      MakeRectangle(Transform toWorld, Float sizex, Float sizey, 
                                           bool reverseOrientation=false);
__host__ MeshDescriptor      MakeMesh(ParsedMesh *mesh);
__host__ BoxDescriptor       MakeBox(Transform toWorld, Float sizex, Float sizey, Float sizez, 
                                     bool reverseOrientation=false);
__host__ DiskDescriptor      MakeDisk(Transform toWorld, Float height, Float radius, 
                                      Float innerRadius, Float phiMax, 
                                      bool reverseOrientation=false);

__host__ TextureDescriptor   MakeTexture(Spectrum value);
__host__ TextureDescriptor   MakeTexture(Float value);
__host__ TextureDescriptor   MakeTexture(ImageData *data);

__host__ MaterialDescriptor  MakeMatteMaterial(Spectrum kd, Float sigma=0);
__host__ MaterialDescriptor  MakeMatteMaterial(TextureDescriptor kd, Float sigma=0);
__host__ MaterialDescriptor  MakeMirrorMaterial(Spectrum kr);
__host__ MaterialDescriptor  MakeMetalMaterial(Spectrum kr, Spectrum k, 
                                               Float etaI, Float etaT);
__host__ MaterialDescriptor  MakeGlassMaterial(Spectrum kr, Spectrum kt, Float index,
                                               Float uRough=0, Float vRough=0);

__host__ MaterialDescriptor  MakeSubsurfaceMaterial(Spectrum kr, Spectrum kt,
                                                    Spectrum sa, Spectrum ss,
                                                    Float scale, Float eta, Float g=0,
                                                    Float uRough=0, Float vRough=0);

__host__ MaterialDescriptor  MakeSubsurfaceMaterial(const char *name, Spectrum kr, 
                                                    Spectrum kt, Float scale, Float eta, 
                                                    Float g=0, Float uRough=0, Float vRough=0);

__host__ MaterialDescriptor  MakeSubsurfaceMaterial(const char *name, Float scale, Float eta, 
                                                    Float g=0, Float uRough=0, Float vRough=0);

__host__ MaterialDescriptor  MakePlasticMaterial(Spectrum kd, Spectrum ks, Float rough);
__host__ MaterialDescriptor  MakeUberMaterial(Spectrum kd, Spectrum ks, Spectrum kr, 
                                              Spectrum kt, Float uRough, Float vRough,
                                              Spectrum op, Float eta);

//NOTE: Because MTL is phong based we really have few options for these,
//      something like one of: matte, plastic or uber.
__host__ MaterialDescriptor  MakeMTLMaterial(MTL *mtl);

//NOTE: There is no emissive material, I'm just reusing this type 
//      for the GeometricEmitterPrimitive
__host__ MaterialDescriptor  MakeEmissive(Spectrum L);

__host__ MediumDescriptor    MakeMedium(Spectrum sigma_a, Spectrum sigma_s, Float g);
__host__ void                InsertCameraMedium(MediumDescriptor medium);
__host__ void                InsertEXRLightMap(const char *path, const Transform &toWorld,
                                               const Spectrum &scale);

__host__ void                InsertPrimitive(SphereDescriptor shape, MaterialDescriptor mat);
__host__ void                InsertPrimitive(RectDescriptor shape, MaterialDescriptor mat);
__host__ void                InsertPrimitive(MeshDescriptor shape, MaterialDescriptor mat);
__host__ void                InsertPrimitive(BoxDescriptor shape, MaterialDescriptor mat);
__host__ void                InsertPrimitive(DiskDescriptor shape, MaterialDescriptor mat);

__host__ void                InsertPrimitive(SphereDescriptor shape, MediumDescriptor medium);
__host__ void                InsertPrimitive(MeshDescriptor shape, MediumDescriptor medium);

__host__ void                InsertPrimitive(SphereDescriptor shape, MaterialDescriptor mat,
                                             MediumDescriptor medium);
__host__ void                InsertPrimitive(MeshDescriptor shape, MaterialDescriptor mat,
                                             MediumDescriptor medium);

__host__ void                PrepareSceneForRendering(Aggregator *scene);

__bidevice__ Spectrum        SpectrumFromURGB(Float r, Float g, Float b);