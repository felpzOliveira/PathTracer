#pragma once

#include <material.h>
#include <geometry.h>
#include <iostream>
#include <shape.h>
#include <primitive.h>

typedef struct{
    Transform toWorld;
    Float radius;
    int id;
}SphereDescriptor;

typedef struct{
    ParsedMesh *mesh;
    int id;
}MeshDescriptor;

typedef struct{
    int is_emissive;
    MaterialType type;
    Spectrum svals[16];
    Float fvals[16];
    int id;
}MaterialDescriptor;

typedef struct{
    ShapeType shapeType;
    SphereDescriptor sphereDesc;
    MeshDescriptor meshDesc;
    MaterialDescriptor mat;
}PrimitiveDescriptor;

__host__ void                BeginScene(Aggregator *scene);
__host__ SphereDescriptor    MakeSphere(Transform toWorld, Float radius);
__host__ MeshDescriptor      MakeMesh(ParsedMesh *mesh);

__host__ MaterialDescriptor  MakeMatteMaterial(Spectrum kd, Float sigma=0);
__host__ MaterialDescriptor  MakeMirrorMaterial(Spectrum kr);
__host__ MaterialDescriptor  MakeMetalMaterial(Spectrum kr, Float etaI, Float etaT, Float k);
__host__ MaterialDescriptor  MakeGlassMaterial(Spectrum kr, Spectrum kt, Float index,
                                               Float uRough=0, Float vRough=0);

__host__ MaterialDescriptor  MakePlasticMaterial(Spectrum kd, Spectrum ks, Float rough);
__host__ MaterialDescriptor  MakeUberMaterial(Spectrum kd, Spectrum ks, Spectrum kr, 
                                              Spectrum kt, Float uRough, Float vRough,
                                              Spectrum op, Float eta);

//NOTE: There is no emissive material, I'm just reusing this type 
//      for the GeometricEmitterPrimitive
__host__ MaterialDescriptor  MakeEmissive(Spectrum L);

__host__ void                InsertPrimitive(SphereDescriptor shape, MaterialDescriptor mat);
__host__ void                InsertPrimitive(MeshDescriptor shape, MaterialDescriptor mat);
__host__ void                PrepareSceneForRendering(Aggregator *scene);