#pragma once

#include <shape.h>

class BSDF;
class SurfaceInteraction;
class Material;
class Pixel;
class PhaseFunction;

class Primitive{
    public:
    Bounds3f worldBound;
    Shape *shape;
    __bidevice__ Primitive(Shape *shape);
    __bidevice__ virtual bool Intersect(const Ray &r, SurfaceInteraction *) const;
    __bidevice__ virtual void Release() const{ delete shape; }
    __bidevice__ virtual void ComputeScatteringFunctions(BSDF *bsdf, SurfaceInteraction *si,
                                                         TransportMode mode, bool mLobes) 
        const = 0;
    __bidevice__ virtual void PrintSelf() const = 0;
    __bidevice__ virtual Spectrum Le() const{ return Spectrum(0.f); }
};

class GeometricPrimitive : public Primitive{
    public:
    Material *material;
    
    __bidevice__ GeometricPrimitive() : Primitive(nullptr){}
    __bidevice__ GeometricPrimitive(Shape *shape, Material *material);
    
    __bidevice__ virtual 
        void ComputeScatteringFunctions(BSDF *bsdf, SurfaceInteraction *si,
                                        TransportMode mode, bool mLobes) const override;
    
    __bidevice__ virtual void PrintSelf() const override{
        printf("Geometric ( ");
        PrintShape(shape);
        printf(" ) \n");
    }
};

class GeometricEmitterPrimitive : public Primitive{
    public:
    Spectrum L;
    Float power;
    
    __bidevice__ GeometricEmitterPrimitive() : Primitive(nullptr){}
    __bidevice__ GeometricEmitterPrimitive(Shape *shape, Spectrum L, Float power=1);
    __bidevice__ virtual 
        void ComputeScatteringFunctions(BSDF *bsdf, SurfaceInteraction *si,
                                        TransportMode mode, bool mLobes) const override;
    
    __bidevice__ virtual Spectrum Le() const{ return L; }
    
    __bidevice__ virtual void PrintSelf() const override{
        printf("Geometric Emitter( ");
        PrintShape(shape);
        printf(" [ " v3fA(L) " ] ) \n", v3aA(L));
    }
    
};

class Aggregator{
    public:
    Primitive **primitives;
    int length;
    int head;
    Node *root;
    PrimitiveHandle *handles;
    int totalNodes;
    
    Mesh **meshPtrs;
    int nAllowedMeshes;
    int nMeshes;
    int lightList[256];
    int lightCounter;
    
    __bidevice__ Aggregator();
    __bidevice__ void Reserve(int size);
    __bidevice__ void Insert(Primitive *pri, int is_light=0);
    __bidevice__ bool Intersect(const Ray &r, SurfaceInteraction *, Pixel *) const;
    __bidevice__ void Release();
    __bidevice__ void PrintHandle(int which=-1);
    __bidevice__ Mesh *AddMesh(const Transform &toWorld, int nTris, int *_indices,
                               int nVerts, Point3f *P, vec3f *S, Normal3f *N, 
                               Point2f *UV);
    __bidevice__ Mesh *AddMesh(const Transform &toWorld, ParsedMesh *mesh, int copy=0);
    __host__ void ReserveMeshes(int n);
    __host__ void Wrap();
    
    private:
    __bidevice__ bool IntersectNode(Node *node, const Ray &r, SurfaceInteraction *) const;
};