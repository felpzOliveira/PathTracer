#pragma once

#include <shape.h>

class BSDF;
class SurfaceInteraction;
class Material;
class Pixel;

class Primitive{
    public:
    Bounds3f worldBound;
    __bidevice__ virtual bool Intersect(const Ray &r, SurfaceInteraction *) const = 0;
    __bidevice__ virtual void Release() const = 0;
    __bidevice__ virtual void ComputeScatteringFunctions(BSDF *bsdf, SurfaceInteraction *si,
                                                         TransportMode mode, bool mLobes) 
        const = 0;
    __bidevice__ virtual void PrintSelf() const = 0;
};

class GeometricPrimitive : public Primitive{
    public:
    Shape *shape;
    Material *material;
    
    __bidevice__ GeometricPrimitive(){}
    __bidevice__ GeometricPrimitive(Shape *shape, Material *material);
    
    __bidevice__ virtual bool Intersect(const Ray &r, SurfaceInteraction *) const override;
    __bidevice__ virtual void Release() const override{
        delete shape;
    }
    
    __bidevice__ virtual 
        void ComputeScatteringFunctions(BSDF *bsdf, SurfaceInteraction *si,
                                        TransportMode mode, bool mLobes) const override;
    
    __bidevice__ virtual void PrintSelf() const override{
        printf("Geometric ( ");
        PrintShape(shape);
        printf(" ) \n");
    }
};

typedef struct{
    Bounds3f bound;
    int handle;
}PrimitiveHandle;

typedef struct Node_t{
    struct Node_t *left, *right;
    PrimitiveHandle *handles;
    int n;
    int is_leaf;
    Bounds3f bound;
}Node;

class Aggregator{
    public:
    Primitive **primitives;
    int length;
    int head;
    Node *root;
    PrimitiveHandle *handles;
    int totalNodes;
    
    __bidevice__ Aggregator();
    __bidevice__ void Reserve(int size);
    __bidevice__ void Insert(Primitive *pri);
    __bidevice__ bool Intersect(const Ray &r, SurfaceInteraction *, Pixel *) const;
    __bidevice__ void Release();
    __bidevice__ void PrintHandle(int which=-1);
    __host__ void Wrap();
    
    private:
    __bidevice__ bool IntersectNode(Node *node, const Ray &r, SurfaceInteraction *) const;
};
