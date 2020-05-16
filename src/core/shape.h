#pragma once
#include <geometry.h>
#include <transform.h>
#include <cutil.h>
#include <interaction.h>

enum ShapeType{
    SPHERE, MESH
};

#define MAX_STACK_SIZE 256
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
typedef Node* NodePtr;


__host__ Node *CreateBVH(PrimitiveHandle *handles, int n, int depth, 
                         int max_depth, int *totalNodes);

class Shape{
    public:
    ShapeType type;
    Transform ObjectToWorld, WorldToObject;
    __bidevice__ Shape(const Transform &toWorld) :
    ObjectToWorld(toWorld), WorldToObject(Inverse(toWorld))
    {}
    
    __bidevice__ virtual Bounds3f GetBounds() const{ return Bounds3f(); }
    __bidevice__ virtual bool Intersect(const Ray &ray, Float *tHit,
                                        SurfaceInteraction *isect) const = 0;
};

class Sphere : public Shape{
    public:
    Float radius;
    Float thetaMin, thetaMax, phiMax;
    Float zMin, zMax;
    
    __bidevice__ Sphere(const Transform &toWorld, Float radius, 
                        Float zMin, Float zMax, Float phiMax) : 
    Shape(toWorld), radius(radius),
    zMin(Clamp(Min(zMin, zMax), -radius, radius)),
    zMax(Clamp(Max(zMin, zMax), -radius, radius)),
    thetaMin(std::acos(Clamp(Min(zMin, zMax) / radius, -1, 1))),
    thetaMax(std::acos(Clamp(Max(zMin, zMax) / radius, -1, 1))),
    phiMax(Radians(Clamp(phiMax, 0, 360))) {type = ShapeType::SPHERE;}
    
    __bidevice__ Sphere(const Transform &toWorld, Float radius) :
    Shape(toWorld), radius(radius),
    zMin(-radius), zMax(radius), 
    thetaMin(std::acos(-1.f)), thetaMax(std::acos(1.f)),
    phiMax(Radians(360)){type = ShapeType::SPHERE;}
    
    __bidevice__ virtual bool Intersect(const Ray &ray, Float *tHit,
                                        SurfaceInteraction *isect) const override;
    
    __bidevice__ virtual Bounds3f GetBounds() const override;
};

//Meshes are in world space?
class Mesh: public Shape{
    public:
    int nTriangles, nVertices;
    int *indices;
    Point3f *p;
    Normal3f *n;
    vec3f *s;
    Point2f *uv;
    
    PrimitiveHandle *handles;
    Node *bvh;
    
    __bidevice__ Mesh() : Shape(Transform()){type = ShapeType::MESH;}
    __bidevice__ Mesh(const Transform &toWorld, int nTris, int *_indices,
                      int nVerts, Point3f *P, vec3f *S, Normal3f *N, Point2f *UV);
    
    __bidevice__ void Set(const Transform &toWorld, int nTris, int *_indices,
                          int nVerts, Point3f *P, vec3f *S, Normal3f *N, Point2f *UV);
    
    __bidevice__ virtual bool Intersect(const Ray &ray, Float *tHit,
                                        SurfaceInteraction *isect) const override;
    
    __bidevice__ virtual Bounds3f GetBounds() const override;
    
    private:
    bool __bidevice__ IntersectMeshNode(Node *node, const Ray &r, 
                                        SurfaceInteraction *, Float *) const;
    bool __bidevice__ IntersectTriangle(const Ray &r, SurfaceInteraction * isect,
                                        int triNum, Float *tHit) const;
    
    __bidevice__ void GetUVs(Point2f uv[3], int nTri) const;
};

inline __bidevice__ void PrintShape(Shape *shape){
    if(shape->type == ShapeType::SPHERE){
        Sphere *sphere = (Sphere *)shape;
        Point3f center = (shape->ObjectToWorld)(Point3f(0,0,0));
        printf("Sphere [ " __vec3_strfmtA(center) " , radius: %g ]", 
               __vec3_argsA(center), sphere->radius);
    }else if(shape->type == ShapeType::MESH){
        printf("Mesh []");
    }else{
        printf("None");
    }
}

__host__ void WrapMesh(Mesh *mesh);