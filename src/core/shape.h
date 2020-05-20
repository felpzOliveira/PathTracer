#pragma once
#include <geometry.h>
#include <transform.h>
#include <cutil.h>
#include <interaction.h>
#include <util.h>

enum ShapeType{
    SPHERE, RECTANGLE, DISK, BOX, MESH
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
    
    __bidevice__ virtual Float Area() const = 0;
    
    __bidevice__ virtual Interaction Sample(const Point2f &u, Float *pdf) const = 0;
    __bidevice__ virtual Interaction Sample(const Interaction &ref, const Point2f &u,
                                            Float *pdf) const = 0;
    __bidevice__ virtual Float Pdf(const Interaction &ref, const vec3f &wi) const;
    __bidevice__ virtual Float Pdf(const Interaction &) const{ return 1 / Area(); }
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
    __bidevice__ virtual Float Area() const override;
    
    __bidevice__ virtual Interaction Sample(const Point2f &u, Float *pdf) const override;
    __bidevice__ virtual Interaction Sample(const Interaction &ref, const Point2f &u,
                                            Float *pdf) const override;
    __bidevice__ virtual Float Pdf(const Interaction &ref, const vec3f &wi) const override;
    __bidevice__ virtual Float Pdf(const Interaction &) const override{ return 1 / Area(); }
};

class Disk : public Shape{
    public:
    const Float height, radius, innerRadius, phiMax;
    __bidevice__ Disk(const Transform toWorld, Float height, Float radius, 
                      Float innerRadius, Float phiMax)
        : Shape(toWorld), height(height), radius(radius), innerRadius(innerRadius),
    phiMax(Radians(Clamp(phiMax, 0, 360))) {type = ShapeType::DISK;}
    
    __bidevice__ virtual bool Intersect(const Ray &ray, Float *tHit,
                                        SurfaceInteraction *isect) const override;
    
    __bidevice__ virtual Bounds3f GetBounds() const override;
    __bidevice__ virtual Float Area() const override;
    
    __bidevice__ virtual Interaction Sample(const Point2f &u, Float *pdf) const override;
    __bidevice__ virtual Interaction Sample(const Interaction &ref, const Point2f &u,
                                            Float *pdf) const override;
    __bidevice__ virtual Float Pdf(const Interaction &ref, const vec3f &wi) const override;
    __bidevice__ virtual Float Pdf(const Interaction &) const override{ return 1 / Area(); }
};

//NOTE: Unit rectangle in XY plane, also its intersect method must be final
//      so that cuda can correctly define Box stack requirements
class Rectangle : public Shape{
    public:
    Float sizex, sizey;
    __bidevice__ Rectangle(const Transform &toWorld, Float sizex, Float sizey)
        : Shape(toWorld), sizex(Max(0, sizex)), sizey(Max(0, sizey))
    {type = ShapeType::RECTANGLE;}
    
    __bidevice__ virtual bool Intersect(const Ray &ray, Float *tHit,
                                        SurfaceInteraction *isect) const override final;
    __bidevice__ virtual Bounds3f GetBounds() const override;
    __bidevice__ virtual Float Area() const override;
    __bidevice__ virtual Interaction Sample(const Point2f &u, Float *pdf) const override;
    __bidevice__ virtual Interaction Sample(const Interaction &ref, const Point2f &u,
                                            Float *pdf) const override;
    __bidevice__ virtual Float Pdf(const Interaction &) const override{ return 1 / Area(); }
    __bidevice__ virtual Float Pdf(const Interaction &ref, const vec3f &wi) const override;
};

class Box : public Shape{
    public:
    Rectangle **rects;
    Float sizex, sizey, sizez;
    __bidevice__ Box(const Transform &toWorld, Float sizex, Float sizey, Float sizez);
    
    __bidevice__ virtual bool Intersect(const Ray &ray, Float *tHit,
                                        SurfaceInteraction *isect) const override;
    __bidevice__ virtual Bounds3f GetBounds() const override;
    __bidevice__ virtual Float Area() const override;
    __bidevice__ virtual Interaction Sample(const Point2f &u, Float *pdf) const override{
        UMETHOD();
        *pdf = 0;
        return Interaction();
    }
    
    __bidevice__ virtual Interaction Sample(const Interaction &ref, const Point2f &u,
                                            Float *pdf) const override
    {
        UMETHOD();
        *pdf = 0;
        return Interaction();
    }
    
    __bidevice__ virtual Float Pdf(const Interaction &) const override{ return 1 / Area(); }
    __bidevice__ virtual Float Pdf(const Interaction &ref, const vec3f &wi) const override{
        UMETHOD();
        return 0.f;
    }
};

typedef struct{
    Point3f *p;
    Normal3f *n;
    vec3f *s;
    Point2f *uv;
    int *indices;
    int nTriangles, nVertices;
    Transform toWorld;
}ParsedMesh;

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
    __bidevice__ Mesh(const Transform &toWorld, ParsedMesh *pMesh, int copy=1);
    
    __bidevice__ void Set(const Transform &toWorld, int nTris, int *_indices,
                          int nVerts, Point3f *P, vec3f *S, Normal3f *N, Point2f *UV);
    
    __bidevice__ virtual bool Intersect(const Ray &ray, Float *tHit,
                                        SurfaceInteraction *isect) const override;
    
    __bidevice__ virtual Bounds3f GetBounds() const override;
    
    __bidevice__ virtual Float Area() const override{
        UMETHOD();
        return 1;
    }
    
    __bidevice__ virtual Interaction Sample(const Point2f &u, Float *pdf) const{
        UMETHOD();
        *pdf = 0;
        return Interaction();
    }
    
    __bidevice__ virtual Interaction Sample(const Interaction &ref, const Point2f &u,
                                            Float *pdf) const override
    {
        UMETHOD();
        *pdf = 0;
        return Interaction();
    }
    
    __bidevice__ virtual Float Pdf(const Interaction &) const override{ return 1 / Area(); }
    __bidevice__ virtual Float Pdf(const Interaction &ref, const vec3f &wi) const override{
        UMETHOD();
        return 0.f;
    }
    
    private:
    __bidevice__ bool IntersectMeshNode(Node *node, const Ray &r, 
                                        SurfaceInteraction *, Float *) const;
    __bidevice__ bool IntersectTriangle(const Ray &r, SurfaceInteraction * isect,
                                        int triNum, Float *tHit) const;
    __bidevice__ bool IntersectTriangleLow(const Ray &r, SurfaceInteraction * isect,
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
        Mesh *mesh = (Mesh *)shape;
        printf("Mesh [ triangles: %d , vertices: %d]", mesh->nTriangles,
               mesh->nVertices);
    }else if(shape->type == ShapeType::RECTANGLE){
        Rectangle *rect = (Rectangle *)shape;
        printf("Rectangle [ %g x %g ]", rect->sizex, rect->sizey);
    }else{
        printf("None");
    }
}

__host__ void WrapMesh(Mesh *mesh);
__host__ bool LoadObjData(const char *obj, ParsedMesh **data);
__host__ ParsedMesh *ParsedMeshFromData(const Transform &toWorld, int nTris, int *_indices,
                                        int nVerts, Point3f *P, vec3f *S, 
                                        Normal3f *N, Point2f *UV, int copy=0);