#pragma once
#include <geometry.h>
#include <transform.h>
#include <cutil.h>
#include <interaction.h>
#include <util.h>

enum ShapeType{
    SPHERE, RECTANGLE, DISK, BOX, MESH, PARTICLECLOUD,
    SPHERE_PROCEDURAL, BOX_PROCEDURAL, COMPONENT_PROCEDURAL,
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

extern int BVH_MAX_DEPTH;
__host__ Node *CreateBVH(PrimitiveHandle *handles, int n, int depth, 
                         int max_depth, int *totalNodes, int *maxNodes);

class Shape{
    public:
    ShapeType type;
    Transform ObjectToWorld, WorldToObject;
    bool reverseOrientation;
    bool transformSwapsHandedness;
    
    __bidevice__ Shape(const Transform &toWorld, bool reverseOrientation=false) :
    ObjectToWorld(toWorld), WorldToObject(Inverse(toWorld)),
    reverseOrientation(reverseOrientation),
    transformSwapsHandedness(ObjectToWorld.SwapsHandedness())
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
                        Float zMin, Float zMax, Float phiMax,
                        bool reverseOrientation) : 
    Shape(toWorld, reverseOrientation), radius(radius),
    zMin(Clamp(Min(zMin, zMax), -radius, radius)),
    zMax(Clamp(Max(zMin, zMax), -radius, radius)),
    thetaMin(std::acos(Clamp(Min(zMin, zMax) / radius, -1, 1))),
    thetaMax(std::acos(Clamp(Max(zMin, zMax) / radius, -1, 1))),
    phiMax(Radians(Clamp(phiMax, 0, 360))){type = ShapeType::SPHERE;}
    
    __bidevice__ Sphere(const Transform &toWorld, Float radius, bool reverseOrientation) :
    Shape(toWorld, reverseOrientation), radius(radius),
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
                      Float innerRadius, Float phiMax, bool reverseOrientation)
        : Shape(toWorld, reverseOrientation), height(height), radius(radius), 
    innerRadius(innerRadius),phiMax(Radians(Clamp(phiMax, 0, 360))) {type = ShapeType::DISK;}
    
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
    __bidevice__ Rectangle(const Transform &toWorld, Float sizex, Float sizey,
                           bool reverseOrientation)
        : Shape(toWorld, reverseOrientation), sizex(Max(0, sizex)), sizey(Max(0, sizey))
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
    __bidevice__ Box(const Transform &toWorld, Float sizex, Float sizey, Float sizez,
                     bool reverseOrientation);
    
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
    Point3i *indices;
    int nTriangles, nVertices;
    int nUvs, nNormals;
    Transform toWorld;
}ParsedMesh;

/*
* For indices (for triangle i):
*    x - vertex indices : Point3f  p = p[indices[3 * i + ...].x]
*    y - normal indices : Normal3f n = n[indices[3 * i + ...].y]
*    z - uv indices     : Point2f uv = uv[indices[3 * i + ..].z]
*/

//Meshes are in world space?
class Mesh: public Shape{
    public:
    int nTriangles, nVertices;
    int nUvs, nNormals;
    Point3i *indices;
    Point3f *p;
    Normal3f *n;
    vec3f *s;
    Point2f *uv;
    PrimitiveHandle *handles;
    Node *bvh;
    
    __bidevice__ Mesh() : Shape(Transform()){type = ShapeType::MESH;}
    __bidevice__ Mesh(const Transform &toWorld, ParsedMesh *pMesh, int copy=0);
    
    __bidevice__ virtual bool Intersect(const Ray &ray, Float *tHit,
                                        SurfaceInteraction *isect) const override;
    
    __bidevice__ virtual Bounds3f GetBounds() const override;
    
    __bidevice__ Float TriangleArea(int triNum) const;
    __bidevice__ virtual Float Area() const override;
    __bidevice__ virtual Interaction Sample(const Interaction &ref, const Point2f &u,
                                            Float *pdf) const override;
    __bidevice__ virtual Float Pdf(const Interaction &ref, const vec3f &wi) const override;
    
    __bidevice__ virtual Interaction Sample(const Point2f &u, Float *pdf) const{
        UMETHOD();
        *pdf = 0;
        return Interaction();
    }
    
    __bidevice__ virtual Float Pdf(const Interaction &) const override{ return 1 / Area(); }
    
    private:
    __bidevice__ bool IntersectMeshNode(Node *node, const Ray &r, 
                                        SurfaceInteraction *, Float *) const;
    __bidevice__ bool IntersectTriangle(const Ray &r, SurfaceInteraction * isect,
                                        int triNum, Float *tHit) const;
    __bidevice__ bool IntersectTriangleLow(const Ray &r, SurfaceInteraction * isect,
                                           int triNum, Float *tHit) const;
    
    __bidevice__ void GetUVs(Point2f uv[3], int nTri) const;
    __bidevice__ void GetNormals(Normal3f n[3], int nTri) const;
};

class ParticleCloud : public Shape{
    public:
    vec3f *positions;
    int nParticles;
    Float radius;
    PrimitiveHandle *handles;
    Node *bvh;
    
    __bidevice__ ParticleCloud() : Shape(Transform()){ type = ShapeType::PARTICLECLOUD; }
    __bidevice__ ParticleCloud(vec3f *pos, int n, Float scale);
    
    __bidevice__ virtual bool Intersect(const Ray &ray, Float *tHit,
                                        SurfaceInteraction *isect) const override;
    
    __bidevice__ virtual Bounds3f GetBounds() const override;
    
    __bidevice__ Float ParticleArea(int triNum) const;
    __bidevice__ virtual Float Area() const override;
    
    __bidevice__ virtual Interaction Sample(const Interaction &ref, const Point2f &u,
                                            Float *pdf) const override;
    
    __bidevice__ Interaction SampleIndex(const Point2f &u, Float *pdf, 
                                         int index) const;
    
    __bidevice__ virtual Float Pdf(const Interaction &ref, const vec3f &wi) const override{
        UMETHOD();
        return 0;
    }
    
    __bidevice__ virtual Interaction Sample(const Point2f &u, Float *pdf) const{
        UMETHOD();
        *pdf = 0;
        return Interaction();
    }
    
    __bidevice__ virtual Float Pdf(const Interaction &) const override{ return 1 / Area(); }
    private:
    __bidevice__ bool IntersectParticleCloudNode(Node *node, const Ray &r, 
                                                 SurfaceInteraction *, Float *) const;
    __bidevice__ bool IntersectSphere(const Ray &r, SurfaceInteraction * isect,
                                      int partNum, Float *tHit) const;
};

inline __bidevice__ void PrintShape(Shape *shape){
    if(shape->type == ShapeType::SPHERE){
        Sphere *sphere = (Sphere *)shape;
        Point3f center = (shape->ObjectToWorld)(Point3f(0,0,0));
        printf("Sphere [ " __vec3_strfmtA(center) " , radius: %g ]\n", 
               __vec3_argsA(center), sphere->radius);
    }else if(shape->type == ShapeType::MESH){
        Mesh *mesh = (Mesh *)shape;
        printf("Mesh [ triangles: %d , vertices: %d]\n", mesh->nTriangles,
               mesh->nVertices);
    }else if(shape->type == ShapeType::RECTANGLE){
        Rectangle *rect = (Rectangle *)shape;
        printf("Rectangle [ %g x %g ]\n", rect->sizex, rect->sizey);
    }else if(shape->type == ShapeType::DISK){
        Disk *disk = (Disk *)shape;
        printf("Disk [ radius: %g , innerRadius: %g , phiMax: %g ]\n",
               disk->radius, disk->innerRadius, disk->phiMax);
    }else{
        printf("None\n");
    }
}

__host__ void WrapMesh(Mesh *mesh);
__host__ void WrapParticleCloud(ParticleCloud *pCloud);
__host__ bool LoadObjData(const char *obj, ParsedMesh **data);

__host__ ParsedMesh *ParsedMeshFromData(int nTris, Point3i *indices, int nVerts, Point3f *P);
__host__ ParsedMesh *ParsedMeshFromData(const Transform &toWorld, int nTris, Point3i *indices,
                                        int nVerts, Point3f *P);