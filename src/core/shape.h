#pragma once
#include <geometry.h>
#include <transform.h>
#include <cutil.h>
#include <interaction.h>

enum ShapeType{
    SPHERE
};

class Shape{
    public:
    ShapeType type;
    Transform ObjectToWorld, WorldToObject;
    __bidevice__ Shape(const Transform &toWorld) :
    ObjectToWorld(toWorld), WorldToObject(Inverse(toWorld))
    {}
    
    __bidevice__ virtual Bounds3f GetBounds() const = 0;
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

inline __bidevice__ void PrintShape(Shape *shape){
    if(shape->type == ShapeType::SPHERE){
        Sphere *sphere = (Sphere *)shape;
        Point3f center = (shape->ObjectToWorld)(Point3f(0,0,0));
        printf("Sphere [ " __vec3_strfmtA(center) " , radius: %g ]", 
               __vec3_argsA(center), sphere->radius);
    }else{
        printf("None");
    }
}