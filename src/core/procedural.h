#pragma once

#include <shape.h>

class SDF{
    public:
    static __bidevice__ Float Sphere(vec3f p, Float radius);
    static __bidevice__ Float Box(vec3f p, vec3f boxLength);
    static __bidevice__ Float Rhombus(vec3f p, vec2f axis, Float height, Float corner);
};

class ProceduralShape: public Shape{
    public:
    
    __bidevice__ ProceduralShape(const Transform &toWorld) : Shape(toWorld, false){}
    __bidevice__ virtual Float Map(Point3f p) const = 0;
    __bidevice__ virtual Bounds3f GetLocalBounds() const = 0;
    __bidevice__ virtual Normal3f ComputeNormal(Point3f p, Float t) const;
    
    __bidevice__ virtual bool Intersect(const Ray &ray, Float *tHit,
                                        SurfaceInteraction *isect) const override;
    
    
    /* I think we can only solve these for closed formulae shapes */
    __bidevice__ virtual Float Area() const override{ UMETHOD(); return 1; }
    __bidevice__ virtual Interaction Sample(const Point2f &u, Float *pdf) const override{
        UMETHOD();
        return Interaction();
    }
    
    __bidevice__ virtual Interaction Sample(const Interaction &ref, const Point2f &u,
                                            Float *pdf) const override
    {
        UMETHOD();
        return Interaction();
    }
};

class ProceduralSphere: public ProceduralShape{
    public:
    Float radius;
    __bidevice__ ProceduralSphere(const Transform &toWorld, Float radius);
    __bidevice__ virtual Bounds3f GetBounds() const override;
    __bidevice__ virtual Bounds3f GetLocalBounds() const override;
    __bidevice__ virtual Float Map(Point3f p) const override;
};