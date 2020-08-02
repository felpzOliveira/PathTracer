#pragma once

#include <shape.h>

// TODO: I have no idea how we can sample this thing. Can we like sample a sphere 
//       and march into the SDF to find a point? Have no idea how to handle area
class SDF{
    public:
    static __bidevice__ Float Sphere(vec3f p, Float radius);
    static __bidevice__ Float Box(vec3f p, vec3f boxLength);
    static __bidevice__ Float BoxRound(vec3f p, vec3f boxLength, Float radius);
    static __bidevice__ Float Ellipse(vec3f p, vec3f center, vec3f axis);
    static __bidevice__ Float Rhombus(vec3f p, vec2f axis, Float height, Float corner);
    static __bidevice__ Float Triangle(vec3f p, vec3f a, vec3f b, vec3f c);
    static __bidevice__ Float Capsule(vec3f p, vec3f a, vec3f b, Float r);
    
    // NOTE: The following are functions used for images and not privimitives
    static __bidevice__ Float OrigamiBoat(vec3f p, vec3f boxHalfLen, int *shapeId);
    static __bidevice__ Float OrigamiDragon(vec3f p, vec3f boxHalfLen, int *shapeId);
    static __bidevice__ Float OrigamiWhale(vec3f p, vec3f boxHalfLen, int *shapeId);
    static __bidevice__ Float OrigamiBird(vec3f p, vec3f boxHalfLen, int *shapeId);
    static __bidevice__ Float Terrain(vec3f p, vec3f boxHalfLen, int *shapeId);
};

class ProceduralMath{
    public:
    static __bidevice__ Float Hash1(Float n);
    static __bidevice__ Float Hash1(const vec2f &p);
    static __bidevice__ vec3f Noise32d(const vec2f &p);
};

class ProceduralShape: public Shape{
    public:
    
    __bidevice__ ProceduralShape(const Transform &toWorld) : Shape(toWorld, false){}
    __bidevice__ virtual Float MapP(Point3f p) const;
    __bidevice__ virtual Float Map(Point3f p, Point2f *uv, int *shapeId) const = 0;
    
    /* If shapes can solve their Normals analytically do it and return true, otherwise false 
( we use derivatives to find the normal in that case ), can also set specific uv 
mappings for shape to be used later in shading */
    __bidevice__ virtual bool RefineHit(Point3f *pHit, Point2f *uv, int *shapeId, 
                                        Normal3f *n) const;
    
    /* Shapes must provide their bounds in local space for faster intersection routine */
    __bidevice__ virtual Bounds3f GetLocalBounds() const = 0;
    
    /* We use finite differences for normals in case shapes cannot compute normals 
analytically, however shapes may override this if they feel there is a better approach
for their equations */
    __bidevice__ virtual Normal3f DerivativeNormal(Point3f p, Float t) const;
    
    /* Perform Ray March to hit this shape */
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

class ProceduralBox : public ProceduralShape{
    public:
    vec3f length;
    __bidevice__ ProceduralBox(const Transform &toWorld, const vec3f &length);
    __bidevice__ virtual Bounds3f GetBounds() const override;
    __bidevice__ virtual Bounds3f GetLocalBounds() const override;
    __bidevice__ virtual Float Map(Point3f p, Point2f *uv, int *shapeId) const override;
    __bidevice__ virtual bool RefineHit(Point3f *pHit, Point2f *uv, 
                                        int *shapeId, Normal3f *n) const override;
};

class ProceduralSphere: public ProceduralShape{
    public:
    Float radius;
    __bidevice__ ProceduralSphere(const Transform &toWorld, Float radius);
    __bidevice__ virtual Bounds3f GetBounds() const override;
    __bidevice__ virtual Bounds3f GetLocalBounds() const override;
    __bidevice__ virtual Float Map(Point3f p, Point2f *uv, int *shapeId) const override;
    __bidevice__ virtual bool RefineHit(Point3f *pHit, Point2f *uv, 
                                        int *shapeId, Normal3f *n) const override;
};

// Generic component for toying
class ProceduralComponent: public ProceduralShape{
    public:
    int id;
    Bounds3f bounds;
    __bidevice__ ProceduralComponent(const Transform &toWorld, const Bounds3f &bounds, int id);
    __bidevice__ virtual Bounds3f GetBounds() const override;
    __bidevice__ virtual Bounds3f GetLocalBounds() const override;
    __bidevice__ virtual Float Map(Point3f p, Point2f *uv, int *shapeId) const override;
};
