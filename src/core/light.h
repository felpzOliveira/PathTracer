#pragma once
#include <transform.h>
#include <interaction.h>
#include <texture.h>

class Aggregator;
class Distribution2D;
/*
* Cuda is being a little c**t again with class inheritance
* so I'm gonna have to *again* wrap all light types in a single type.
*/

enum class LightFlags : int{
    DeltaPosition = 1,
    DeltaDirection = 2,
    Area = 4,
    Infinite = 8
};

enum LightType{
    DiffuseArea = 1,
    Distant,
    Infinite,
};

typedef struct{
    LightType type;
    int shapeId; // id in light vector to find shape for this light
    int flags;
    Transform toWorld;
    
    vec3f wLight; // direction light vector
    Spectrum L; // spectrum
    
    MipMap<Spectrum> *Ls;
    Distribution2D *dist;
}LightDesc;

inline __bidevice__ bool IsDeltaLight(int flags){
    return flags & (int)LightFlags::DeltaPosition || flags & (int)LightFlags::DeltaDirection;
}

class VisibilityTester{
    public:
    Interaction p0, p1;
    
    __bidevice__ VisibilityTester() {}
    __bidevice__ VisibilityTester(const Interaction &p0, const Interaction &p1): p0(p0), p1(p1) {}
    __bidevice__ const Interaction &P0() const { return p0; }
    __bidevice__ const Interaction &P1() const { return p1; }
    __bidevice__ bool Unoccluded(const Aggregator *scene) const;
    __bidevice__ Spectrum Tr(const Aggregator *scene) const;
};

class Light{
    public:
    // Generic for any light
    const Transform LightToWorld, WorldToLight;
    int flags;
    LightType type;
    const int nSamples;
    
    Spectrum Lemit;
    
    // Area light properties
    Shape *shape;
    bool twoSided;
    Float area;
    
    // Distant light
    vec3f wLight; // which direction this thing is comming from
    Point3f worldCenter;
    Float worldRadius;
    
    // Infinite light
    MipMap<Spectrum> *Lmap;
    Distribution2D *distribution;
    
    __bidevice__ Light(const Transform &LightToWorld, int flags, int nSamples=1);
    
    __bidevice__ void Init_DiffuseArea(const Spectrum &Le, Shape *shape, 
                                       bool twoSided = false);
    
    __bidevice__ void Init_Distant(const Spectrum &Le, const vec3f &w);
    
    /*
* NOTE:
* In order to use the EXR we have you need to make sure the LightToWorld matrix
* transform has a component RotateX(-(95+)) this will make the y < 0 be black
* and y > 0 receive the correct light. Those images are half black so they must
* face positions where ground is expected, otherwise you will get partial black sky.
*/
    __bidevice__ void Init_Infinite(MipMap<Spectrum> *lightMap, Distribution2D *distr);
    
    
    __bidevice__ Spectrum Le(const RayDifferential &r) const;
    __bidevice__ Spectrum L(const Interaction &intr, const vec3f &w) const;
    
    __bidevice__ Spectrum Sample_Le(const Point2f &u1, const Point2f &u2, Float time,
                                    Ray *ray, Normal3f *nLight, Float *pdfPos,
                                    Float *pdfDir) const;
    
    __bidevice__ void Pdf_Le(const Ray &, const Normal3f &, Float *pdfPos,
                             Float *pdfDir) const;
    __bidevice__ Float Pdf_Li(const Interaction &, const vec3f &) const;
    
    __bidevice__ Spectrum Sample_Li(const Interaction &ref, const Point2f &u, 
                                    vec3f *wo, Float *pdf, VisibilityTester *vis) const;
    
    __bidevice__ void Prepare(Aggregator *scene);
    
    private:
    __bidevice__ Spectrum DiffuseArea_L(const Interaction &intr, const vec3f &w) const;
    __bidevice__ Spectrum DiffuseArea_Sample_Li(const Interaction &ref, const Point2f &u,
                                                vec3f *wi, Float *pdf, 
                                                VisibilityTester *vis) const;
    __bidevice__ Float DiffuseArea_Pdf_Li(const Interaction &ref, const vec3f &wi) const;
    __bidevice__ Spectrum DiffuseArea_Le(const RayDifferential &r) const;
    __bidevice__ Spectrum DiffuseArea_Sample_Le(const Point2f &u1, const Point2f &u2,
                                                Float time, Ray *ray, Normal3f *nLight,
                                                Float *pdfPos, Float *pdfDir) const;
    __bidevice__ void DiffuseArea_Pdf_Le(const Ray &ray, const Normal3f &n,
                                         Float *pdfPos, Float *pdfDir) const;
    __bidevice__ void DiffuseArea_Prepare(Aggregator *scene);
    
    __bidevice__ Spectrum Distant_L(const Interaction &intr, const vec3f &w) const;
    __bidevice__ Spectrum Distant_Sample_Li(const Interaction &ref, const Point2f &u,
                                            vec3f *wi, Float *pdf, 
                                            VisibilityTester *vis) const;
    __bidevice__ Float Distant_Pdf_Li(const Interaction &ref, const vec3f &wi) const;
    __bidevice__ Spectrum Distant_Le(const RayDifferential &r) const;
    __bidevice__ Spectrum Distant_Sample_Le(const Point2f &u1, const Point2f &u2,
                                            Float time, Ray *ray, Normal3f *nLight,
                                            Float *pdfPos, Float *pdfDir) const;
    __bidevice__ void Distant_Pdf_Le(const Ray &ray, const Normal3f &n,
                                     Float *pdfPos, Float *pdfDir) const;
    __bidevice__ void Distant_Prepare(Aggregator *scene);
    
    __bidevice__ Spectrum Infinite_L(const Interaction &intr, const vec3f &w) const;
    __bidevice__ Spectrum Infinite_Sample_Li(const Interaction &ref, const Point2f &u,
                                             vec3f *wi, Float *pdf, 
                                             VisibilityTester *vis) const;
    __bidevice__ Float Infinite_Pdf_Li(const Interaction &ref, const vec3f &wi) const;
    __bidevice__ Spectrum Infinite_Le(const RayDifferential &r) const;
    __bidevice__ Spectrum Infinite_Sample_Le(const Point2f &u1, const Point2f &u2,
                                             Float time, Ray *ray, Normal3f *nLight,
                                             Float *pdfPos, Float *pdfDir) const;
    __bidevice__ void Infinite_Pdf_Le(const Ray &ray, const Normal3f &n,
                                      Float *pdfPos, Float *pdfDir) const;
    __bidevice__ void Infinite_Prepare(Aggregator *scene);
};
