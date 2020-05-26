#pragma once
#include <transform.h>
#include <interaction.h>

class Aggregator;

enum class LightFlags : int {
    DeltaPosition = 1,
    DeltaDirection = 2,
    Area = 4,
    Infinite = 8
};

inline __bidevice__ bool IsDeltaLight(int flags) {
    return flags & (int)LightFlags::DeltaPosition || flags & (int)LightFlags::DeltaDirection;
}

class VisibilityTester {
    public:
    Interaction p0, p1;
    
    __bidevice__ VisibilityTester() {}
    __bidevice__ VisibilityTester(const Interaction &p0, const Interaction &p1): p0(p0), p1(p1) {}
    __bidevice__ const Interaction &P0() const { return p0; }
    __bidevice__ const Interaction &P1() const { return p1; }
    __bidevice__ bool Unoccluded(const Aggregator *scene) const;
    __bidevice__ Spectrum Tr(const Aggregator *scene) const;
};

class DiffuseAreaLight{
    public:
    const Spectrum Lemit;
    Shape *shape;
    const bool twoSided;
    const Float area;
    const int flags;
    const int nSamples;
    const Transform LightToWorld, WorldToLight;
    
    __bidevice__ DiffuseAreaLight(const Transform &LightToWorld, const Spectrum &Le,
                                  int nSamples, Shape *shape, bool twoSided = false);
    
    __bidevice__ Spectrum Le(const RayDifferential &r) const{ return Spectrum(0.f); }
    
    __bidevice__ Spectrum L(const Interaction &intr, const vec3f &w) const{
        return (twoSided || Dot(ToVec3(intr.n), w) > 0) ? Lemit : Spectrum(0.f);
    }
    
    __bidevice__ Spectrum Sample_Li(const Interaction &ref, const Point2f &u, 
                                    vec3f *wo, Float *pdf, 
                                    VisibilityTester *vis) const;
    
    __bidevice__ Float Pdf_Li(const Interaction &, const vec3f &) const;
    
    __bidevice__ Spectrum Sample_Le(const Point2f &u1, const Point2f &u2, Float time,
                                    Ray *ray, Normal3f *nLight, Float *pdfPos,
                                    Float *pdfDir) const;
    
    __bidevice__ void Pdf_Le(const Ray &, const Normal3f &, Float *pdfPos,
                             Float *pdfDir) const;
    
};