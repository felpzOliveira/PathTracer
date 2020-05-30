#include <light.h>
#include <primitive.h>
#include <geometry.h>

__bidevice__ Spectrum Light::Distant_L(const Interaction &intr, const vec3f &w) const{
    return Spectrum(0);
}

__bidevice__ void Light::Distant_Prepare(Aggregator *scene){
    Bounds3f bound = scene->WorldBound();
    bound.BoundingSphere(&worldCenter, &worldRadius);
    printf("Light::Distant world radius: %g\n", worldRadius);
}

__bidevice__ Spectrum Light::Distant_Le(const RayDifferential &r) const{
    return Spectrum(0);
}

__bidevice__ Spectrum Light::Distant_Sample_Li(const Interaction &ref, const Point2f &u,
                                               vec3f *wi, Float *pdf, 
                                               VisibilityTester *vis) const
{
    *wi = wLight;
    *pdf = 1;
    Point3f P = ref.p + wLight * 2 * worldRadius;
    *vis = VisibilityTester(ref, Interaction(P, ref.time));
    return Lemit;
}

__bidevice__ Float Light::Distant_Pdf_Li(const Interaction &ref, const vec3f &wi) const{
    return 0;
}

__bidevice__ Spectrum Light::Distant_Sample_Le(const Point2f &u1, const Point2f &u2,
                                               Float time, Ray *ray, Normal3f *nLight,
                                               Float *pdfPos, Float *pdfDir) const
{
    vec3f v1, v2;
    CoordinateSystem(wLight, &v1, &v2);
    Point2f cd = ConcentricSampleDisk(u1);
    Point3f pDisk = worldCenter + worldRadius * (cd.x * v1 + cd.y * v2);
    *ray = Ray(pDisk + worldRadius * wLight, -wLight, Infinity, time);
    *nLight = (Normal3f)ray->d;
    *pdfPos = 1 / (Pi * worldRadius * worldRadius);
    *pdfDir = 1;
    return Lemit;
}

__bidevice__ void Light::Distant_Pdf_Le(const Ray &ray, const Normal3f &n, 
                                        Float *pdfPos, Float *pdfDir) const
{
    *pdfPos = 1 / (Pi * worldRadius * worldRadius);
    *pdfDir = 0;
}