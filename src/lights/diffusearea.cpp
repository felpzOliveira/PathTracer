#include <light.h>
#include <primitive.h>
#include <geometry.h>

__bidevice__ Spectrum Light::DiffuseArea_L(const Interaction &intr, const vec3f &w) const{
    return (twoSided || Dot(ToVec3(intr.n), w) > 0) ? Lemit : Spectrum(0.f);
}

__bidevice__ void Light::DiffuseArea_Prepare(Aggregator *scene){}

__bidevice__ Spectrum Light::DiffuseArea_Le(const RayDifferential &r) const{
    return Spectrum(0);
}

__bidevice__ Spectrum Light::DiffuseArea_Sample_Li(const Interaction &ref, const Point2f &u,
                                                   vec3f *wi, Float *pdf, 
                                                   VisibilityTester *vis) const
{
    Interaction pShape = shape->Sample(ref, u, pdf);
    if(IsZero(*pdf) || IsZero((pShape.p - ref.p).LengthSquared())){
        *pdf = 0;
        return 0.f;
    }
    
    *wi = Normalize(pShape.p - ref.p);
    *vis = VisibilityTester(ref, pShape);
    return L(pShape, -*wi);
}

__bidevice__ Float Light::DiffuseArea_Pdf_Li(const Interaction &ref, const vec3f &wi) const{
    return shape->Pdf(ref, wi);
}

__bidevice__ Spectrum Light::DiffuseArea_Sample_Le(const Point2f &u1, const Point2f &u2,
                                                   Float time, Ray *ray, Normal3f *nLight,
                                                   Float *pdfPos, Float *pdfDir) const
{
    Interaction pShape = shape->Sample(u1, pdfPos);
    *nLight = pShape.n;
    
    vec3f w;
    if(twoSided){
        Point2f u = u2;
        if(u[0] < .5){
            u[0] = Min(u[0] * 2, OneMinusEpsilon);
            w = CosineSampleHemisphere(u);
        }else{
            u[0] = Min((u[0] - .5f) * 2, OneMinusEpsilon);
            w = CosineSampleHemisphere(u);
            w.z *= -1;
        }
        
        *pdfDir = 0.5f * CosineHemispherePdf(Absf(w.z));
    }else{
        w = CosineSampleHemisphere(u2);
        *pdfDir = CosineHemispherePdf(w.z);
    }
    
    vec3f v1, v2, n(ToVec3(pShape.n));
    CoordinateSystem(n, &v1, &v2);
    w = w.x * v1 + w.y * v2 + w.z * n;
    *ray = pShape.SpawnRay(w);
    return L(pShape, w);
}

__bidevice__ void Light::DiffuseArea_Pdf_Le(const Ray &ray, const Normal3f &n, 
                                            Float *pdfPos, Float *pdfDir) const
{
    Interaction it(ray.o, n, vec3f(), ToVec3(n), ray.time);
    *pdfPos = shape->Pdf(it);
    *pdfDir = twoSided ? (.5 * CosineHemispherePdf(AbsDot(n, ray.d)))
        : CosineHemispherePdf(Dot(n, ray.d));
}
