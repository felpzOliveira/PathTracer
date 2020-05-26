#include <light.h>
#include <primitive.h>
#include <geometry.h>

__bidevice__ bool VisibilityTester::Unoccluded(const Aggregator *scene) const{
    SurfaceInteraction tmp;
    return !scene->Intersect(p0.SpawnRayTo(p1), &tmp);
}

__bidevice__ Spectrum VisibilityTester::Tr(const Aggregator *scene) const{
    Ray ray(p0.SpawnRayTo(p1));
    Spectrum Tr(1.f);
    int debug = 0;
    int it = 0;
    int warned = 0;
    while(true){
        SurfaceInteraction isect;
        bool hitSurface = scene->Intersect(ray, &isect);
        if(hitSurface && isect.primitive->GetMaterial() != nullptr) 
            return Spectrum(0.0f);
        
        if(ray.medium){
            Tr *= ray.medium->Tr(ray);
        }
        
        if(!hitSurface || isect.primitive->IsEmissive()) break;
        ray = isect.SpawnRayTo(p1);
        if(debug){
            if(it++ > WARN_BOUNCE_COUNT){
                if(!warned){
                    printf("Warning: Dangerously high bounce count (%d) in Aggregator::Tr ( " v3fA(Tr) " )\n",
                           it, v3aA(Tr));
                    warned = 1;
                }
            }
        }
    }
    
    return Tr;
}

__bidevice__ DiffuseAreaLight::DiffuseAreaLight(const Transform &LightToWorld, 
                                                const Spectrum &Le, int nSamples, 
                                                Shape *shape, bool twoSided)
:flags((int)LightFlags::Area), nSamples(Max(1, nSamples)), LightToWorld(LightToWorld),
WorldToLight(Inverse(LightToWorld)) , Lemit(Le), shape(shape),
twoSided(twoSided), area(shape->Area()) 
{
    printf(v3fA(Lemit) "\n", v3aA(Lemit));
}

__bidevice__ Spectrum DiffuseAreaLight::Sample_Li(const Interaction &ref, const Point2f &u,
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

__bidevice__ Float DiffuseAreaLight::Pdf_Li(const Interaction &ref, const vec3f &wi) const{
    return shape->Pdf(ref, wi);
}

__bidevice__ Spectrum DiffuseAreaLight::Sample_Le(const Point2f &u1, const Point2f &u2,
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

__bidevice__ void DiffuseAreaLight::Pdf_Le(const Ray &ray, const Normal3f &n, 
                                           Float *pdfPos, Float *pdfDir) const
{
    Interaction it(ray.o, n, vec3f(), ToVec3(n), ray.time);
    *pdfPos = shape->Pdf(it);
    *pdfDir = twoSided ? (.5 * CosineHemispherePdf(AbsDot(n, ray.d)))
        : CosineHemispherePdf(Dot(n, ray.d));
}
