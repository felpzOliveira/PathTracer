#include <light.h>
#include <geometry.h>
#include <sampling.h>
#include <primitive.h>
#include <medium.h>

__host__ Distribution1D *CreateDistribution1D(Float *f, int n){
    Distribution1D *dist = cudaAllocateVx(Distribution1D, 1);
    dist->cdf = cudaAllocateVx(Float, n+1);
    dist->func = f;
    dist->size = n;
    dist->cdf[0] = 0;
    for(int i = 1; i < n + 1; ++i) dist->cdf[i] = dist->cdf[i - 1] + dist->func[i - 1] / n;
    dist->funcInt = dist->cdf[n];
    
    if(dist->funcInt == 0){
        for(int i = 1; i < n + 1; ++i) dist->cdf[i] = Float(i) / Float(n);
    }else{
        for(int i = 1; i < n + 1; ++i) dist->cdf[i] /= dist->funcInt;
    }
    return dist;
}

__host__ Distribution2D *CreateDistribution2D(Float *func, int nu, int nv){
    Distribution2D *dist = cudaAllocateVx(Distribution2D, 1);
    dist->pConditionalV = cudaAllocateVx(Distribution1D*, nv);
    
    for(int v = 0; v < nv; ++v){
        dist->pConditionalV[v] = CreateDistribution1D(&func[v * nu], nu);
    }
    
    Float *marginalFunc = cudaAllocateVx(Float, nv);
    for(int v = 0; v < nv; ++v)
        marginalFunc[v] = dist->pConditionalV[v]->funcInt;
    
    dist->pMarginal = CreateDistribution1D(&marginalFunc[0], nv);
    
    return dist;
}

__bidevice__ void Light::Init_Infinite(MipMap<Spectrum> *lightMap, Distribution2D *distr){
    flags = (int)LightFlags::Infinite;
    type = LightType::Infinite;
    Lmap = lightMap;
    distribution = distr;
}

__bidevice__ Spectrum Light::Infinite_L(const Interaction &intr, const vec3f &w) const{
    return Spectrum(0);
}

__bidevice__ void Light::Infinite_Prepare(Aggregator *scene){
    Bounds3f bound = scene->WorldBound();
    bound.BoundingSphere(&worldCenter, &worldRadius);
    printf("Light::Infinite world radius: %g\n", worldRadius);
}

__bidevice__ Spectrum Light::Infinite_Le(const RayDifferential &ray) const{
    vec3f w = Normalize(WorldToLight(ray.d));
    Point2f st(SphericalPhi(w) * Inv2Pi, SphericalTheta(w) * InvPi);
    return Lmap->Lookup(st);
}

__bidevice__ Spectrum Light::Infinite_Sample_Li(const Interaction &ref, const Point2f &u,
                                                vec3f *wi, Float *pdf, 
                                                VisibilityTester *vis) const
{
    Float mapPdf;
    Point2f uv = distribution->SampleContinuous(u, &mapPdf);
    if(IsZero(mapPdf)) return Spectrum(0.f);
    
    Float theta = uv[1] * Pi, phi = uv[0] * 2 * Pi;
    Float cosTheta = std::cos(theta), sinTheta = std::sin(theta);
    Float sinPhi = std::sin(phi), cosPhi = std::cos(phi);
    *wi = LightToWorld(vec3f(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta));
    
    *pdf = mapPdf / (2 * Pi * Pi * sinTheta);
    if(IsZero(sinTheta)) *pdf = 0;
    
    *vis = VisibilityTester(ref, Interaction(ref.p + *wi * (2 * worldRadius),
                                             ref.time));
    return Lmap->Lookup(uv);
}

__bidevice__ Float Light::Infinite_Pdf_Li(const Interaction &ref, const vec3f &w) const{
    vec3f wi = WorldToLight(w);
    Float theta = SphericalTheta(wi), phi = SphericalPhi(wi);
    Float sinTheta = std::sin(theta);
    if(IsZero(sinTheta)) return 0;
    return distribution->Pdf(Point2f(phi * Inv2Pi, theta * InvPi)) / (2 * Pi * Pi * sinTheta);
}

__bidevice__ Spectrum Light::Infinite_Sample_Le(const Point2f &u1, const Point2f &u2,
                                                Float time, Ray *ray, Normal3f *nLight,
                                                Float *pdfPos, Float *pdfDir) const
{
    UMETHOD();
    return 0;
}

__bidevice__ void Light::Infinite_Pdf_Le(const Ray &ray, const Normal3f &n, 
                                         Float *pdfPos, Float *pdfDir) const
{
    UMETHOD();
    *pdfPos = 0;
    *pdfDir = 0;
}