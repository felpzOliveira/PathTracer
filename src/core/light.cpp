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

__bidevice__ Light::Light(const Transform &LightToWorld, int flags, int nSamples)
: flags(flags), nSamples(Max(1, nSamples)), LightToWorld(LightToWorld),
WorldToLight(Inverse(LightToWorld)){}

__bidevice__ void Light::Init_DiffuseArea(const Spectrum &Le, Shape *sshape, 
                                          bool btwoSided)
{
    flags = (int)LightFlags::Area;
    type = LightType::DiffuseArea;
    twoSided = btwoSided;
    shape = sshape;
    Lemit = Le;
    area = shape->Area();
}


__bidevice__ void Light::Init_Distant(const Spectrum &Le, const vec3f &w){
    flags = (int)LightFlags::DeltaDirection;
    type = LightType::Distant;
    Lemit = Le;
    wLight = Normalize(LightToWorld(w));
}

__bidevice__ Spectrum Light::Le(const RayDifferential &r) const{
    switch(type){
        case LightType::DiffuseArea:{
            return DiffuseArea_Le(r);
        } break;
        
        case LightType::Distant:{
            return Distant_Le(r);
        } break;
        
        default:{
            printf("Unknown light type\n");
        }
    }
    
    return Spectrum(0);
}

__bidevice__ Spectrum Light::L(const Interaction &intr, const vec3f &w) const{
    switch(type){
        case LightType::DiffuseArea:{
            return DiffuseArea_L(intr, w);
        } break;
        
        case LightType::Distant:{
            return Distant_L(intr, w);
        } break;
        
        default:{
            printf("Unknown light type\n");
        }
    }
    
    return Spectrum(0);
}

__bidevice__ Spectrum Light::Sample_Le(const Point2f &u1, const Point2f &u2, Float time,
                                       Ray *ray, Normal3f *nLight, Float *pdfPos,
                                       Float *pdfDir) const
{
    switch(type){
        case LightType::DiffuseArea:{
            return DiffuseArea_Sample_Le(u1, u2, time, ray, nLight, pdfPos, pdfDir);
        } break;
        
        case LightType::Distant:{
            return Distant_Sample_Le(u1, u2, time, ray, nLight, pdfPos, pdfDir);
        } break;
        
        default:{
            printf("Unknown light type\n");
        }
    }
    
    *pdfPos = 0;
    *pdfDir = 0;
    return Spectrum(0);
}

__bidevice__ void Light::Pdf_Le(const Ray &ray, const Normal3f &n, 
                                Float *pdfPos, Float *pdfDir) const
{
    switch(type){
        case LightType::DiffuseArea:{
            DiffuseArea_Pdf_Le(ray, n, pdfPos, pdfDir);
        } break;
        
        case LightType::Distant:{
            Distant_Pdf_Le(ray, n, pdfPos, pdfDir);
        } break;
        
        default:{
            *pdfPos = 0;
            *pdfDir = 0;
            printf("Unknown light type\n");
        }
    }
}

__bidevice__ Float Light::Pdf_Li(const Interaction &ref, const vec3f &wi) const{
    switch(type){
        case LightType::DiffuseArea:{
            return DiffuseArea_Pdf_Li(ref, wi);
        } break;
        
        case LightType::Distant:{
            return Distant_Pdf_Li(ref, wi);
        } break;
        
        default:{
            printf("Unknown light type\n");
        }
    }
    
    return 0;
}

__bidevice__ Spectrum Light::Sample_Li(const Interaction &ref, const Point2f &u, 
                                       vec3f *wo, Float *pdf, VisibilityTester *vis) const
{
    switch(type){
        case LightType::DiffuseArea:{
            return DiffuseArea_Sample_Li(ref, u, wo, pdf, vis);
        } break;
        
        case LightType::Distant:{
            return Distant_Sample_Li(ref, u, wo, pdf, vis);
        } break;
        
        default:{
            printf("Unknown light type\n");
        }
    }
    
    *pdf = 0;
    return Spectrum(0);
}

__bidevice__ void Light::Prepare(Aggregator *scene){
    switch(type){
        case LightType::DiffuseArea:{
            DiffuseArea_Prepare(scene);
        } break;
        
        case LightType::Distant:{
            Distant_Prepare(scene);
        } break;
        
        default:{
            printf("Unknown light type\n");
        }
    }
}