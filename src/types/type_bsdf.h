#if !defined(TYPE_BSDF_H)
#define TYPE_BSDF_H
#include <cutil.h>
#include <spectrum.h>

#include <types/type_microfacet.h>
#include <types/type_fresnel.h>
#include <onb.h>

#define MAX_BSDFS 8

enum BxDFType {
    BSDF_REFLECTION = 1 << 0,
    BSDF_TRANSMISSION = 1 << 1,
    BSDF_DIFFUSE = 1 << 2,
    BSDF_GLOSSY = 1 << 3,
    BSDF_SPECULAR = 1 << 4,
    BSDF_ALL = BSDF_DIFFUSE | BSDF_GLOSSY | BSDF_SPECULAR | BSDF_REFLECTION |
        BSDF_TRANSMISSION,
};

enum BxDFModel{
    SpecularReflectionBxDF,
    SpecularTransmissionBxDF,
    FresnelSpecularBxDF,
    LambertianReflectionBxDF,
    LambertianTransmissionBxDF,
    OrenNayarBxDF,
    MicrofacetReflectionBxDF,
    MicrofacetTransmissionBxDF,
    NoneBxDF
};

typedef struct{
    BxDFType type;
    BxDFModel model;
}BxDF_handle;

struct BxDF{
    BxDF_handle self; //identifier
    Spectrum S;
    Spectrum R;
    
    __bidevice__ BxDF(){
        self.model = NoneBxDF;
    }
    
    __bidevice__ BxDF(const BxDF &o){
        memcpy(this, &o, sizeof(BxDF));
    }
    
    union{
        struct{
            float A; 
            float B;
        }OrenNayar;
        
        struct{
            Fresnel fresnel;
            float etaA, etaB;
        }Specular;
        
        struct{
            Fresnel fresnel;
            MicrofacetDistribution dist;
            float etaA, etaB;
        }Microfacet;
    };
};

class BSDF{
    public:
    BxDF bxdfs[MAX_BSDFS];
    glm::vec3 ng;
    float eta;
    int nBxDFs;
    Onb uvw;
    
    __bidevice__ BSDF(){ nBxDFs = 0; eta = 1.0f; }
    
    __bidevice__ void Set(glm::vec3 n){
        onb_from_w(&uvw, n);
        ng = n;
    }
    
    __bidevice__ BSDF(glm::vec3 n){
        nBxDFs = 0; eta = 1.0f;
        Set(n);
    }
    
    __bidevice__ glm::vec3 WorldToLocal(glm::vec3 v){
        return onb_world_to_local(&uvw, v);
    }
    
    __bidevice__ glm::vec3 LocalToWorld(glm::vec3 v){
        return onb_local_to_world(&uvw, v);
    }
    
};

inline __bidevice__
void BSDF_Insert(BSDF *bssrdf, BxDF *bxdf){
    if(bssrdf->nBxDFs < MAX_BSDFS){
        memcpy(&bssrdf->bxdfs[bssrdf->nBxDFs], bxdf, sizeof(BxDF));
        bssrdf->nBxDFs++;
    }else{
        printf("Took too many BxDFs!\n");
    }
}

#endif