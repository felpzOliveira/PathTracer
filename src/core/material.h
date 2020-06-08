#pragma once

#include <geometry.h>
#include <interaction.h>
#include <reflection.h>
#include <texture.h>
#include <bssrdf.h>

class Material;

class MatteMaterial{
    public:
    Texture<Spectrum> *Kd;
    Texture<Float> *sigma;
    Texture<Float> *bumpMap;
    int has_bump;
    
    __bidevice__ MatteMaterial(Texture<Spectrum> *kd, Texture<Float> *sig, 
                               Texture<Float> *bump);
    __bidevice__ MatteMaterial(Texture<Spectrum> *kd, Texture<Float> *sig);
    
    __bidevice__ void ComputeScatteringFunctions(BSDF *bsdf, SurfaceInteraction *si, 
                                                 TransportMode mode, bool mLobes) const;
    
    __bidevice__ ~MatteMaterial(){
        TexPtrClean(Kd); TexPtrClean(sigma); TexPtrClean(bumpMap);
    }
};

class PlasticMaterial{
    public:
    Texture<Spectrum> *Kd, *Ks;
    Texture<Float> *roughness;
    Texture<Float> *bumpMap;
    int has_bump;
    
    __bidevice__ PlasticMaterial(Texture<Spectrum> *kd, Texture<Spectrum> *ks, 
                                 Texture<Float> *rough, Texture<Float> *bump);
    
    __bidevice__ PlasticMaterial(Texture<Spectrum> *kd, Texture<Spectrum> *ks, 
                                 Texture<Float> *rough);
    
    __bidevice__ void ComputeScatteringFunctions(BSDF *bsdf, SurfaceInteraction *si, 
                                                 TransportMode mode, bool mLobes) const;
    
    __bidevice__ ~PlasticMaterial(){
        TexPtrClean(Kd); TexPtrClean(Ks); TexPtrClean(roughness); TexPtrClean(bumpMap);
    }
    
};

class MirrorMaterial{
    public:
    Texture<Spectrum> *Kr;
    Texture<Float> *bumpMap;
    int has_bump;
    
    __bidevice__ MirrorMaterial(Texture<Spectrum> *kr, Texture<Float> *bump);
    __bidevice__ MirrorMaterial(Texture<Spectrum> *kr);
    __bidevice__ void ComputeScatteringFunctions(BSDF *bsdf, SurfaceInteraction *si, 
                                                 TransportMode mode, bool mLobes) const;
    
    __bidevice__ ~MirrorMaterial(){
        TexPtrClean(Kr); TexPtrClean(bumpMap);
    }
};

class MetalMaterial{
    public:
    Texture<Spectrum> *eta, *k;
    Texture<Float> *uRough, *vRough;
    Texture<Float> *bumpMap;
    int has_bump;
    
    __bidevice__ MetalMaterial(Texture<Spectrum> *eta, Texture<Spectrum> *k, 
                               Texture<Float> *urough, Texture<Float> *vrough,
                               Texture<Float> *bump);
    
    __bidevice__ MetalMaterial(Texture<Spectrum> *eta, Texture<Spectrum> *k, 
                               Texture<Float> *urough, Texture<Float> *vrough);
    
    __bidevice__ void ComputeScatteringFunctions(BSDF *bsdf, SurfaceInteraction *si, 
                                                 TransportMode mode, bool mLobes) const;
    
    __bidevice__ ~MetalMaterial(){
        TexPtrClean(eta); TexPtrClean(k);
        TexPtrClean(uRough); TexPtrClean(vRough);
        TexPtrClean(bumpMap);
    }
};

class GlassMaterial{
    public:
    Texture<Spectrum> *Kr, *Kt;
    Texture<Float> *uRough, *vRough, *index;
    Texture<Float> *bumpMap;
    int has_bump;
    
    __bidevice__ GlassMaterial(Texture<Spectrum> *kr, Texture<Spectrum> *kt,
                               Texture<Float> *urough, Texture<Float> *vrough,
                               Texture<Float> *index, Texture<Float> *bump);
    
    __bidevice__ GlassMaterial(Texture<Spectrum> *kr, Texture<Spectrum> *kt,
                               Texture<Float> *urough, Texture<Float> *vrough,
                               Texture<Float> *index);
    
    __bidevice__ void ComputeScatteringFunctions(BSDF *bsdf, SurfaceInteraction *si, 
                                                 TransportMode mode, bool mLobes) const;
    
    __bidevice__ ~GlassMaterial(){
        TexPtrClean(Kr); TexPtrClean(Kt);
        TexPtrClean(vRough); TexPtrClean(uRough);
        TexPtrClean(bumpMap); TexPtrClean(index);
    }
};

class UberMaterial{
    public:
    Texture<Spectrum> *Kd, *Ks, *Kr, *Kt, *opacity;
    Texture<Float> *uRough, *vRough, *eta;
    Texture<Float> *bumpMap;
    int has_bump;
    
    __bidevice__ UberMaterial(Texture<Spectrum> *kd, Texture<Spectrum> *ks,
                              Texture<Spectrum> *kr, Texture<Spectrum> *kt,
                              Texture<Float> *urough, Texture<Float> *vrough,
                              Texture<Spectrum> *op, Texture<Float> *eta,
                              Texture<Float> *bump);
    
    __bidevice__ UberMaterial(Texture<Spectrum> *kd, Texture<Spectrum> *ks,
                              Texture<Spectrum> *kr, Texture<Spectrum> *kt,
                              Texture<Float> *urough, Texture<Float> *vrough,
                              Texture<Spectrum> *op, Texture<Float> *eta);
    
    __bidevice__ void ComputeScatteringFunctions(BSDF *bsdf, SurfaceInteraction *si, 
                                                 TransportMode mode, bool mLobes) const;
    __bidevice__ ~UberMaterial(){
        TexPtrClean(Kr); TexPtrClean(Kt);
        TexPtrClean(vRough); TexPtrClean(uRough);
        TexPtrClean(bumpMap); TexPtrClean(eta);
        TexPtrClean(Kd); TexPtrClean(Ks);
        TexPtrClean(opacity);
    }
};

class TranslucentMaterial{
    public:
    Texture<Spectrum> *Kd, *Ks, *reflect, *transmit;
    Texture<Float> *roughness;
    Texture<Float> *bumpMap;
    int has_bump;
    
    __bidevice__ TranslucentMaterial(Texture<Spectrum> *kd, Texture<Spectrum> *ks,
                                     Texture<Spectrum> *refl, Texture<Spectrum> *trans,
                                     Texture<Float> *rough, Texture<Float> *bump);
    
    __bidevice__ TranslucentMaterial(Texture<Spectrum> *kd, Texture<Spectrum> *ks,
                                     Texture<Spectrum> *refl, Texture<Spectrum> *trans,
                                     Texture<Float> *rough);
    
    __bidevice__ void ComputeScatteringFunctions(BSDF *bsdf, SurfaceInteraction *si, 
                                                 TransportMode mode, bool mLobes) const;
    
    __bidevice__ ~TranslucentMaterial(){
        TexPtrClean(Kd); TexPtrClean(Ks);
        TexPtrClean(roughness); TexPtrClean(bumpMap);
        TexPtrClean(reflect); TexPtrClean(transmit);
    }
};

class SubsurfaceMaterial{
    public:
    Float scale;
    Texture<Spectrum> *Kr, *Kt, *sigma_a, *sigma_s;
    Texture<Float> *uRough, *vRough;
    Float eta;
    BSSRDFTable *table;
    
    __bidevice__ SubsurfaceMaterial(Texture<Spectrum> *kr, Texture<Spectrum> *kt,
                                    Texture<Spectrum> *sa, Texture<Spectrum> *ss,
                                    Texture<Float> *urough, Texture<Float> *vrough,
                                    Float g, Float e, Float scale=1);
    
    __bidevice__ void ComputeScatteringFunctions(BSDF *bsdf, SurfaceInteraction *si, 
                                                 TransportMode mode, bool mLobes,
                                                 Material *sourceMat) const;
    
    __bidevice__ ~SubsurfaceMaterial(){
        TexPtrClean(Kr); TexPtrClean(Kt);
        TexPtrClean(sigma_a); TexPtrClean(sigma_s);
        TexPtrClean(vRough); TexPtrClean(uRough);
        //TODO: cleanup BSSRDFTable
    }
};

class KdSubsurfaceMaterial{
    public:
    Float scale;
    Texture<Spectrum> *Kr, *Kt, *Kd, *mfp;
    Texture<Float> *uRough, *vRough;
    Texture<Float> *bumpMap;
    int has_bump;
    Float eta;
    BSSRDFTable *table;
    
    __bidevice__ KdSubsurfaceMaterial(Texture<Spectrum> *kd, Texture<Spectrum> *kr,
                                      Texture<Spectrum> *kt, Texture<Spectrum> *mfp,
                                      Texture<Float> *urough, Texture<Float> *vrough,
                                      Texture<Float> *map, Float e, Float scale);
    
    __bidevice__ KdSubsurfaceMaterial(Texture<Spectrum> *kd, Texture<Spectrum> *kr,
                                      Texture<Spectrum> *kt, Texture<Spectrum> *mfp,
                                      Texture<Float> *urough, Texture<Float> *vrough,
                                      Float e, Float scale);
    
    __bidevice__ void ComputeScatteringFunctions(BSDF *bsdf, SurfaceInteraction *si, 
                                                 TransportMode mode, bool mLobes,
                                                 Material *sourceMat) const;
    __bidevice__ ~KdSubsurfaceMaterial(){
        TexPtrClean(Kr); TexPtrClean(Kt);
        TexPtrClean(Kd);TexPtrClean(bumpMap);
        TexPtrClean(vRough); TexPtrClean(uRough);
        //TODO: cleanup BSSRDFTable
    }
};

// This does not implement any material, it serves for Procedural shapes
// that wish to have multiple Spectrums and BRDFs/BSSRDFs based on the
// interaction->faceIndex. Can control the shape being referenced by the 
// 'id' parameter.
class PlayGroundMaterial{
    public:
    Texture<Spectrum> **Ls;
    int size;
    int id;
    
    __bidevice__ PlayGroundMaterial(Texture<Spectrum> **ls, int size, int id);
    
    __bidevice__ void ComputeScatteringFunctions(BSDF *bsdf, SurfaceInteraction *si, 
                                                 TransportMode mode, bool mLobes,
                                                 Material *sourceMat) const;
    
    __bidevice__ ~PlayGroundMaterial(){
        for(int i = 0; i < size; i++){
            TexPtrClean(Ls[i]);
        }
    }
};

enum MaterialType{
    Matte=1, Mirror, Glass, Metal, Translucent, Plastic, Uber,
    Subsurface, KdSubsurface,
    PlayGround
};

class Material{
    public:
    MaterialType type;
    void *material;
    __bidevice__ Material(MaterialType tp){ type = tp; material = nullptr; }
    __bidevice__ Material(){ material = nullptr; type = MaterialType(0); }
    __bidevice__ void Set(void *ptr) { material = ptr; }
    __bidevice__ void ComputeScatteringFunctions(BSDF *bsdf, SurfaceInteraction *si, 
                                                 TransportMode mode, bool mLobes);
};