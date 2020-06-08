#include <material.h>

__bidevice__ PlayGroundMaterial::PlayGroundMaterial(Texture<Spectrum> **ls, int size, int id) 
: Ls(ls), size(size), id(id){ AssertA(ls && size > 0, "Invalid Spectrums"); }

__bidevice__ void PlayGroundMaterial::ComputeScatteringFunctions(BSDF *bsdf, 
                                                                 SurfaceInteraction *si, 
                                                                 TransportMode mode, 
                                                                 bool mLobes,
                                                                 Material *sourceMat) const
{
    Spectrum base = Ls[0]->Evaluate(si);
    for(int i = 0; i < size; i++){
        if(si->faceIndex == i){
            base = Ls[i]->Evaluate(si);
            break;
        }
    }
    
    BxDF bxdf(BxDFImpl::LambertianReflection);
    bxdf.Init_LambertianReflection(base);
    bsdf->Push(&bxdf);
}