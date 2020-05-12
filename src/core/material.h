#pragma once

#include <geometry.h>
#include <interaction.h>
#include <reflection.h>

//NOTE: Cuda is generating incorrect stack size for some inheritance objects
// so we will have to combine these

enum TextureType{
    Constant,
};

class Texture{
    public:
    TextureType type;
    Spectrum C;
    
    __bidevice__ Texture(){}
    __bidevice__ Texture(const Spectrum &t){
        Init_ConstantTexture(t);
    }
    __bidevice__ void operator=(const Texture &t){
        C = t.C;
        type = t.type;
    }
    
    __bidevice__ Spectrum Evaluate(SurfaceInteraction *si) const;
    __bidevice__ void Init_ConstantTexture(Spectrum K);
    
    private:
    __bidevice__ Spectrum EvaluateConstant(SurfaceInteraction *si) const;
};


enum MaterialType{
    Matte, Mirror, Glass
};

class Material{
    public:
    MaterialType type;
    Texture K, sigma;
    
    Texture Kt, uRough, vRough, index;
    
    __bidevice__ Material(){}
    __bidevice__ void ComputeScatteringFunctions(BSDF *bsdf, SurfaceInteraction *si, 
                                                 TransportMode mode, bool mLobes) const;
    
    __bidevice__ void Init_Matte(Texture Kd, Texture sigma);
    __bidevice__ void Init_Mirror(Texture Kr);
    __bidevice__ void Init_Glass(Texture Kr, Texture Kt, Texture uRough,
                                 Texture vRough, Texture index);
    
    __bidevice__ void Init_Glass(Spectrum Kr, Spectrum Kt, Float uRough,
                                 Float vRough, Float index);
    
    private:
    __bidevice__ void ComputeScatteringFunctionsMatte(BSDF *bsdf, SurfaceInteraction *si, 
                                                      TransportMode mode, bool mLobes) const;
    __bidevice__ void ComputeScatteringFunctionsMirror(BSDF *bsdf, SurfaceInteraction *si, 
                                                       TransportMode mode, bool mLobes) const;
    __bidevice__ void ComputeScatteringFunctionsGlass(BSDF *bsdf, SurfaceInteraction *si, 
                                                      TransportMode mode, bool mLobes) const;
};