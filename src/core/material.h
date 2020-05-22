#pragma once

#include <geometry.h>
#include <interaction.h>
#include <reflection.h>

//NOTE: Cuda is generating incorrect stack size for some inheritance objects
// so we will have to combine these

enum TextureType{
    ConstantTexture, ImageTexture
};

struct TextureImage{
    unsigned char *data;
    int width, height;
    int is_valid;
};

class Texture{
    public:
    TextureType type;
    Spectrum C;
    
    TextureImage *imagePtr;
    
    __bidevice__ Texture(){}
    __bidevice__ Texture(const Spectrum &t){ Init_ConstantTexture(t); }
    
    //NOTE: When we do add support for images, this might be usefull
    __bidevice__ void operator=(const Texture &t){
        C = t.C;
        type = t.type;
        imagePtr = t.imagePtr;
    }
    
    __bidevice__ Spectrum Evaluate(SurfaceInteraction *si) const;
    
    __bidevice__ void Init_ConstantTexture(Spectrum K);
    __bidevice__ void Init_ImageTexture(TextureImage *imageptr);
    
    private:
    __bidevice__ Spectrum EvaluateConstant(SurfaceInteraction *si) const;
    __bidevice__ Spectrum EvaluateImage(SurfaceInteraction *si) const;
};


__host__ TextureImage *CreateTextureImage(const char *path);

enum MaterialType{
    Matte, Mirror, Glass, Metal, Translucent, Plastic, Uber
};

class Material{
    public:
    MaterialType type;
    Texture K, sigma;
    
    Texture Kt, uRough, vRough, index;
    
    Texture Kr, I, T;
    
    __bidevice__ Material(){}
    __bidevice__ void ComputeScatteringFunctions(BSDF *bsdf, SurfaceInteraction *si, 
                                                 TransportMode mode, bool mLobes) const;
    
    __bidevice__ void Init_Matte(Texture Kd, Texture sigma);
    __bidevice__ void Init_Matte(Spectrum kd, Float sigma=0);
    
    __bidevice__ void Init_Mirror(Texture Kr);
    __bidevice__ void Init_Mirror(Spectrum kr);
    
    __bidevice__ void Init_Metal(Texture R, Texture etaI, Texture etaT, Texture k);
    __bidevice__ void Init_Metal(Spectrum R, Float etaI, Float etaT, Float k);
    
    __bidevice__ void Init_Plastic(Spectrum kd, Spectrum ks, Float rough);
    __bidevice__ void Init_Plastic(Texture kd, Texture ks, Texture rough);
    
    
    __bidevice__ void Init_Glass(Texture Kr, Texture Kt, Texture uRough,
                                 Texture vRough, Texture index);
    
    __bidevice__ void Init_Glass(Spectrum Kr, Spectrum Kt, Float uRough,
                                 Float vRough, Float index);
    
    __bidevice__ void Init_Glass(Spectrum Kr, Spectrum Kt, Float index);
    
    __bidevice__ void Init_Translucent(Spectrum kd, Spectrum ks, Float rough,
                                       Spectrum refl, Spectrum trans);
    
    __bidevice__ void Init_Uber(Texture kd, Texture ks, Texture kr, Texture kt,
                                Texture roughu, Texture roughv, Texture op, Texture eta);
    
    __bidevice__ void Init_Uber(Spectrum kd, Spectrum ks, Spectrum kr, Spectrum kt,
                                Float roughu, Float roughv, Spectrum op, Float eta);
    
    private:
    __bidevice__ void ComputeScatteringFunctionsMatte(BSDF *bsdf, SurfaceInteraction *si, 
                                                      TransportMode mode, bool mLobes) const;
    __bidevice__ void ComputeScatteringFunctionsMirror(BSDF *bsdf, SurfaceInteraction *si, 
                                                       TransportMode mode, bool mLobes) const;
    __bidevice__ void ComputeScatteringFunctionsGlass(BSDF *bsdf, SurfaceInteraction *si, 
                                                      TransportMode mode, bool mLobes) const;
    __bidevice__ void ComputeScatteringFunctionsMetal(BSDF *bsdf, SurfaceInteraction *si, 
                                                      TransportMode mode, bool mLobes) const;
    __bidevice__ void ComputeScatteringFunctionsTranslucent(BSDF *bsdf, SurfaceInteraction *si, 
                                                            TransportMode mode, bool mLobes) const;
    __bidevice__ void ComputeScatteringFunctionsPlastic(BSDF *bsdf, SurfaceInteraction *si, 
                                                        TransportMode mode, bool mLobes) const;
    __bidevice__ void ComputeScatteringFunctionsUber(BSDF *bsdf, SurfaceInteraction *si, 
                                                     TransportMode mode, bool mLobes) const;
};