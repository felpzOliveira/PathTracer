#pragma once

#include <interaction.h>
#include <reflection.h>
#include <util.h>

#define TexPtrClean(ptr) if(ptr){ ptr->Release(); delete ptr; }

struct ImageData{
    unsigned char *data;
    int width, height;
    int is_valid;
    
    __bidevice__ Float GetAspectRatio(){ return (Float)(width) / (Float)(height); }
};

inline __bidevice__ 
void convertIn(const Spectrum &from, Spectrum *to, Float scale, bool gamma){
    for (int i = 0; i < 3; ++i)
        (*to)[i] = scale * (gamma ? InverseGammaCorrect(from[i]): from[i]);
}

template<typename T> class Texture{
    public:
    __bidevice__ virtual T Evaluate(SurfaceInteraction *) const = 0;
    __bidevice__ virtual void Release() = 0;
};

template<typename T> class TextureConstant : public Texture<T>{
    public:
    T value;
    __bidevice__ TextureConstant(const T &val) : value(val) {}
    __bidevice__ virtual T Evaluate(SurfaceInteraction *) const override{
        return value;
    }
    
    __bidevice__ virtual void Release() override {}
};

template<typename T> class TextureImage : public Texture<T>{
    public:
    ImageData *image;
    int dimension;
    __bidevice__ TextureImage(ImageData *ptr){
        image = ptr;
        SetDim(T(0));
    }
    
    __bidevice__ virtual T Evaluate(SurfaceInteraction *si) const override{
        //TODO: Needs mapping for scaling and stuff
        int i = si->uv.x * image->width;
        int j = si->uv.y * image->height;
        if(i < 0) i = 0; if(i > image->width - 1)  i = image->width  - 1;
        if(j < 0) j = 0; if(j > image->height - 1) j = image->height - 1;
        Float r = image->data[3*i + 3*image->width*j+0] / 255.f;
        Float g = image->data[3*i + 3*image->width*j+1] / 255.f;
        Float b = image->data[3*i + 3*image->width*j+2] / 255.f;
        Spectrum rgb(r, g, b);
        convertIn(rgb, &rgb, 1.f, true);
        return ConvertOut(rgb, T(0));
    }
    
    __bidevice__ virtual void Release() override{}
    
    
    __bidevice__ Float ConvertOut(Spectrum rgb, Float trigger) const{
        (void)trigger;
        return rgb[0];
    }
    
    __bidevice__ Spectrum ConvertOut(Spectrum rgb, Spectrum trigger) const{
        (void)trigger;
        return rgb;
    }
    
    __bidevice__ void SetDim(Float val){
        (void)val;
        dimension = 1;
    }
    
    __bidevice__ void SetDim(Spectrum val){
        (void)val;
        dimension = 3;
    }
    
};

template<typename T> inline __bidevice__ 
TextureConstant<T> *ConstantTexture(T v){ return new TextureConstant<T>(v); }

template<typename T> inline __bidevice__
TextureImage<T> *ImageTexture(ImageData *data){ return new TextureImage<T>(data); }

__host__ ImageData *LoadTextureImageData(const char *path);