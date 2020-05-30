#pragma once

#include <interaction.h>
#include <reflection.h>
#include <util.h>
#include <sampling.h>

#define TexPtrClean(ptr) if(ptr){ ptr->Release(); delete ptr; }

struct ImageData{
    Spectrum *data;
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
        Spectrum rgb = image->data[i + image->width * j];
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

__bidevice__ Float Lanczos(Float x, Float tau=2);

enum class ImageWrap{ Repeat, Black, Clamp };

template <typename T> inline __bidevice__ bool IsPowerOf2(T v){
    return v && !(v & (v - 1));
}

inline __bidevice__ int32_t RoundUpPow2(int32_t v){
    v--;
    v |= v >> 1;    v |= v >> 2;
    v |= v >> 4;    v |= v >> 8;
    v |= v >> 16;
    return v+1;
}


template<typename T>
struct PyramidLevel{
    T *texels;
    int rx, ry;
};


template<typename T>
class MipMap{
    public:
    ImageWrap wrapMode;
    Point2i resolution;
    PyramidLevel<T> *pyramid;
    int nLevels;
    T black;
    
    __bidevice__ int Width() const{ return resolution[0]; }
    __bidevice__ int Height() const{ return resolution[1]; }
    __bidevice__ int Levels() const{ return nLevels; }
    
    __bidevice__ MipMap(){}
    
    __bidevice__ const T &Texel(int level, int s, int t) const{
        AssertA(level < nLevels && level >= 0, "Invalid texel fetch level");
        PyramidLevel<T> *pLevel = &pyramid[level];
        switch(wrapMode){
            case ImageWrap::Repeat:{
                s = Mod(s, pLevel->rx);
                t = Mod(t, pLevel->ry);
            } break;
            
            case ImageWrap::Clamp:{
                s = Clamp(s, 0, pLevel->rx - 1);
                t = Clamp(t, 0, pLevel->ry - 1);
            } break;
            
            case ImageWrap::Black:{
                if(s < 0 || s >= pLevel->rx || t < 0 || t >= pLevel->ry)
                    return black;
            } break;
        }
        
        return pLevel->texels[s + t * pLevel->rx];
    }
    
    __bidevice__ T triangle(int level, const Point2f &st) const{
        level = Clamp(level, 0, Levels() - 1);
        PyramidLevel<T> *pLevel = &pyramid[level];
        Float s = st[0] * pLevel->rx - 0.5f;
        Float t = st[1] * pLevel->ry - 0.5f;
        int s0 = Floor(s), t0 = Floor(t);
        Float ds = s - s0, dt = t - t0;
        return (1 - ds) * (1 - dt) * Texel(level, s0, t0) +
            (1 - ds) * dt * Texel(level, s0, t0 + 1) +
            ds * (1 - dt) * Texel(level, s0 + 1, t0) +
            ds * dt * Texel(level, s0 + 1, t0 + 1);
    }
    
    __bidevice__ T Lookup(const Point2f &st, Float width = 0.f) const{
        Float level = Levels() - 1 + Log2(Max(width, (Float)1e-8));
        if(level < 0){
            return triangle(0, st);
        }else if(level >= Levels() - 1){
            return Texel(Levels() - 1, 0, 0);
        }else{
            int iLevel = Floor(level);
            Float delta = level - iLevel;
            return Lerp(delta, triangle(iLevel, st), triangle(iLevel + 1, st));
        }
    }
    
    __bidevice__ T Lookup(const Point2f &st, vec2f dst0, vec2f dst1) const{
        Float width = Max(Max(Absf(dst0[0]), Absf(dst0[1])),
                          Max(Absf(dst1[0]), Absf(dst1[1])));
        return Lookup(st, 2 * width);
    }
};

typedef MipMap<Spectrum> MMSpectrum;
typedef MipMap<Float> MMFloat;

__host__ MMSpectrum *BuildSpectrumMipMap(const char *path, Distribution2D **distr,
                                         const Spectrum &scale = Spectrum(1));