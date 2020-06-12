#include <texture.h>
#include <image_util.h>
#include <string>

struct ResampleWeight{
    int firstTexel;
    Float weight[4];
};

__host__ bool hasEnding(std::string const &fullString, std::string const &ending){
    if(fullString.length() >= ending.length()){
        return (0 == fullString.compare (fullString.length() - ending.length(), 
                                         ending.length(), ending));
    }else{
        return false;
    }
}

__bidevice__ Float Lanczos(Float x, Float tau){
    x = std::abs(x);
    if(x < 1e-5f) return 1;
    if(x > 1.f) return 0;
    x *= Pi;
    Float s = std::sin(x * tau) / (x * tau);
    Float lanczos = std::sin(x) / x;
    return s * lanczos;
}

__host__ ImageData *LoadTextureImageData(const char *path){
    int nx = 0, ny = 0, nn = 0;
    ImageData *tImage = cudaAllocateVx(ImageData, 1);
    if(hasEnding(path, ".exr")){
        tImage->data = ReadImageEXR(path, nx, ny);
        Assert(tImage->data && nx > 0 && ny > 0);
    }else{
        unsigned char *ptr = ReadImage(path, nx, ny, nn);
        Assert(ptr && nx > 0 && ny > 0);
        tImage->data = cudaAllocateVx(Spectrum, nx * ny);
        tImage->width = nx;
        tImage->height = ny;
        
        int imit = 0;
        for(int i = 0; i < nx * ny; i++){
            Float r = 0, g = 0, b = 0;
            if(nn > 0){
                r = ptr[imit] / 255.f; imit++;
            }
            if(nn > 1){
                g = ptr[imit] / 255.f; imit++;
            }
            if(nn > 2){
                b = ptr[imit] / 255.f; imit++;
            }
            
            if(nn == 4) imit++; //skip alpha
            tImage->data[i] = Spectrum(r, g, b);
        }
        
        free(ptr);
    }
    
    tImage->is_valid = 1;
    return tImage;
}



template<typename T>
__host__ T Texel(PyramidLevel<T> *pLevel, int s, int t, ImageWrap wrapMode){
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
                return T(0);
        } break;
    }
    
    return pLevel->texels[s + t * pLevel->rx];
}


__host__ ResampleWeight *GetResampleWeights(int oldRes, int newRes){
    AssertA(oldRes <= newRes, "New resolution must be bigger than old one");
    ResampleWeight * wt = new ResampleWeight[newRes];
    Float filterwidth = 2.f;
    for(int i = 0; i < newRes; i++){
        Float center = (i + .5f) * oldRes / newRes;
        wt[i].firstTexel = std::floor((center - filterwidth) + 0.5f);
        for(int j = 0; j < 4; j++){
            Float pos = wt[i].firstTexel + j + .5f;
            wt[i].weight[j] = Lanczos((pos - center) / filterwidth);
        }
        
        Float invSumWts = 1 / (wt[i].weight[0] + wt[i].weight[1] +
                               wt[i].weight[2] + wt[i].weight[3]);
        for(int j = 0; j < 4; ++j) wt[i].weight[j] *= invSumWts;
    }
    
    return wt;
}

template<typename T> __host__ 
PyramidLevel<T> *BuildMipMap(T *img, Point2i &resolution, int &nLevels, ImageWrap wrapMode){
    
    T *resampledImage = nullptr;
    printf(" * Building MipMap");
    // resample the image to be power of 2
    if(!IsPowerOf2(resolution[0]) || !IsPowerOf2(resolution[1])){
        printf("\r * Building MipMap [ Re-scaling ]");
        Point2i resPow2(RoundUpPow2(resolution[0]), RoundUpPow2(resolution[1]));
        
        resampledImage = cudaAllocateVx(T, resPow2[0] * resPow2[1]);
        ResampleWeight *sWeights = GetResampleWeights(resolution[0], resPow2[0]);
        
        for(int t = 0; t < resolution[1]; t++){
            for(int s = 0; s < resPow2[0]; s++){
                resampledImage[t * resPow2[0] + s] = 0.f;
                for(int j = 0; j < 4; j++){
                    int origS = sWeights[s].firstTexel + j;
                    if(wrapMode == ImageWrap::Repeat){
                        origS = Mod(origS, resolution[0]);
                    }else if(wrapMode == ImageWrap::Clamp){
                        origS = Clamp(origS, 0, resolution[0] - 1);
                    }
                    
                    if(origS >= 0 && origS < (int)resolution[0]){
                        resampledImage[t * resPow2[0] + s] +=
                            sWeights[s].weight[j] * img[t * resolution[0] + origS];
                    }
                }
            }
        }
        
        ResampleWeight *tWeights = GetResampleWeights(resolution[1], resPow2[1]);
        
        T *workData = new T[resPow2[1]];
        
        for(int s = 0; s < resPow2[0]; s++){
            for(int t = 0; t < resPow2[1]; t++){
                workData[t] = 0.f;
                for(int j = 0; j < 4; j++){
                    int offset = tWeights[t].firstTexel + j;
                    if(wrapMode == ImageWrap::Repeat){
                        offset = Mod(offset, resolution[1]);
                    }else if (wrapMode == ImageWrap::Clamp){
                        offset = Clamp(offset, 0, (int)resolution[1] - 1);
                    }
                    
                    if(offset >= 0 && offset < (int)resolution[1]){
                        workData[t] += tWeights[t].weight[j] *
                            resampledImage[offset * resPow2[0] + s];
                    }
                }
            }
            
            for (int t = 0; t < resPow2[1]; t++)
                resampledImage[t * resPow2[0] + s] = Clamp(workData[t], T(0), T(1));
        }
        
        delete[] workData;
        delete[] sWeights;
        delete[] tWeights;
        resolution = resPow2;
    }else{
        resampledImage = cudaAllocateVx(T, (resolution[0] * resolution[1]));
        for(int i = 0; i < resolution[0] * resolution[1]; i++){
            resampledImage[i] = img[i];
        }
    }
    
    nLevels = 1 + Log2Int(Max(resolution[0], resolution[1]));
    
    PyramidLevel<T> *pyramid = cudaAllocateVx(PyramidLevel<T>, nLevels);
    PyramidLevel<T> *pLevel = &pyramid[0];
    
    pLevel->texels = resampledImage;
    pLevel->rx = resolution[0];
    pLevel->ry = resolution[1];
    
    for(int i = 1; i < nLevels; i++){
        printf("\r * Building MipMap [ Level: %d / %d ]", i, nLevels);
        PyramidLevel<T> *cLevel = &pyramid[i];
        pLevel = &pyramid[i-1];
        
        int sRes = Max(1, pLevel->rx / 2);
        int tRes = Max(1, pLevel->ry / 2);
        cLevel->texels = cudaAllocateVx(T, sRes * tRes);
        
        for(int t = 0; t < tRes; t++){
            for(int s = 0; s < sRes; s++){
                int index = s + t * sRes;
                cLevel->texels[index] = 
                    0.25f*(Texel<T>(pLevel, 2 * s, 2 * t, wrapMode) +
                           Texel<T>(pLevel, 2 * s + 1, 2 * t, wrapMode) +
                           Texel<T>(pLevel, 2 * s, 2 * t + 1, wrapMode) +
                           Texel<T>(pLevel, 2 * s + 1, 2 * t + 1, wrapMode));
                
            }
        }
        
        cLevel->rx = sRes;
        cLevel->ry = tRes;
    }
    printf("\r * Building MipMap [ Level: %d / %d ]\n", nLevels, nLevels);
    return pyramid;
}

__host__ MMSpectrum *BuildSpectrumMipMap(const char *path, Distribution2D **distr,
                                         const Spectrum &scale)
{
    int width, height;
    MMSpectrum *mipmap = cudaAllocateVx(MMSpectrum, 1);
    printf(" * Reading EXR Map...");
    Spectrum *L = ReadImageEXR(path, width, height);
    for(int i = 0; i < width * height; i++){
        L[i] *= scale;
    }
    
    printf("OK\n");
    Point2i resolution(width, height);
    mipmap->wrapMode = ImageWrap::Repeat;
    mipmap->pyramid = BuildMipMap(L, resolution, mipmap->nLevels, ImageWrap::Repeat);
    mipmap->black = Spectrum(0);
    mipmap->resolution = resolution;
    
    if(distr){
        width = 2 * mipmap->Width();
        height = 2 * mipmap->Height();
        Float *img = cudaAllocateVx(Float, width * height);
        float fwidth = 0.5f / Min(width, height);
        printf(" * Creating light distribution...");
        fflush(stdout);
        for(int v = 0; v < height; v++){
            Float vp = (v + .5f) / (Float)height;
            Float sinTheta = Sin(Pi * (v + .5f) / height);
            for(int u = 0; u < width; u++){
                Float up = (u + .5f) / (Float)width;
                img[u + v * width] = mipmap->Lookup(Point2f(up, vp), fwidth).Y();
                img[u + v * width] *= sinTheta;
            }
        }
        
        *distr = CreateDistribution2D(img, width, height);
        printf("OK\n");
        fflush(stdout);
    }
    
    cudaFree(L);
    
    return mipmap;
}