#include <image_util.h>
#include <cutil.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <miniz.h>
#define TINYEXR_USE_MINIZ 0
#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>

unsigned char * ReadImage(const char *path, int &width, int &height, int &channels){
    int nx = 0, ny = 0, nn = 0;
    unsigned char * data = stbi_load(path, &nx, &ny, &nn, 0);
    width = nx;
    height = ny;
    channels = nn;
    return data;
}

Spectrum *ReadImageEXR(const char *path, int &width, int &height){
    int nx, ny;
    Float *out = nullptr;
    Spectrum *sOut = nullptr;
    const char *err = nullptr;
    int ret = LoadEXR(&out, &nx, &ny, path, &err);
    if(ret != TINYEXR_SUCCESS){
        if(err){
            printf("Fail read %s\n", err);
            FreeEXRErrorMessage(err);
        }
    }else{
        width = nx; height = ny;
        sOut = cudaAllocateVx(Spectrum, nx * ny);
        int it = 0;
        for(int i = 0; i < nx * ny; i++){
            sOut[i] = Flip(Spectrum(out[it++], out[it++], out[it++]));
            it++; // alpha
        }
    }
    
    if(out) free(out);
    
    return sOut;
}
