#if !defined(IMAGE_H)
#define IMAGE_H
#include <types.h>
#include <cutil.h>
#include <miniz.h>
#include <iostream>
#include <stdio.h>
#include <utilities.h>

__host__ __device__ float clamp01(float x){ return x<0?0:(x>0.999f?0.999f:x); }

inline __host__ __device__ glm::ivec3 rgb_to_unsigned(glm::vec3 ccol, int samples){
    if(IsNaN<glm::vec3>(ccol)){
        printf("NaN\n");
    }
    
    glm::vec3 col = ccol;
    
    col = glm::vec3(glm::sqrt(col.x),
                    glm::sqrt(col.y),
                    glm::sqrt(col.z));
    
    return glm::ivec3(int(256.0f * clamp01(col.x)),
                      int(256.0f * clamp01(col.y)),
                      int(256.0f * clamp01(col.z)));
}

inline __host__ Image * image_new(int width, int height){
    Image *img = (Image *)cudaAllocate(sizeof(Image));
    
    img->pixels_count = width * height;
    img->width = width;
    img->height = height;
    size_t rgb_size = img->pixels_count * sizeof(glm::vec3);
    size_t rng_size = img->pixels_count * sizeof(curandState);
    //TODO: Pack this allocation
    img->pixels = (glm::vec3 *)cudaAllocate(rgb_size);
    img->states = (curandState *)cudaAllocate(rng_size);
    
    for(int i = 0; i < img->pixels_count; i += 1){
        img->pixels[i] = glm::vec3(0.0f, 0.0f, 0.0f);
    }
    
    return img;
}

inline __host__ void image_free(Image *img){
    if(img){
        if(img->pixels){
            cudaFree(img->pixels);
        }
        
        if(img->states){
            cudaFree(img->states);
        }
        cudaFree(img);
    }
}

inline __host__ void image_write(Image *image, const char *path, int samples){
    int size = image->pixels_count * 3;
    unsigned char *data = new unsigned char[size];
    size_t png_data_size = 0;
    void *png_data = nullptr;
    int it = 0;
    
    for(int i = 0; i < image->pixels_count; i += 1){
        glm::ivec3 v = rgb_to_unsigned(image->pixels[i], samples);
        data[it++] = v.x;
        data[it++] = v.y;
        data[it++] = v.z;
    }
    
    png_data = tdefl_write_image_to_png_file_in_memory_ex(data, image->width, 
                                                          image->height, 3,
                                                          &png_data_size, 6,
                                                          MZ_TRUE);
    
    if(!png_data){
        std::cout << "Failed to get PNG" << std::endl;
    }else{
        remove(path);
        FILE *fp = fopen(path, "wb");
        fwrite(png_data, 1, png_data_size, fp);
        fclose(fp);
        std::cout << "Saved PNG " << path << std::endl;
        std::string cmd("display ");
        cmd += path;
        system(cmd.c_str());
    }
    
    delete[] data;
}

#endif