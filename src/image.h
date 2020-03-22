#if !defined(IMAGE_H)
#define IMAGE_H
#include <types.h>
#include <cuda_util.cuh>
#include <miniz.h>
#include <iostream>
#include <stdio.h>

/*
for(int x = 0; x < image->width; x += 1){
    for(int y = 0; y < image->height; y += 1){
        int offset = x + y * image->width;
        float fx = (float)x / (float)image->width;
        float fy = (float)y / (float)image->height;
        image->pixels[offset] = glm::vec3(fx, fy, 0.0);
    }
}
*/

__host__ __device__ float clamp(float x){ return x < 0 ? 0 : x > 1 ? 1 : x; }

inline __host__ __device__ glm::vec3 color_remap_to_01(glm::vec3 col){
    float x = col.x;
    float y = col.y;
    float z = col.z;
    
    if(x > 1.0f || y > 1.0f || z > 1.0f){
        float mv = glm::max(glm::max(x, y), z);
        x /= mv;
        y /= mv;
        z /= mv;
    }
    
    return glm::vec3(x,y,z);
}

inline __host__ __device__ glm::ivec3 rgb_to_unsigned(glm::vec3 ccol){
    glm::vec3 col = color_remap_to_01(ccol);
    glm::vec3 value(sqrt(clamp(col.x)), 
                    sqrt(clamp(col.y)), 
                    sqrt(clamp(col.z)));
    
    return glm::ivec3(int(255.99 * value.x),
                      int(255.99 * value.y),
                      int(255.99 * value.z));
}

inline __host__ Image * image_new(int width, int height){
    Image *img = nullptr;
    CHECK(cudaMallocManaged(&img, sizeof(Image)));
    
    img->pixels_count = width * height;
    img->width = width;
    img->height = height;
    size_t rgb_size = img->pixels_count * sizeof(glm::vec3);
    size_t rng_size = img->pixels_count * sizeof(curandState);
    //TODO: Pack this allocation
    img->pixels = (glm::vec3 *)cudaAllocOrFail(rgb_size);
    img->states = (curandState *)cudaAllocOrFail(rng_size);
    
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

inline __host__ void image_write(Image *image, const char *path){
    int size = image->pixels_count * 3;
    unsigned char *data = new unsigned char[size];
    size_t png_data_size = 0;
    void *png_data = nullptr;
    int it = 0;
    
    for(int i = 0; i < image->pixels_count; i += 1){
        glm::ivec3 v = rgb_to_unsigned(image->pixels[i]);
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
    }
    
    delete[] data;
}

#endif