#pragma once
#include <cutil.h>
#include <stdio.h>
#include <geometry.h>
#include <miniz.h>
#include <ppm.h>

#define UMETHOD() printf("Warning: Invocation of unimplemented method [ %s ]", __FUNCTION__)

template<typename T, class C> inline __bidevice__
bool QuickSort(T *arr, int elements, C compare){
#define  MAX_LEVELS  1000
    int beg[MAX_LEVELS], end[MAX_LEVELS], i=0, L, R;
    T piv;
    beg[0] = 0;
    end[0] = elements;
    while(i >= 0){
        L = beg[i]; 
        R = end[i]-1;
        if(L < R){
            piv = arr[L]; 
            if(i == MAX_LEVELS-1) return false;
            
            while(L < R){
                while(compare(&arr[R], &piv) && L < R) R--; if (L < R) arr[L++] = arr[R];
                while(compare(&piv, &arr[L]) && L < R) L++; if (L < R) arr[R--] = arr[L]; 
            }
            
            arr[L] = piv; 
            beg[i+1] = L+1; 
            end[i+1] = end[i]; 
            end[i++] = L; 
        }else{
            i--; 
        }
    }
    
    return true;
}

inline __host__ void ConvertPPMtoPNG(const char *path, const char *out){
    unsigned char *values = nullptr;
    int width = 0, height = 0;
    if(!PPMRead(path, &values, width, height)){
        printf("Failed to get values for %s\n", path);
    }else{
        size_t png_data_size = 0;
        void *png_data = nullptr;
        
        png_data = tdefl_write_image_to_png_file_in_memory_ex(values, width, height, 3,
                                                              &png_data_size, 6, MZ_TRUE);
        if(!png_data){
            printf("Failed to get PNG\n");
        }else{
            FILE *fp = fopen(out, "wb");
            fwrite(png_data, 1, png_data_size, fp);
            fclose(fp);
            printf("Done\n");
        }
    }
}


inline __host__ Float rand_float(){
    return rand() / (RAND_MAX + 1.f);
}

