#if !defined(STL_LOADER_H)
#define STL_LOADER_H

#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#define STL_ASSERT(x) if((x) == 0) { printf("Failed to read!\n"); exit(0); }

template<typename T>
static T * stl_load(const char *path, int *n_triangles){
    FILE *fp = fopen(path, "rb");
    if(fp){
        char header[80];
        char dummy[2];
        unsigned int triangles = 0;
        memset(header, 0x00, sizeof(header));
        //read header
        STL_ASSERT(fread(header, sizeof(header), 1, fp));
        STL_ASSERT(fread(&triangles, sizeof(unsigned int), 1, fp));
        
        std::cout << "Read " << triangles << " triangles" << std::endl;
        float p[3];
        
        T * resp = (T *)malloc(sizeof(T)*triangles * 3);
        
        int it = 0;
        for(unsigned int i = 0; i < triangles; i += 1){
            STL_ASSERT(fread(p, 3, sizeof(float), fp)); //normals
            STL_ASSERT(fread(p, 3, sizeof(float), fp)); //vertex
            resp[it][0] = p[0];
            resp[it][1] = p[1];
            resp[it][2] = p[2];
            
            it += 1;
            STL_ASSERT(fread(p, 3, sizeof(float), fp)); //vertex
            resp[it][0] = p[0];
            resp[it][1] = p[1];
            resp[it][2] = p[2];
            
            it += 1;
            STL_ASSERT(fread(p, 3, sizeof(float), fp)); //vertex
            resp[it][0] = p[0];
            resp[it][1] = p[1];
            resp[it][2] = p[2];
            
            it += 1;
            STL_ASSERT(fread(dummy, 2, sizeof(char), fp));
        }
        
        *n_triangles = triangles;
        fclose(fp);
        return resp;
        
    }else{
        std::cout << "Could not open file " << path << std::endl;
        return NULL;
    }
}

#endif