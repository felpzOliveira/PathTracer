#include <ppm.h>

bool PPMWriteFloat(float *values, int width, int height,
                   const char *path, Process handler)
{
    bool rv = false;
    FILE *fp = fopen(path, "wb");
    if(fp){
        (void)fprintf(fp, "P6\n%d %d\n255\n", width, height);
        for(int j = height-1; j >= 0; --j){
            for(int i = 0; i < width; ++i){
                size_t pidx = j*3*width + i*3;
                float pixel[3] = {values[pidx+0], values[pidx+1], values[pidx+2]};
                float fcolor[3];
                unsigned char color[3];
                handler(pixel, &fcolor[0]);
                
                color[0] = static_cast<int>(255.999 * fcolor[0]);
                color[1] = static_cast<int>(255.999 * fcolor[1]);
                color[2] = static_cast<int>(255.999 * fcolor[2]);
                
                (void)fwrite(color, 1, 3, fp);
            }
        }
        
        (void)fclose(fp);
        rv = true;
    }
    
    return rv;
}

bool PPMRead(const char *path, unsigned char **values, int &width, int &height){
    bool rv = false;
    FILE *fp = fopen(path, "rb");
    if(fp){
        char *line = nullptr;
        size_t len = 0;
        (void)getline(&line, &len, fp);
        (void)getline(&line, &len, fp);
        sscanf(line, "%d %d", &width, &height);
        (void)getline(&line, &len, fp);
        
        unsigned char *pixels = new unsigned char[width*height*3];
        if(!pixels){
            (void)fclose(fp);
            return rv;
        }
        
        for(int j = height-1; j >= 0; --j){
            for(int i = 0; i < width; ++i){
                unsigned char color[3];
                size_t pidx = j*3*width + i*3;
                (void)fread(color, 1, 3, fp);
                pixels[pidx+0] = color[0];
                pixels[pidx+1] = color[1];
                pixels[pidx+2] = color[2];
            }
        }
        
        *values = pixels;
        (void)fclose(fp);
        rv = true;
    }
    
    return rv;
}

bool PPMReadFloat(const char *path, float **values, int &width, int &height){
    bool rv = false;
    FILE *fp = fopen(path, "rb");
    if(fp){
        char *line = nullptr;
        size_t len = 0;
        (void)getline(&line, &len, fp);
        (void)getline(&line, &len, fp);
        sscanf(line, "%d %d", &width, &height);
        (void)getline(&line, &len, fp);
        
        float *pixels = new float[width*height*3];
        if(!pixels){
            (void)fclose(fp);
            return rv;
        }
        
        for(int j = height-1; j >= 0; --j){
            for(int i = 0; i < width; ++i){
                unsigned char color[3];
                size_t pidx = j*3*width + i*3;
                (void)fread(color, 1, 3, fp);
                pixels[pidx+0] = static_cast<float>(((int)color[0]))/255.f;
                pixels[pidx+1] = static_cast<float>(((int)color[1]))/255.f;
                pixels[pidx+2] = static_cast<float>(((int)color[2]))/255.f;
            }
        }
        
        *values = pixels;
        (void)fclose(fp);
        rv = true;
    }
    
    return rv;
}