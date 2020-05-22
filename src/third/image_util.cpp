#include <image_util.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

unsigned char * ReadImage(const char *path, int &width, int &height, int &channels){
    int nx = 0, ny = 0, nn = 0;
    unsigned char * data = stbi_load(path, &nx, &ny, &nn, 0);
    width = nx;
    height = ny;
    channels = nn;
    return data;
}