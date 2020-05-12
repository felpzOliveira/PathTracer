#pragma once

#include <geometry.h>
#include <curand_kernel.h>

enum ToneMapAlgorithm{
    Reinhard,
    Exponential,
    NaughtyDog
};

struct Pixel{
    curandState state;
    Spectrum we;
    Float u, v;
    
    long hits;
    long misses;
    long samples;
    int max_transverse_tests;
};

struct Image{
    Pixel *pixels;
    int pixels_count;
    int width, height;
};

__host__ Image *CreateImage(int res_x, int res_y);
__host__ void ImageWrite(Image *image, const char *path, Float exposure, 
                         ToneMapAlgorithm algorithm);
__host__ void ImageFree(Image *image);

class Camera{
    public:
    Point3f position;
    Point3f lower_left;
    vec3f horizontal;
    vec3f vertical;
    Float lensRadius;
    vec3f u, v, w;
    
    __bidevice__ Camera(Point3f eye, Point3f at, vec3f up, Float fov, Float aspect);
    
    __bidevice__ Camera(Point3f eye, Point3f at, vec3f up, Float fov, 
                        Float aspect, Float aperture);
    
    __bidevice__ Camera(Point3f eye, Point3f at, vec3f up, Float fov, 
                        Float aspect, Float aperture, Float focus_dist);
    
    __bidevice__ Ray SpawnRay(Float u, Float v);
    __bidevice__ Ray SpawnRay(Float u, Float v, Point2f disk);
};