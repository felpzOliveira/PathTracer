#pragma once

#include <geometry.h>
#include <curand_kernel.h>

enum ToneMapAlgorithm{
    Reinhard,
    Exponential,
    NaughtyDog
};

struct PixelStats{
    long hits;
    long misses;
    long mediumHits;
    long zeroRadiancePaths;
    long totalPaths;
    long lightHitFromMedium;
    long lightHits;
    int max_transverse_tests;
};

struct Pixel{
    curandState state;
    Spectrum we;
    Float u, v;
    double accWeight;
    long samples;
    PixelStats stats;
};

struct Image{
    Pixel *pixels;
    int pixels_count;
    int width, height;
};

__host__ Image *CreateImage(int res_x, int res_y);
__host__ void ImageWrite(Image *image, const char *path, Float exposure, 
                         ToneMapAlgorithm algorithm);
__host__ void ImageStats(Image *image);
__host__ void ImageFree(Image *image);

__bidevice__ Spectrum ReinhardMap(Spectrum value, Float exposure);
__bidevice__ Spectrum NaughtyDogMap(Spectrum value, Float exposure);
__bidevice__ Spectrum ExponentialMap(Spectrum value, Float exposure);
__bidevice__ vec3i GetPixelRGB(Pixel *pixel, Float exposure, ToneMapAlgorithm algo);

class Camera{
    public:
    Point3f position;
    Point3f lower_left;
    vec3f horizontal;
    vec3f vertical;
    Float lensRadius;
    vec3f u, v, w;
    Medium *medium;
    
    __bidevice__ Camera() : medium(nullptr){}
    __bidevice__ Camera(Point3f eye, Point3f at, vec3f up, Float fov, Float aspect);
    
    __bidevice__ Camera(Point3f eye, Point3f at, vec3f up, Float fov, 
                        Float aspect, Float aperture);
    
    __bidevice__ Camera(Point3f eye, Point3f at, vec3f up, Float fov, 
                        Float aspect, Float aperture, Float focus_dist);
    
    __bidevice__ void SetMedium(Medium *m){ medium = m; }
    
    __bidevice__ void Config(Point3f eye, Point3f at, vec3f up, Float fov, Float aspect);
    
    __bidevice__ void Config(Point3f eye, Point3f at, vec3f up, Float fov, 
                             Float aspect, Float aperture);
    
    __bidevice__ void Config(Point3f eye, Point3f at, vec3f up, Float fov, 
                             Float aspect, Float aperture, Float focus_dist);
    
    __bidevice__ Ray SpawnRay(Float u, Float v);
    __bidevice__ Ray SpawnRay(Float u, Float v, Point2f disk);
};