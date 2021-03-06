#include <camera.h>
#include <cutil.h>
#include <miniz.h>
#include <iostream>

__bidevice__ Spectrum ReinhardMap(Spectrum value, Float exposure){
    (void)exposure;
    value = (value / (value + 1.f));
    value = GammaCorrect(value);
    return value;
}

__bidevice__ Spectrum NaughtyDogMap(Spectrum value, Float exposure){
    float A = 0.15f;
    float B = 0.50f;
    float C = 0.10f;
    float D = 0.20f;
    float E = 0.02f;
    float F = 0.30f;
    float W = 11.2f;
    value *= exposure;
    value = ((value * (A*value+C*B)+D*E)/(value*(A*value+B)+D*F))-E/F;
    float white = ((W*(A*W+C*B)+D*E)/(W*(A*W+B)+D*F))-E/F;
    value = value / white;
    value = GammaCorrect(value);
    return value;
}

__bidevice__ Spectrum ExponentialMap(Spectrum value, Float exposure){
    value = (Spectrum(1.f) - Exp(-value * exposure));
    value = GammaCorrect(value);
    return value;
}


__bidevice__ vec3i GetPixelRGB(Pixel *pixel, Float exposure, ToneMapAlgorithm algo){
    Spectrum e = pixel->we;
    AssertA(!IsZero(pixel->accWeight), "Zero accWeight");
    Float invNs = 1.0f / pixel->accWeight;
    e *= invNs;
    switch(algo){
        case Reinhard: e = ReinhardMap(e, exposure); break;
        case Exponential: e = ExponentialMap(e, exposure); break;
        case NaughtyDog: e = NaughtyDogMap(e, exposure); break;
        default:{
            printf("Unknown Tone Mapping algorithm!\n");
        }
    }
    
    return vec3i((int)(255.999 * e.x), (int)(255.999 * e.y), (int)(255.999 * e.z));
}

__host__ void ImageWrite(Image *image, const char *path, Float exposure, 
                         ToneMapAlgorithm algorithm)
{
    int pixels_count = image->pixels_count;
    int width = image->width;
    int height = image->height;
    int size = pixels_count * 3;
    unsigned char *data = new unsigned char[size];
    size_t png_data_size = 0;
    void *png_data = nullptr;
    int it = 0;
    
    for(int i = 0; i < pixels_count; i += 1){
        vec3i v = GetPixelRGB(&image->pixels[i], exposure, algorithm);
        data[it++] = v.x; data[it++] = v.y; data[it++] = v.z;
    }   
    
    png_data = tdefl_write_image_to_png_file_in_memory_ex(data, width, height, 3,
                                                          &png_data_size, 6, MZ_TRUE);
    
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

__host__ void ImageStats(Image *image){
    long zeroRadiance = 0;
    long totalPaths = 0;
    long totalHits = 0;
    long mediumHits = 0;
    long surfaceHits = 0;
    long misses = 0;
    long lightHits = 0;
    long lHitsMedium = 0;
    int pixels_count = image->pixels_count;
    for(int i = 0; i < pixels_count; i++){
        zeroRadiance += image->pixels[i].stats.zeroRadiancePaths;
        totalPaths   += image->pixels[i].stats.totalPaths;
        totalHits    += image->pixels[i].stats.mediumHits + image->pixels[i].stats.hits;
        surfaceHits  += image->pixels[i].stats.hits;
        mediumHits   += image->pixels[i].stats.mediumHits;
        misses       += image->pixels[i].stats.misses;
        lightHits    += image->pixels[i].stats.lightHits;
        lHitsMedium  += image->pixels[i].stats.lightHitFromMedium;
    }
    
    double zeroPathsPct = static_cast<double>(zeroRadiance) / static_cast<double>(totalPaths);
    double mediumHitsPct = static_cast<double>(mediumHits) / static_cast<double>(totalHits);
    double surfaceHitsPct = static_cast<double>(surfaceHits) / static_cast<double>(totalHits);
    double missesPct = static_cast<double>(misses) / static_cast<double>(totalPaths);
    double lightHitsPct = static_cast<double>(lightHits) / static_cast<double>(totalHits);
    double lightHitsMedium = static_cast<double>(lHitsMedium) / static_cast<double>(lightHits);
    
    zeroPathsPct *= 100; mediumHitsPct *= 100;
    surfaceHits *= 100; missesPct *= 100;
    lightHitsPct *= 100; lightHitsMedium *= 100;
    
    printf(" * Rendered using %ld samples\n", image->pixels[0].samples);
    printf(" * Zero radiance paths: %g%%\n", zeroPathsPct);
    printf(" * Medium hits: %g%%\n", mediumHitsPct);
    printf(" * Surface hits: %g%%\n", surfaceHitsPct);
    printf(" * Light hits: %g%%\n", lightHitsPct);
    printf(" * Light hits from medium: %g%%\n", lightHitsMedium);
    printf(" * Misses: %g%%\n", missesPct);
}

__host__ void ImageFree(Image *image){
    if(image){
        if(image->pixels) cudaFree(image->pixels);
        cudaFree(image);
    }
}

__host__ Image *CreateImage(int res_x, int res_y){
    int pixel_count = res_x * res_y;
    Image *image = (Image *)cudaAllocate(sizeof(Image));
    image->pixels = cudaAllocateVx(Pixel, pixel_count);
    image->width = res_x;
    image->height = res_y;
    image->pixels_count = pixel_count;
    
    return image;
}

__bidevice__ void Camera::Config(Point3f eye, Point3f at, vec3f up, Float fov, 
                                 Float aspect, Float aperture, Float focus_dist)
{
    position = eye;
    lensRadius = aperture / 2;
    Float theta = Radians(fov);
    Float half_height = tan(theta/2.f);
    Float half_width  = aspect * half_height;
    w = Normalize(eye - at);
    u = Normalize(Cross(up, w));
    v = Normalize(Cross(w, u));
    
    lower_left = position - half_width * focus_dist * u - 
        half_height * focus_dist * v - focus_dist * w;
    
    horizontal = 2.f * half_width * focus_dist * u;
    vertical   = 2.f * half_height * focus_dist * v;
}

__bidevice__ void Camera::Config(Point3f eye, Point3f at, vec3f up, Float fov, 
                                 Float aspect, Float aperture)
{
    Float focus_dist = (eye - at).Length();
    position = eye;
    lensRadius = aperture / 2;
    Float theta = Radians(fov);
    Float half_height = tan(theta/2.f);
    Float half_width  = aspect * half_height;
    w = Normalize(eye - at);
    u = Normalize(Cross(up, w));
    v = Normalize(Cross(w, u));
    
    lower_left = position - half_width * focus_dist * u - 
        half_height * focus_dist * v - focus_dist * w;
    
    horizontal = 2.f * half_width * focus_dist * u;
    vertical   = 2.f * half_height * focus_dist * v;
}

__bidevice__ void Camera::Config(Point3f eye, Point3f at, vec3f up, Float fov, Float aspect){
    position = eye;
    Float theta = Radians(fov);
    Float half_height = tan(theta/2.f);
    Float half_width  = aspect * half_height;
    lensRadius = 0;
    w = Normalize(eye - at);
    u = Normalize(Cross(up, w));
    v = Normalize(Cross(w, u));
    
    lower_left = position - half_width * u - half_height * v - w;
    horizontal = 2.f * half_width * u;
    vertical   = 2.f * half_height * v;
}

__bidevice__ Camera::Camera(Point3f eye, Point3f at, vec3f up, Float fov, 
                            Float aspect, Float aperture, Float focus_dist)
:medium(nullptr)
{
    Config(eye, at, up, fov, aspect, aperture, focus_dist);
}

__bidevice__ Camera::Camera(Point3f eye, Point3f at, vec3f up, Float fov, 
                            Float aspect, Float aperture)
:medium(nullptr)
{
    Config(eye, at, up, fov, aspect, aperture);
}

__bidevice__ Camera::Camera(Point3f eye, Point3f at, vec3f up, Float fov, Float aspect)
:medium(nullptr)
{
    Config(eye, at, up, fov, aspect);
}

__bidevice__ Ray Camera::SpawnRay(Float s, Float t, Point2f disk){
    Point2f rd = lensRadius * disk;
    vec3f offset = u * rd.x + v * rd.y;
    vec3f d = lower_left + s * horizontal + t * vertical - position - offset;
    return Ray(position + offset, Normalize(d), Infinity, 0, medium);
}

__bidevice__ Ray Camera::SpawnRay(Float s, Float t){
    vec3f d = lower_left + s * horizontal + t * vertical - position;
    return Ray(position, Normalize(d), Infinity, 0, medium);
}
