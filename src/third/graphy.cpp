#include <graphy.h>
#include <gr_display.hpp>
#include <gr_opengl.hpp>
#include <dlfcn.h>
#include <iostream>
#include <thread>
#include <camera.h>
#include <ppm.h>

#define GraphyPath "/home/felpz/Documents/Graphics/build/libgraphy.so"
#define BACKUP_SAVE "temp.ppm"
#define BACKUP_AT_EACH 10
#define USE_THREADS 0

static float *vals;
static gr_display *display = nullptr;

typedef gr_display*(*GraphyGetDisplay)(int, int);
typedef void(*GraphyRenderPixels)(float *, int, int, gr_display *);

void *GraphyHandle = nullptr;
GraphyGetDisplay graphy_get_display;
GraphyRenderPixels graphy_render_pixels;

static int graphy_ok = 0;

void * LoadSymbol(void *handle, const char *name){
    void *ptr = dlsym(handle, name);
    if(!ptr){
        std::cout << "Failed to load symbol " << name << " [ " << 
            dlerror() << " ]" << std::endl;
    }
    
    return ptr;
}

void graphy_initialize(int width, int height){
    if(!display){
        GraphyHandle = dlopen(GraphyPath, RTLD_LAZY);
        graphy_ok = -1;
        if(!GraphyHandle){
            std::cout << "Failed to get Graphy library pointer" << std::endl;
        }else{
            graphy_get_display = (GraphyGetDisplay) LoadSymbol(GraphyHandle, "_Z14gr_new_displayii");
            graphy_render_pixels = (GraphyRenderPixels) LoadSymbol(GraphyHandle, "_Z23gr_opengl_render_pixelsPfiiP12gr_display_t");
            
            if(graphy_get_display && graphy_render_pixels){
                display = graphy_get_display(width, height);
                vals = new float[width * height * 3];
                graphy_ok = 1;
            }else{
                std::cout << "Failed to load Graphy symbols" << std::endl;
            }
        }
    }
}

void ProcessWritePixel(float pixel[3], float *out){
    out[0] = pixel[0]; out[1] = pixel[1]; out[2] = pixel[2];
}

void threaded_save_tmp_image(int width, int height){
    if(!PPMWriteFloat(vals, width, height, BACKUP_SAVE, ProcessWritePixel))
        printf(" * Failed to save backup\n");
}

void graphy_display_pixels(Image *image, int count, int filter){
    if(graphy_ok == 0) graphy_initialize(image->width, image->height);
    
    int it = 0;
    int save = ((count % BACKUP_AT_EACH) == 0 && count > 0) ? 1 : 0;
    if(save || graphy_ok > 0){
        for(int k = 0; k < image->pixels_count; k++){
            Pixel *pixel = &image->pixels[k];
            Spectrum we = pixel->we;
            if(filter){
                AssertA(pixel->samples != 0, "Zero samples on graphy_display");
                Float invNS = 1.0f / (Float)(pixel->samples);
                we = ExponentialMap(pixel->we * invNS, 1.f);
            }
            
            vals[it++] = we[0]; vals[it++] = we[1]; vals[it++] = we[2];
        }
        
        if(save){
            if(USE_THREADS){
                std::thread(threaded_save_tmp_image, image->width, image->height).detach();
            }else{
                threaded_save_tmp_image(image->width, image->height);
            }
        }
        
        if(graphy_ok > 0){
            graphy_render_pixels(vals, image->width, image->height, display);
        }
    }
}

void graphy_display_pixels(Spectrum *pixels, int width, int height){
    if(graphy_ok == 0) graphy_initialize(width, height);
    int it = 0;
    if(graphy_ok > 0){
        for(int k = 0; k < width * height; k++){
            Spectrum we = pixels[k];
            we = ExponentialMap(we, 1.f);
            vals[it++] = we[0]; vals[it++] = we[1]; vals[it++] = we[2];
        }
        
        if(graphy_ok > 0){
            graphy_render_pixels(vals, width, height, display);
        }
    }
}