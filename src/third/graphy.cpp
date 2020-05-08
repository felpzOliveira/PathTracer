#include <graphy.h>
#include <gr_display.hpp>
#include <gr_opengl.hpp>
#include <dlfcn.h>
#include <iostream>

#define GraphyPath "/home/felpz/Documents/Graphics/build/libgraphy.so"

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

void graphy_initialize(Image *image){
    if(!display){
        GraphyHandle = dlopen(GraphyPath, RTLD_LAZY);
        graphy_ok = -1;
        if(!GraphyHandle){
            std::cout << "Failed to get Graphy library pointer" << std::endl;
        }else{
            graphy_get_display = (GraphyGetDisplay) LoadSymbol(GraphyHandle, "_Z14gr_new_displayii");
            graphy_render_pixels = (GraphyRenderPixels) LoadSymbol(GraphyHandle, "_Z23gr_opengl_render_pixelsPfiiP12gr_display_t");
            
            if(graphy_get_display && graphy_render_pixels){
                display = graphy_get_display(image->width, image->height);
                vals = new float[image->pixels_count * 3];
                graphy_ok = 1;
            }else{
                std::cout << "Failed to load Graphy symbols" << std::endl;
            }
        }
    }
}

void graphy_display_pixels(Image *image){
    if(graphy_ok == 0) graphy_initialize(image);
    
    if(graphy_ok > 0){
        int it = 0;
        for(int k = 0; k < image->pixels_count; k++){
            Pixel *pixel = &image->pixels[k];
            Float invNS = 1.0f / (Float)(pixel->samples);
            Spectrum we = pixel->we * invNS;
            vals[it++] = we[0]; vals[it++] = we[1]; vals[it++] = we[2];
        }
        
        graphy_render_pixels(vals, image->width, image->height, display);
    }
}