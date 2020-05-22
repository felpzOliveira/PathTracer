#pragma once
#include <camera.h>

/*
* This function invokes the Graphy library for display so that
* we can do that cool 'see your image being built'. However it is not
* binded in the Toy Tracer, it dynamically loads Graphy from a pre-defined
* path. If it does not find it is OK, rendering is still being performed.
* Because I wanted Toy-Tracer and Graphy to be not linked together the pixel
* buffer being used to render here is not Binded with OpenGL capabilities
* so every time you call this a new texture is generated making this call
* slow. However because we are currently Path Tracing this time is not relevant.
*/
void graphy_display_pixels(Image *image, int count, int filter=1);