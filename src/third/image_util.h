#pragma once
#include <geometry.h>

/*
* I'm not really gonna add a image util files, this is just
* so we don't have to recompile stb_image all the time.
*/

unsigned char *ReadImage(const char *path, int &width, int &height, int &channels);

Spectrum *ReadImageEXR(const char *path, int &width, int &height);