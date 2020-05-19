#pragma once
#include <stdio.h>
#include <stdlib.h>

typedef void(*Process)(float pixel[3], float *out);

bool PPMWriteFloat(float *values, int width, int height,
                   const char *path, Process handler);

bool PPMReadFloat(const char *path, float **values, int &width, int &height);
bool PPMRead(const char *path, unsigned char **values, int &width, int &height);

