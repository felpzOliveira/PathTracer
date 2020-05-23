#pragma once
#include <vector>
#include <map>
#include <string>
#include <geometry.h>

typedef struct{
    std::string mtlname;
    std::map<std::string, Spectrum> data;
}MTL;

/*
* Because we are doing materials with BSDFs we can't really use
* the MTL format for coloring since most of its equations are
* not physically based, but this will at least *guide* us on
* choosing a material for a mesh. I'm also not gonna parse everything
* as I doubt I'll need it.
* NOTE: All pointers allocated by these functions are in CPU.
*/

/*
* Parses a mtl file and generates a list of materials and its properties.
*/
bool MTLParse(const char *path, std::vector<MTL *> *materials);

/*
* Gets a Spectrum for a given properties inside a material, return black
* Spectrum in case it does not exist.
*/
Spectrum MTLGetSpectrum(const char *property, MTL *mtl);

/*
* Find a material index inside a already parsed mtl.
*/
int MTLFindMaterial(const char *name, std::vector<MTL *> *materials);