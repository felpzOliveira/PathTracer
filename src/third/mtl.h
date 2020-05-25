#pragma once
#include <vector>
#include <map>
#include <string>
#include <geometry.h>

typedef struct{
    std::string file;
    std::string name;
}MeshMtl;

typedef struct{
    std::string mtlname;
    std::string basedir;
    std::map<std::string, std::vector<std::string>> data;
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
bool MTLParse(const char *path, std::vector<MTL *> *materials, const char *basedir);

/*
* Gets a Spectrum for a given property inside a material, return black
* Spectrum in case it does not exist.
*/
Spectrum MTLGetSpectrum(const char *property, MTL *mtl, bool &found);

/*
* Gets a raw (string) value of a content, most for textures.
* NOTE: If the property chosen is a number/Spectrum a string is returned in form:
*      a b c
*/
std::string MTLGetValue(const char *property, MTL *mtl, bool &found);

/*
* Find a material index inside a already parsed mtl.
*/
MTL *MTLFindMaterial(const char *name, std::vector<MTL *> *materials);

/*
* Parse a sequence of mtl descriptors from a obj loaded from the obj_loader API.
*/
bool MTLParseAll(std::vector<MTL *> *materials, std::vector<MeshMtl> *desc,
                 const char *basedir);

/*
* Attempts to parse a double value from string s, return end point.
* Exposing to use in obj loader as well.
*/
bool ParseDouble(const char *s, const char *s_end, Float *result);