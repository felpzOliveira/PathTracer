#pragma once
#include <cutil.h>
#include <geometry.h>
#include <mtl.h>
#include <vector>
#include <shape.h>

typedef struct{
    std::string file;
    std::string name;
}MeshMtl;

/*TODO: Not working */
__host__ std::vector<ParsedMesh*> *LoadObj(const char *path, std::vector<MeshMtl> *mtls);