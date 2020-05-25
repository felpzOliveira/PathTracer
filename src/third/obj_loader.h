#pragma once
#include <cutil.h>
#include <geometry.h>
#include <mtl.h>
#include <vector>
#include <shape.h>

/*
* NOTE: Heavily based on tiny_obj_loader. I'm basically splitting the mesh
* into multiple objects, I don't like the way tiny_obj_loader does it,
* makes it very hard to fragment the mesh. I need a API that can actually
* return a series of meshes for a single obj file if needed. The memory 
* returned is GPU ready for the interior pointers of the std::vector object,
* meaning it can be directly passed to CUDA without any memory transfers.
*/
__host__ std::vector<ParsedMesh*> *LoadObj(const char *path, std::vector<MeshMtl> *mtls,
                                           bool split_mesh=true);

/*
* If you want to simply load the mesh and not split and not get any materials
* and do your own thing, call this. It will be faster and use less memory. Memory
* returned is already GPU ready for mesh binding operations.
*/
__host__ ParsedMesh *LoadObjOnly(const char *path);