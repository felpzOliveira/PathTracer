#include <shape.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include <fstream>
#include <vector>

//TODO: We might want to write our own obj parser so we don't 
//      have to waste time with data type translation


__host__ bool LoadObjData(const char *obj, ParsedMesh **data){
    bool rv = false;
    Point3f center(0.f);
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> tshapes;
    std::vector<tinyobj::material_t> materials;
    std::ifstream ifs(obj);
    std::string err;
    printf("Attempting to load %s...", obj);
    bool ret = tinyobj::LoadObj(&attrib, &tshapes, &materials, &err, &ifs);
    
    if(ret && err.empty()){
        ParsedMesh *mesh = cudaAllocateVx(ParsedMesh, 1);
        mesh->nVertices = attrib.vertices.size()/3;
        mesh->p = cudaAllocateVx(Point3f, mesh->nVertices);
        
        int it = 0;
        int hasUv = 0;
        size_t size = 0;
        if(attrib.texcoords.size() > 0){
            if(attrib.texcoords.size()/2 == mesh->nVertices){
                mesh->uv = cudaAllocateVx(Point2f, mesh->nVertices);
                hasUv = 1;
                printf("\n * [ Adding UVs ]");
            }else{
                printf("\n * [ Skipping UVs - multi-tex ]");
            }
        }
        
        //TODO:Get normals and tangents
        mesh->s = nullptr;
        mesh->n = nullptr;
        
        for(size_t idx = 0; idx < mesh->nVertices; ++idx){
            tinyobj::real_t vx = attrib.vertices[3 * idx + 0];
            tinyobj::real_t vy = attrib.vertices[3 * idx + 1];
            tinyobj::real_t vz = attrib.vertices[3 * idx + 2];
            mesh->p[idx] = Point3f(vx, vy, vz);
            
            center += mesh->p[idx];
            
            if(hasUv){
                tinyobj::real_t uvx = attrib.texcoords[2 * idx + 0];
                tinyobj::real_t uvy = attrib.texcoords[2 * idx + 1];
                mesh->uv[idx] = Point2f(uvx, uvy);
            }
        }
        
        center = center / (Float)mesh->nVertices;
        
        printf("\n * [ Found %ld shapes ]", tshapes.size());
        for(auto &tshape : tshapes){
            for(size_t f = 0; f < tshape.mesh.num_face_vertices.size(); ++f){
                const size_t fx = tshape.mesh.num_face_vertices[f];
                if(fx != 3){
                    printf("\n * Not a triangular mesh\n");
                    cudaSafeExit();
                }
                
                size += fx;
            }
        }
        
        printf("\n * [ Found %ld vertices ( %ld triangles ) ]", size, size/3);
        mesh->indices = cudaAllocateVx(int, size);
        mesh->nTriangles = 0;
        it = 0;
        for(auto &tshape : tshapes){
            size_t idx = 0;
            for(size_t f = 0; f < tshape.mesh.num_face_vertices.size(); ++f){
                mesh->indices[it++] = tshape.mesh.indices[idx+0].vertex_index;
                mesh->indices[it++] = tshape.mesh.indices[idx+1].vertex_index;
                mesh->indices[it++] = tshape.mesh.indices[idx+2].vertex_index;
                mesh->nTriangles += 1;
                idx += 3;
            }
        }
        
        Assert(mesh->nTriangles == size/3);
        
        *data = mesh;
        printf("\n * OK. Mesh has " v3fA(center) "\n", v3aA(center));
        rv = true;
    }else{
        printf("\n * Fail [ %s ]\n", err.c_str());
    }
    
    return rv;
}