#if !defined(MESH_H)
#define MESH_H

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include <stl_loader.h>
#include <fstream>
#include <types.h>
#include <cuda_util.cuh>
#include <transform.h>


struct MeshData{
    std::vector<glm::vec3> vertices;
    std::vector<unsigned int> indices;
    std::vector<glm::vec2> uvs;
};

inline __host__ void _internal_load_obj(const char *path, MeshData *data);


inline __host__ Mesh * load_mesh_stl(const char *path, material_handle mat_handle,
                                     Transforms transforms, int ninstances = 1)
{
    Mesh *mesh = nullptr;
    int tri_size = 0;
    glm::vec3 *vertices = stl_load<glm::vec3>(path, &tri_size);
    if(!vertices || tri_size == 0){
        std::cout << "Wut ? " << std::endl;
        exit(0);
    }
    
    mesh = (Mesh *)cudaAllocOrFail(sizeof(Mesh));
    mesh->triangles = (Triangle *)cudaAllocOrFail(sizeof(Triangle)*tri_size);
    mesh->instances = (Transforms *)cudaAllocOrFail(sizeof(Transforms)*ninstances);
    mesh->instances_it = ninstances;
    
    Object *handles = new Object[tri_size];
    int it = 0;
    glm::vec3 min(FLT_MAX);
    glm::vec3 max(-FLT_MAX);
    glm::vec3 arr[3];
    
    for(int i = 0; i < 3*tri_size; i += 3){
        mesh->triangles[i].has_uvs = false;
        glm::vec3 v0 = vertices[i + 0];
        glm::vec3 v1 = vertices[i + 1];
        glm::vec3 v2 = vertices[i + 2];
        
        //TODO: Take this out of here!
        arr[0] = toWorld(v0, transforms);
        arr[1] = toWorld(v1, transforms);
        arr[2] = toWorld(v2, transforms);
        
        for(int k = 0; k < 3; k += 1){
            for(int j = 0; j < 3; j += 1){
                if(min[j] > arr[k][j]){
                    min[j] = arr[k][j]-0.001f;
                }
                
                if(max[j] < arr[k][j]){
                    max[j] = arr[k][j]+0.001f;
                }
            }
        }
        //////////////////////////////////////
        
        mesh->triangles[it].v0 = arr[0];
        mesh->triangles[it].v1 = arr[1];
        mesh->triangles[it].v2 = arr[2];
        mesh->triangles[it].mat_handle = mat_handle;
        mesh->triangles[it].handle = it;
        
        handles[it].object_type = OBJECT_TRIANGLE;
        handles[it].handle = it;
        handles[it].isvalid = 1;
        handles[it].isbinded = 0;
        
        it ++;
    }
    
    mesh->triangles_it = it;
    aabb_init(&mesh->aabb, min, max, OBJECT_MESH);
    mesh->instances_it = ninstances;
    mesh->instances[0] = transforms;
    mesh->bvh = build_bvh<Mesh>(mesh, handles, it, 0, BVH_MAX_DEPTH);
    
    delete[] handles;
    
    return mesh;
}

inline __host__ Mesh * load_mesh_obj(const char *path, material_handle mat_handle,
                                     Transforms transforms, int ninstances = 1)
{
    Mesh *mesh = nullptr;
    MeshData data;
    _internal_load_obj(path, &data);
    size_t vertices_size = data.vertices.size();
    size_t indices_size = data.indices.size();
    size_t uvs_size = data.uvs.size();
    size_t tri_size = indices_size/3;
    
    if(vertices_size == 0){
        std::cout << "Failed to get vertices" << std::endl;
        exit(0);
    }
    
    if(indices_size == 0){
        std::cout << "Failed to get indices" << std::endl;
        exit(0);
    }
    
    if(ninstances < 1){
        std::cout << "Are you drunk? " << ninstances << " instances?" << std::endl;
        exit(0);
    }
    
    mesh = (Mesh *)cudaAllocOrFail(sizeof(Mesh));
    mesh->triangles = (Triangle *)cudaAllocOrFail(sizeof(Triangle)*tri_size);
    mesh->instances = (Transforms *)cudaAllocOrFail(sizeof(Transforms)*ninstances);
    mesh->instances_it = ninstances;
    
    Object *handles = new Object[tri_size];
    
    int it = 0;
    glm::vec3 min(FLT_MAX);
    glm::vec3 max(-FLT_MAX);
    glm::vec3 arr[3];
    for(size_t i = 0; i < indices_size; i += 3){
        unsigned int i0 = data.indices[i + 0];
        unsigned int i1 = data.indices[i + 1];
        unsigned int i2 = data.indices[i + 2];
        
        glm::vec3 v0 = data.vertices[i0];
        glm::vec3 v1 = data.vertices[i1];
        glm::vec3 v2 = data.vertices[i2];
        
        //TODO: Take this out of here!
        arr[0] = toWorld(v0, transforms);
        arr[1] = toWorld(v1, transforms);
        arr[2] = toWorld(v2, transforms);
        
        for(int k = 0; k < 3; k += 1){
            for(int j = 0; j < 3; j += 1){
                if(min[j] > arr[k][j]){
                    min[j] = arr[k][j]-0.001f;
                }
                
                if(max[j] < arr[k][j]){
                    max[j] = arr[k][j]+0.001f;
                }
            }
        }
        //////////////////////////////////////
        
        mesh->triangles[it].v0 = arr[0];
        mesh->triangles[it].v1 = arr[1];
        mesh->triangles[it].v2 = arr[2];
        mesh->triangles[it].mat_handle = mat_handle;
        mesh->triangles[it].has_uvs = false;
        mesh->triangles[it].handle = it;
        
        handles[it].object_type = OBJECT_TRIANGLE;
        handles[it].handle = it;
        handles[it].isvalid = 1;
        handles[it].isbinded = 0;
        
        it ++;
    }
    
    mesh->triangles_it = it;
    aabb_init(&mesh->aabb, min, max, OBJECT_MESH);
    mesh->instances_it = ninstances;
    mesh->instances[0] = transforms;
    
    std::cout << "Finished mesh generation, building BVH" << std::endl;
    mesh->bvh = build_bvh<Mesh>(mesh, handles, it, 0, BVH_MAX_DEPTH);
    std::cout << "Finished mesh BVH" << std::endl;
    
    delete[] handles;
    
    return mesh;
}

inline __host__ MeshData * LoadMesh(const char *path){
    MeshData *data = new MeshData;
    _internal_load_obj(path, data);
    return data;
}

inline __host__ void _internal_load_obj(const char *path, MeshData *data){
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::ifstream ifs(path);
    std::string err;
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, &ifs);
    if(!err.empty()){
        std::cout << "Failed to load model " << path << " " << err << std::endl;
        exit(0);
    }
    
    std::cout << path << " => " << attrib.vertices.size() << " vertices" << std::endl;
    
    for(size_t idx = 0; idx < attrib.vertices.size()/3; ++idx){
        tinyobj::real_t vx = attrib.vertices[3 * idx + 0];
        tinyobj::real_t vy = attrib.vertices[3 * idx + 1];
        tinyobj::real_t vz = attrib.vertices[3 * idx + 2];
        
        data->vertices.push_back(glm::vec3(vx,vy,vz));
    }
    
    for(size_t idx = 0; idx < attrib.texcoords.size()/2; ++idx){
        tinyobj::real_t tx = attrib.texcoords[2*idx+0];
        tinyobj::real_t ty = attrib.texcoords[2*idx+1];
        data->uvs.push_back(glm::vec2(tx,ty));
    }
    
    for(auto &shape : shapes){
        size_t idx = 0;
        for(size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f){
            const size_t fx = shape.mesh.num_face_vertices[f];
            if(fx == 3){
                data->indices.push_back(shape.mesh.indices[idx + 0].vertex_index);
                data->indices.push_back(shape.mesh.indices[idx + 1].vertex_index);
                data->indices.push_back(shape.mesh.indices[idx + 2].vertex_index);
            }else{
                std::cout << "NOT A TRIANGLE MESH!" << std::endl;
                exit(0);
            }
            
            idx += fx;
        }
    }
}



#endif