#include <shape.h>

__bidevice__ Mesh::Mesh(const Transform &toWorld, int nTris, int *_indices,
                        int nVerts, Point3f *P, vec3f *S, Normal3f *N, 
                        Point2f *UV) : Shape(toWorld)
{
    Set(toWorld, nTris, _indices, nVerts, P, S, N, UV);
    type = ShapeType::MESH;
}

__bidevice__ void Mesh::Set(const Transform &toWorld, int nTris, int *_indices,
                            int nVerts, Point3f *P, vec3f *S, Normal3f *N, 
                            Point2f *UV)
{
    nTriangles = nTris;
    nVertices = nVerts;
    ObjectToWorld = toWorld;
    WorldToObject = Inverse(toWorld);
    
    indices = new int[3 * nTriangles];
    memcpy(indices, _indices, sizeof(int) * 3 * nTriangles);
    
    p = new Point3f[nVertices];
    for(int i = 0; i < nVertices; i++) p[i] = ObjectToWorld(P[i]);
    
    if(UV){
        uv = new Point2f[nVertices];
        memcpy(uv, UV, nVertices * sizeof(Point2f));
        printf(" >> Added %d UVs\n", nVertices);
    }
    
    if(N){
        n = new Normal3f[nVertices];
        for(int i = 0; i < nVertices; i++) n[i] = ObjectToWorld(N[i]);
        printf(" >> Added %d Normals\n", nVertices);
    }
    
    if(S){
        s = new vec3f[nVertices];
        for(int i = 0; i < nVertices; i++) s[i] = ObjectToWorld(S[i]);
        printf(" >> Added %d Tangents\n", nVertices);
    }
    
    type = ShapeType::MESH;
}

__bidevice__ bool Mesh::IntersectMeshNode(Node *node, const Ray &r, 
                                          SurfaceInteraction * isect,
                                          Float *tHit) const
{
    Assert(node->n > 0 && node->is_leaf && node->handles);
    bool hit_anything = false;
    for(int i = 0; i < node->n; i++){
        int nTri = node->handles[i].handle;
        printf("Testing triangle %d\n", nTri);
        //TODO
    }
    
    return hit_anything;
}

//TODO: Refactor this function so that we can re-use code from primitive.cpp
__bidevice__ bool Mesh::Intersect(const Ray &r, Float *tHit,
                                  SurfaceInteraction *isect) const
{
    NodePtr stack[MAX_STACK_SIZE];
    NodePtr *stackPtr = stack;
    *stackPtr++ = NULL;
    
    NodePtr node = bvh;
    SurfaceInteraction tmp;
    int curr_depth = 1;
    int hit_tests = 1;
    bool hit_anything = false;
    
    Float t0, t1;
    bool hit_bound = node->bound.IntersectP(r, &t0, &t1);
    
    if(hit_bound && node->is_leaf){
        hit_tests += node->n;
        return IntersectMeshNode(node, r, isect, tHit);
    }
    
    do{
        if(hit_bound){
            NodePtr childL = node->left;
            NodePtr childR = node->right;
            bool hitl = false;
            bool hitr = false;
            if(childL->n > 0 || childR->n > 0){
                hit_tests += 2;
                hitl = childL->bound.IntersectP(r, &t0, &t1);
                hitr = childR->bound.IntersectP(r, &t0, &t1);
            }
            
            if(hitl && childL->is_leaf){
                hit_tests += childL->n;
                if(IntersectMeshNode(childL, r, &tmp, tHit)){
                    hit_anything = true;
                }
            }
            
            if(hitr && childR->is_leaf){
                hit_tests += childR->n;
                if(IntersectMeshNode(childR, r, &tmp, tHit)){
                    hit_anything = true;
                }
            }
            
            bool transverseL = (hitl && !childL->is_leaf);
            bool transverseR = (hitr && !childR->is_leaf);
            if(!transverseR && !transverseL){
                node = *--stackPtr;
                curr_depth -= 1;
            }else{
                node = (transverseL) ? childL : childR;
                if(transverseL && transverseR){
                    *stackPtr++ = childR;
                    curr_depth += 1;
                }
            }
        }else{
            node = *--stackPtr;
            curr_depth -= 1;
        }
        
        Assert(curr_depth <= MAX_STACK_SIZE-2);
        
        if(node){
            hit_bound = node->bound.IntersectP(r, &t0, &t1);
        }
        
    }while(node != NULL);
    
    if(hit_anything){
        *isect = tmp;
    }
    
    return hit_anything;
}

__bidevice__ Bounds3f Mesh::GetBounds() const{
    if(bvh){
        return bvh->bound;
    }else{
        printf("Called Mesh::GetBounds() on unintialized Mesh!\n");
        return Bounds3f();
    }
}

__global__ void MeshGetBounds(Mesh *mesh){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < mesh->nTriangles){
        PrimitiveHandle *pHandle = &mesh->handles[tid];
        int i0 = mesh->indices[3 * tid + 0];
        int i1 = mesh->indices[3 * tid + 1];
        int i2 = mesh->indices[3 * tid + 2];
        Point3f p0 = mesh->p[i0];
        Point3f p1 = mesh->p[i1];
        Point3f p2 = mesh->p[i2];
        
        Bounds3f bound(p0, p1);
        bound = Union(bound, p2);
        
        pHandle->bound = bound;
        pHandle->handle = tid;
    }
}

__host__ void Mesh::Wrap(){
    handles = cudaAllocateVx(PrimitiveHandle, nTriangles);
    size_t pThreads = 64;
    size_t pBlocks = (nTriangles + pThreads - 1)/pThreads;
    printf("Computing triangles Bounds...");
    MeshGetBounds<<<pBlocks, pThreads>>>(this);
    cudaDeviceAssert();
    printf("OK\nPacking...");
    
    int max_depth = 12;
    int totalNodes = 0;
    bvh = CreateBVH(handles, nTriangles, 0, max_depth, &totalNodes);
    printf("OK [ Build BVH with %d nodes ]\n", totalNodes);
}