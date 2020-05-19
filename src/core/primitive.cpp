#include <primitive.h>
#include <material.h>
#include <util.h>
#include <camera.h>
#include <light.h>

__bidevice__ bool PrimitiveIntersect(const Primitive *primitive, const Ray &ray,
                                     SurfaceInteraction *isect)
{
    Float tHit;
    if(!primitive->shape->Intersect(ray, &tHit, isect)) return false;
    ray.tMax = tHit;
    isect->primitive = primitive;
    return true;
}

__bidevice__ Primitive::Primitive(Shape *shape) : shape(shape){}

__bidevice__ bool Primitive::Intersect(const Ray &ray, SurfaceInteraction *isect) const{
    return PrimitiveIntersect(this, ray, isect);
}

__bidevice__ GeometricEmitterPrimitive::GeometricEmitterPrimitive(Shape *_shape, 
                                                                  Spectrum _L, Float power)
: Primitive(_shape), L(_L), power(power) {}

__bidevice__ GeometricPrimitive::GeometricPrimitive(Shape *_shape, Material *material)
: Primitive(_shape), material(material) {}


__bidevice__ void GeometricEmitterPrimitive::ComputeScatteringFunctions(BSDF *bsdf, 
                                                                        SurfaceInteraction *si,
                                                                        TransportMode mode, 
                                                                        bool mLobes) const
{
    BxDF bxdf; bxdf.Invalidate();
    bsdf->Push(&bxdf);
}

__bidevice__ void GeometricPrimitive::ComputeScatteringFunctions(BSDF *bsdf, 
                                                                 SurfaceInteraction *si,
                                                                 TransportMode mode, 
                                                                 bool mLobes) const
{
    if(material){
        material->ComputeScatteringFunctions(bsdf, si, mode, mLobes);
    }
}

__bidevice__ Aggregator::Aggregator(){
    length = 0;
    head = 0;
    nMeshes = 0;
    nAllowedMeshes = 0;
    lightCounter = 0;
}

__host__ void Aggregator::ReserveMeshes(int n){
    nAllowedMeshes = n;
    meshPtrs = cudaAllocateVx(Mesh*, n);
    nMeshes = 0;
}

__bidevice__ Mesh *Aggregator::AddMesh(const Transform &toWorld, ParsedMesh *pMesh, int copy){
    Mesh *ptr = nullptr;
    if(nMeshes < nAllowedMeshes){
        meshPtrs[nMeshes] = new Mesh(toWorld, pMesh, copy);
        ptr = meshPtrs[nMeshes];
        nMeshes ++;
    }else{
        printf("Hit maximum meshes allowed [%d] \n", nMeshes);
    }
    
    return ptr;
}

__bidevice__ Mesh *Aggregator::AddMesh(const Transform &toWorld, int nTris, int *_indices,
                                       int nVerts, Point3f *P, vec3f *S, Normal3f *N, 
                                       Point2f *UV)
{
    Mesh *ptr = nullptr;
    if(nMeshes < nAllowedMeshes){
        meshPtrs[nMeshes] = new Mesh(toWorld, nTris, _indices, nVerts, P, S, N, UV);
        ptr = meshPtrs[nMeshes];
        nMeshes ++;
    }else{
        printf("Hit maximum meshes allowed [%d] \n", nMeshes);
    }
    
    return ptr;
}

__bidevice__ void Aggregator::Reserve(int size){
    Assert(size > 0);
    length = size;
    primitives = new Primitive*[size];
}

__bidevice__ void Aggregator::Insert(Primitive *pri, int is_light){
    Assert(head < length && primitives);
    primitives[head] = pri;
    if(is_light){
        Assert(lightCounter < 256);
        lightList[lightCounter++] = head;
    }
    
    head++;
}

__bidevice__ void Aggregator::SetLights(){
    if(lightCounter > 0){
        lights = new DiffuseAreaLight*[lightCounter];
        for(int i = 0; i < lightCounter; i++){
            Primitive *pri = primitives[lightList[i]];
            GeometricEmitterPrimitive *gPri = (GeometricEmitterPrimitive *)pri;
            
            lights[i] = new DiffuseAreaLight(gPri->shape->ObjectToWorld, 
                                             gPri->L, 1, gPri->shape);
        }
        
        printf(" * Created %d DiffuseLights\n", lightCounter);
    }
}

__bidevice__ bool Aggregator::IntersectNode(Node *node, const Ray &r, 
                                            SurfaceInteraction * isect) const
{
    Assert(node->n > 0 && node->is_leaf && node->handles);
    bool hit_anything = false;
    for(int i = 0; i < node->n; i++){
        Primitive *pri = primitives[node->handles[i].handle];
        hit_anything |= pri->Intersect(r, isect);
    }
    
    return hit_anything;
}

__bidevice__ bool Aggregator::Intersect(const Ray &r, SurfaceInteraction *isect, 
                                        Pixel *pixel) const
{
    NodePtr stack[MAX_STACK_SIZE];
    NodePtr *stackPtr = stack;
    *stackPtr++ = NULL;
    
    NodePtr node = root;
    SurfaceInteraction tmp;
    int curr_depth = 1;
    int hit_tests = 1;
    bool hit_anything = false;
    
    Float t0, t1;
    bool hit_bound = node->bound.IntersectP(r, &t0, &t1);
    
    if(hit_bound && node->is_leaf){
        hit_tests += node->n;
        if(pixel){
            if(hit_tests > pixel->max_transverse_tests) 
                pixel->max_transverse_tests = hit_tests;
        }
        
        return IntersectNode(node, r, isect);
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
                if(IntersectNode(childL, r, &tmp)){
                    hit_anything = true;
                }
            }
            
            if(hitr && childR->is_leaf){
                hit_tests += childR->n;
                if(IntersectNode(childR, r, &tmp)){
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
            hit_tests += 1;
            hit_bound = node->bound.IntersectP(r, &t0, &t1);
        }
        
    }while(node != NULL);
    
    if(hit_anything){
        *isect = tmp;
    }
    
    if(pixel){
        if(pixel->max_transverse_tests < hit_tests)
            pixel->max_transverse_tests = hit_tests;
    }
    
    return hit_anything;
}

__bidevice__ void Aggregator::Release(){
    for(int i = 0; i < head; i++){
        Primitive *pri = primitives[i];
        pri->Release();
    }
    
    if(lightCounter > 0){
        for(int i = 0; i < lightCounter; i++){
            delete lights[i];
        }
        
        delete[] lights;
    }
    
    delete[] primitives;
}

__bidevice__ void Aggregator::PrintHandle(int which){
    if(which >= 0 && which < head){
        PrimitiveHandle *pri = &handles[which];
        pri->bound.PrintSelf();
        printf("[ %d ]\n", pri->handle);
    }else{
        printf("Current handles:\n");
        for(int i = 0; i < head; i++){
            PrimitiveHandle *pri = &handles[i];
            pri->bound.PrintSelf();
            printf("[ %d ]\n", pri->handle);
        }
    }
}

__bidevice__ int CompareX(PrimitiveHandle *p0, PrimitiveHandle *p1){
    return p0->bound.pMin.x >= p1->bound.pMin.x ? 1 : 0;
}

__bidevice__ int CompareY(PrimitiveHandle *p0, PrimitiveHandle *p1){
    return p0->bound.pMin.y >= p1->bound.pMin.y ? 1 : 0;
}

__bidevice__ int CompareZ(PrimitiveHandle *p0, PrimitiveHandle *p1){
    return p0->bound.pMin.z >= p1->bound.pMin.z ? 1 : 0;
}

__host__ Node *CreateNode(int n){
    Node *node = cudaAllocateVx(Node, 1);
    node->left = nullptr;
    node->right = nullptr;
    node->handles = nullptr;
    node->n = n;
    node->is_leaf = 0;
    return node;
}

__host__ void NodeSetItens(Node *node, int n){
    node->n = n;
    node->handles = cudaAllocateVx(PrimitiveHandle, n);
}

__host__ Float rand_float(){
    return rand() / (RAND_MAX + 1.f);
}

__host__ Node *CreateBVH(PrimitiveHandle *handles,int n, int depth, 
                         int max_depth, int *totalNodes)
{
    int axis = int(3 * rand_float());
    Node *node = CreateNode(n);
    (*totalNodes)++;
    if(axis == 0){
        QuickSort(handles, n, CompareX);
    }else if(axis == 1){
        QuickSort(handles, n, CompareY);
    }else if(axis == 2){
        QuickSort(handles, n, CompareZ);
    }
    
    if(n == 1){
        NodeSetItens(node, n);
        memcpy(node->handles, handles, n * sizeof(PrimitiveHandle));
        node->bound = handles[0].bound;
        node->is_leaf = 1;
        return node;
    }else if(n == 2){
        node->left = CreateNode(0);
        node->right = CreateNode(0);
        (*totalNodes) += 2;
        NodeSetItens(node->left, 1);
        NodeSetItens(node->right, 1);
        memcpy(node->left->handles, &handles[0], sizeof(PrimitiveHandle));
        memcpy(node->right->handles, &handles[1], sizeof(PrimitiveHandle));
        node->left->is_leaf = 1;
        node->right->is_leaf = 1;
        node->left->bound = handles[0].bound;
        node->right->bound = handles[1].bound;
    }else if(depth > max_depth){
        NodeSetItens(node, n);
        memcpy(node->handles, handles, n*sizeof(PrimitiveHandle));
        node->bound = handles[0].bound;
        for(int i = 1; i < n; i++){
            node->bound = Union(node->bound, handles[i].bound);
        }
        
        node->is_leaf = 1;
        return node;
    }else{
        node->left = CreateBVH(handles, n/2, depth+1, max_depth, totalNodes);
        node->right = CreateBVH(&handles[n/2], n-n/2, depth+1, max_depth, totalNodes);
    }
    
    node->bound = Union(node->left->bound, node->right->bound);
    return node;
}

__bidevice__ void MakeSceneTable(Aggregator *scene, int id){
    if(id < scene->head){
        Primitive *pri = scene->primitives[id];
        Shape *shape = pri->shape;
        AssertA(shape, "No shape in kernel");
        scene->handles[id].bound = shape->GetBounds();
        scene->handles[id].handle = id;
    }
}

__global__ void BuildSceneTable(Aggregator *scene){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < scene->head){
        MakeSceneTable(scene, tid);
    }
}

__host__ void Aggregator::Wrap(){
    int max_depth = 12;
    totalNodes = 0;
    handles = cudaAllocateVx(PrimitiveHandle, head);
    
    for(int i = 0; i < nMeshes; i++){
        WrapMesh(meshPtrs[i]);
    }
    
    size_t pThreads = 64;
    size_t pBlocks = (head + pThreads - 1)/pThreads;
    BuildSceneTable<<<pBlocks, pThreads>>>(this);
    cudaDeviceAssert();
    
    printf("Wrapping primitives...");
    root = CreateBVH(handles, head, 0, max_depth, &totalNodes);
    printf("OK [ Build BVH with %d nodes ]\n", totalNodes);
}