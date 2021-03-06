#include <primitive.h>
#include <material.h>
#include <util.h>
#include <camera.h>
#include <light.h>
#define DangerousDistance 1e-6

int BVH_MAX_DEPTH = 20;

struct NodeDistribution{
    Node *nodes;
    PrimitiveHandle *handles;
    int length;
    int head;
    int maxElements;
    int handleHead;
    int maxHandles;
    int skippedSorts;
    int totalSorts;
};

__host__ void MakeNodeDistribution(NodeDistribution *dist, int nElements,
                                   int maxDepth)
{
    Float fh = Log2(nElements);
    int h = ceil(fh);
    h = h > maxDepth ? maxDepth : h;
    int c = std::pow(2, h+1) - 1;
    int leafs = std::pow(2, h);
    long mem = sizeof(Node) * c;
    mem /= (1024 * 1024);
    
    printf(" * Requesting %ld Mb for nodes ...", mem);
    dist->nodes = cudaAllocateVx(Node, c);
    printf("OK\n");
    
    mem = sizeof(PrimitiveHandle) * nElements;
    mem /= (1024 * 1024);
    printf(" * Requsting %ld Mb for handles ...", mem);
    dist->handles = cudaAllocateVx(PrimitiveHandle, nElements);
    printf("OK\n");
    
    dist->length = c;
    dist->head = 0;
    dist->handleHead = 0;
    dist->maxHandles = nElements;
    dist->maxElements = 0;
    dist->totalSorts = 0;
    dist->skippedSorts = 0;
}

__bidevice__ bool PrimitiveIntersect(const Primitive *primitive, const Ray &ray,
                                     SurfaceInteraction *isect)
{
    Float tHit;
    SurfaceInteraction tmp;
    if(!primitive->shape->Intersect(ray, &tHit, &tmp)) return false;
    
    if(IsZero(tHit - DangerousDistance)){
        //printf("Warning: Possible self intersection with distance: %g\n", tHit);
    }
    
    *isect = tmp;
    ray.tMax = tHit;
    
    if(primitive->mediumInterface.IsMediumTransition())
        isect->mediumInterface = primitive->mediumInterface;
    else
        isect->mediumInterface = MediumInterface(ray.medium);
    
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

__bidevice__ Bounds3f Aggregator::WorldBound() const{
    return root->bound;
}

__host__ void Aggregator::ReserveMeshes(int n){
    nAllowedMeshes = n;
    meshPtrs = cudaAllocateVx(Mesh*, n);
    nMeshes = 0;
}

__host__ void Aggregator::ReserveParticleClouds(int n){
    nAllowedpClouds = n;
    pClouds = cudaAllocateVx(ParticleCloud*, n);
    npClouds = 0;
}

__bidevice__ ParticleCloud *Aggregator::AddParticleCloud(vec3f *pos, int n, Float scale){
    ParticleCloud *ptr = nullptr;
    if(npClouds < nAllowedpClouds){
        pClouds[npClouds] = new ParticleCloud(pos, n, scale);
        ptr = pClouds[npClouds];
        npClouds++;
    }else{
        printf("Hit maximum particle clouds allowed [%d] \n", npClouds);
    }
    
    return ptr;
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

__bidevice__ void Aggregator::Reserve(int size){
    Assert(size > 0);
    length = size;
    primitives = new Primitive*[size];
}

__bidevice__ void Aggregator::Insert(Primitive *pri, int is_light){
    Assert(head < length && primitives);
    primitives[head] = pri;
    if(is_light){
        LightDesc desc;
        Assert(lightCounter < 256);
        desc.type = LightType::DiffuseArea;
        desc.flags = (int)LightFlags::Area;
        desc.shapeId = head;
        desc.toWorld = pri->shape->ObjectToWorld;
        lightList[lightCounter++] = desc;
    }
    
    head++;
}

__bidevice__ void Aggregator::InsertDistantLight(const Spectrum &L, const vec3f &w){
    Assert(lightCounter < 256);
    LightDesc desc;
    desc.type = LightType::Distant;
    desc.flags = (int)LightFlags::DeltaDirection;
    desc.toWorld = Transform();
    desc.wLight = w;
    desc.L = L;
    lightList[lightCounter++] = desc;
}

__bidevice__ void Aggregator::InsertInfiniteLight(MipMap<Spectrum> *mipmap, 
                                                  Distribution2D *dist,
                                                  const Transform &LightToWorld)
{
    Assert(lightCounter < 256);
    LightDesc desc;
    desc.type = LightType::Infinite;
    desc.flags = (int)LightFlags::Infinite;
    desc.toWorld = LightToWorld;
    desc.Ls = mipmap;
    desc.dist = dist;
    lightList[lightCounter++] = desc;
}

__bidevice__ void Aggregator::SetLights(){
    if(lightCounter > 0){
        int lightAreaCount = 0;
        int lightDistantCount = 0;
        int lightInfiniteCount = 0;
        lights = new Light*[lightCounter];
        for(int i = 0; i < lightCounter; i++){
            LightDesc desc = lightList[i];
            Light *light = new Light(desc.toWorld, desc.flags);
            
            if(desc.type == LightType::DiffuseArea){
                Primitive *pri = primitives[desc.shapeId];
                GeometricEmitterPrimitive *gPri = (GeometricEmitterPrimitive *)pri;
                /* Diffuse areas need to set geometry light pointer upon light creation */
                light->Init_DiffuseArea(pri->Le(), pri->shape, true);
                gPri->light = light;
                lightAreaCount++;
            }else if(desc.type == LightType::Distant){
                light->Init_Distant(desc.L, desc.wLight);
                lightDistantCount++;
            }else if(desc.type == LightType::Infinite){
                light->Init_Infinite(desc.Ls, desc.dist);
                lightInfiniteCount++;
            }else{
                //TODO
                AssertA(0, "Unsupported light type");
            }
            
            light->Prepare(this);
            lights[i] = light;
        }
        
        if(lightAreaCount > 0)
            printf(" * Created %d Area Light(s)\n", lightAreaCount);
        if(lightDistantCount > 0)
            printf(" * Created %d Distant Light(s)\n", lightDistantCount);
        if(lightInfiniteCount > 0)
            printf(" * Created %d Infinite Light(s)\n", lightInfiniteCount);
    }
}

__bidevice__ Spectrum Aggregator::EstimateDirect(const Interaction &it, BSDF *bsdf,
                                                 const Point2f &uScattering,
                                                 Light *light, 
                                                 const Point2f &uLight, 
                                                 bool handleMedium,
                                                 bool specular) const
{
    BxDFType bsdfFlags = specular ? BSDF_ALL : BxDFType(BSDF_ALL & ~BSDF_SPECULAR);
    Spectrum Ld(0.f);
    vec3f wi;
    Float lightPdf = 0, scatteringPdf = 0;
    VisibilityTester visibility;
    Spectrum Li = light->Sample_Li(it, uLight, &wi, &lightPdf, &visibility);
    //printf("Sampled " v3fA(Li) "\n", v3aA(Li));
    
    if(lightPdf > 0 && !Li.IsBlack() && !IsZero(lightPdf)){
        Spectrum f;
        if(it.IsSurfaceInteraction() && bsdf){ //is surface interaction
            const SurfaceInteraction &isect = (const SurfaceInteraction &)it;
            f = bsdf->f(isect.wo, wi, bsdfFlags) * AbsDot(wi, ToVec3(isect.n));
            scatteringPdf = bsdf->Pdf(isect.wo, wi, bsdfFlags);
        }else{
            const MediumInteraction &mi = (const MediumInteraction &)it;
            Float p = mi.phase.p(mi.wo, wi);
            f = Spectrum(p);
            scatteringPdf = p;
        }
        
        if(!f.IsBlack()){
            if(handleMedium){
                Li *= visibility.Tr(this);
            }else{
                if(!visibility.Unoccluded(this)){ //something in front
                    Li = Spectrum(0.f);
                }
            }
            
            if(!Li.IsBlack()){
                if(IsDeltaLight(light->flags)){ //not happening here
                    Ld += f * Li / lightPdf;
                }else{
                    Float weight = PowerHeuristic(1, lightPdf, 1, scatteringPdf);
                    Ld += f * Li * weight / lightPdf;
                }
            }
        }
    }
    
    if(!IsDeltaLight(light->flags)){
        Spectrum f;
        bool sampledSpecular = false;
        if(it.IsSurfaceInteraction() && bsdf){ //is surface interaction
            BxDFType sampledType;
            const SurfaceInteraction &isect = (const SurfaceInteraction &)it;
            f = bsdf->Sample_f(isect.wo, &wi, uScattering, 
                               &scatteringPdf, bsdfFlags, &sampledType);
            f *= AbsDot(wi, ToVec3(isect.n));
            sampledSpecular = (sampledType & BSDF_SPECULAR) != 0;
        }else{
            const MediumInteraction &mi = (const MediumInteraction &)it;
            Float p = mi.phase.Sample_p(mi.wo, &wi, uScattering);
            f = Spectrum(p);
            scatteringPdf = p;
        }
        
        if(!f.IsBlack() && scatteringPdf > 0 && !IsZero(scatteringPdf)){
            Float weight = 1;
            if(!sampledSpecular){
                lightPdf = light->Pdf_Li(it, wi);
                if(IsZero(lightPdf)) return Ld;
                weight = PowerHeuristic(1, scatteringPdf, 1, lightPdf);
            }
            
            SurfaceInteraction lightIsect;
            Ray ray = it.SpawnRay(wi);
            Spectrum Tr(1.f);
            
            bool foundSurfaceInteraction =
                handleMedium? IntersectTr(ray, &lightIsect, &Tr) 
                : Intersect(ray, &lightIsect);
            
            Spectrum Li(0.f);
            if(foundSurfaceInteraction){
                if(lightIsect.primitive->GetLight() == light) Li = lightIsect.Le(-wi);
            }else{
                Li = light->Le(ray);
            }
            
            if(!Li.IsBlack()) Ld += f * Li * Tr * weight / scatteringPdf;
        }
    }
    
    return Ld;
}

__bidevice__ Spectrum Aggregator::UniformSampleOneLight(const Interaction &it, BSDF *bsdf,
                                                        Point2f u2, Point3f u3,
                                                        bool handleMedium) const
{
    int nLights = lightCounter;
    if(lightCounter < 1) return Spectrum(0.f);
    int lightNum;
    Float lightPdf;
    
    lightNum = Min((int)(u2[0] * nLights), nLights - 1);
    lightPdf = Float(1) / nLights;
    
    Light *light = lights[lightNum];
    Point2f uLight(u2[1], u3[0]);
    Point2f uScattering(u3[1], u3[2]);
    
    return EstimateDirect(it, bsdf, uScattering, light, uLight, handleMedium) / lightPdf;
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
    int hit_tests = 0;
    bool hit_anything = false;
    
    while(node != nullptr){
        bool hit_bound = node->bound.IntersectP(r);
        if(hit_bound){
            if(node->is_leaf){
                hit_anything |= IntersectNode(node, r, isect);
                hit_tests += node->n;
                node = *--stackPtr;
            }else{
                Float tl0, tr0;
                NodePtr childL = node->left;
                NodePtr childR = node->right;
                bool shouldVisitLeft  = childL->bound.IntersectP(r, &tl0);
                bool shouldVisitRight = childR->bound.IntersectP(r, &tr0);
                hit_tests += 2;
                
                if(shouldVisitRight && shouldVisitLeft){
                    NodePtr firstChild = nullptr, secondChild = nullptr;
                    if(tr0 < tl0){
                        firstChild  = childR;
                        secondChild = childL;
                    }else{
                        firstChild  = childL;
                        secondChild = childR;
                    }
                    
                    *stackPtr++ = secondChild;
                    node = firstChild;
                }else if(shouldVisitLeft){
                    node = childL;
                }else if(shouldVisitRight){
                    node = childR;
                }else{
                    node = *--stackPtr;
                }
            }
        }else{
            node = *--stackPtr;
        }
    }
    
    if(pixel){
        if(pixel->stats.max_transverse_tests < hit_tests)
            pixel->stats.max_transverse_tests = hit_tests;
    }
    
    return hit_anything;
}

__bidevice__ bool Aggregator::IntersectTr(Ray ray, SurfaceInteraction *isect, 
                                          Spectrum *Tr, Pixel *pixel) const
{
    *Tr = Spectrum(1);
    int it = 0;
    int warned = 0;
    int debug = 0;
    while(true){
        bool hitSurface = Intersect(ray, isect, pixel);
        if(ray.medium){
            *Tr *= ray.medium->Tr(ray);
        }
        
        if(!hitSurface || isect->primitive->IsEmissive()) return false;
        if(isect->primitive->GetMaterial() != nullptr) return true;
        ray = isect->SpawnRay(ray.d);
        if(debug){
            if(it++ > WARN_BOUNCE_COUNT){
                if(!warned){
                    printf("Warning: Dangerously high bounce count (%d) in Aggregator::Tr ( Tr = [ %g %g %g ])\n",
                           it, __vec3_args((*Tr)));
                    warned = 1;
                }
            }
        }
    }
    
    return false;
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

__host__ Node *GetNode(int n, NodeDistribution *nodeDist){
    if(!(nodeDist->head < nodeDist->length)){
        printf(" ** [ERROR] : Allocated %d but requested more nodes\n", nodeDist->length);
        AssertA(0, "Too many node requirement");
    }
    
    Node *node = &nodeDist->nodes[nodeDist->head++];
    node->left = nullptr;
    node->right = nullptr;
    node->handles = nullptr;
    node->n = n;
    node->is_leaf = 0;
    return node;
}

__host__ void NodeSetItens(Node *node, int n, NodeDistribution *dist){
    AssertA(dist->handleHead+n <= dist->maxHandles, "Too many handles requirement");
    node->n = n;
    node->handles = &dist->handles[dist->handleHead];
    dist->handleHead += n;
}


/*
* This BVH construction algorithm is adapted from the original proposal. 
* It no longers manages memory dinamically getting a decent speedup. We also
* perform axis comparison before sorting having a reduction of about 33% on sort
* operations which on 7M triangle mesh reduced the construction time (on my machine)
* from 2 minutes to 14 seconds.
*/
__host__ Node *_CreateBVH(PrimitiveHandle *handles,int n, int depth, 
                          int max_depth, NodeDistribution *distr, int last_axis=-1)
{
    Node *node = GetNode(n, distr);
    int axis = int(3 * rand_float());
    
    if(axis != last_axis){
        last_axis = axis;
        distr->totalSorts ++;
        if(axis == 0){
            QuickSort(handles, n, CompareX);
        }else if(axis == 1){
            QuickSort(handles, n, CompareY);
        }else if(axis == 2){
            QuickSort(handles, n, CompareZ);
        }
    }else{
        distr->skippedSorts ++;
    }
    
    if(n == 1){
        NodeSetItens(node, n, distr);
        memcpy(node->handles, handles, n * sizeof(PrimitiveHandle));
        node->bound = handles[0].bound;
        node->is_leaf = 1;
        if(distr->maxElements < n) distr->maxElements = n;
        return node;
    }else if(depth >= max_depth){
        NodeSetItens(node, n, distr);
        memcpy(node->handles, handles, n*sizeof(PrimitiveHandle));
        node->bound = handles[0].bound;
        for(int i = 1; i < n; i++){
            node->bound = Union(node->bound, handles[i].bound);
        }
        
        node->is_leaf = 1;
        if(distr->maxElements < n) distr->maxElements = n;
        return node;
    }else{
        node->left  = _CreateBVH(handles, n/2, depth+1, max_depth, distr, last_axis);
        node->right = _CreateBVH(&handles[n/2], n-n/2, depth+1, max_depth, distr, last_axis);
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

__host__ Node *CreateBVH(PrimitiveHandle *handles, int n, int depth, 
                         int max_depth, int *totalNodes, int *maxNodes)
{
    NodeDistribution distr;
    memset(&distr, 0x00, sizeof(NodeDistribution));
    MakeNodeDistribution(&distr, n, max_depth);
    clock_t start = clock();
    Node *root = _CreateBVH(handles, n, depth, max_depth, &distr);
    clock_t end = clock();
    *maxNodes = distr.maxElements;
    *totalNodes = distr.head;
    double time_taken = to_cpu_time(start, end);
    Float totalSorts = (Float)distr.totalSorts + (Float)distr.skippedSorts;
    Float sortReduction = 100.0f * (((Float)distr.skippedSorts) / totalSorts);
    printf(" * Time: %gs\n", time_taken);
    printf(" * Sort reduction algorihtm gain: %g%%\n", sortReduction);
    return root;
}

__host__ void Aggregator::Wrap(){
    int max_depth = BVH_MAX_DEPTH;
    int maxNodes = 0;
    totalNodes = 0;
    handles = cudaAllocateVx(PrimitiveHandle, head);
    
    for(int i = 0; i < nMeshes; i++){
        WrapMesh(meshPtrs[i]);
    }
    
    for(int i = 0; i < npClouds; i++){
        WrapParticleCloud(pClouds[i]);
    }
    
    size_t pThreads = 64;
    size_t pBlocks = (head + pThreads - 1)/pThreads;
    BuildSceneTable<<<pBlocks, pThreads>>>(this);
    cudaDeviceAssert();
    
    printf("Wrapping primitives\n");
    root = CreateBVH(handles, head, 0, max_depth, &totalNodes, &maxNodes);
    printf("[ Build BVH with %d nodes, max: %d ]\n", totalNodes, maxNodes);
}
