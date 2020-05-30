#include <shape.h>

__bidevice__ Mesh::Mesh(const Transform &toWorld, ParsedMesh *pMesh, int copy) 
: Shape(toWorld)
{
    nTriangles = pMesh->nTriangles;
    nVertices = pMesh->nVertices;
    nUvs = pMesh->nUvs;
    nNormals = pMesh->nNormals;
    ObjectToWorld = pMesh->toWorld;
    WorldToObject = Inverse(toWorld);
    if(copy){
        indices = new Point3i[3 * pMesh->nTriangles];
        memcpy(indices, pMesh->indices, sizeof(Point3i) * 3 * pMesh->nTriangles);
        
        p = new Point3f[pMesh->nVertices];
        for(int i = 0; i < pMesh->nVertices; i++) p[i] = ObjectToWorld(pMesh->p[i]);
        
        if(pMesh->uv && pMesh->nUvs > 0){
            uv = new Point2f[pMesh->nUvs];
            memcpy(uv, pMesh->uv, pMesh->nUvs * sizeof(Point2f));
        }
        
        if(pMesh->n && pMesh->nNormals > 0){
            n = new Normal3f[pMesh->nNormals];
            for(int i = 0; i < pMesh->nNormals; i++) n[i] = ObjectToWorld(pMesh->n[i]);
        }
        
    }else{
        indices = pMesh->indices;
        p = pMesh->p;
        n = pMesh->n;
        s = pMesh->s;
        uv = pMesh->uv;
        
        for(int i = 0; i < pMesh->nVertices; i++) p[i] = ObjectToWorld(p[i]);
        
        if(n && pMesh->nNormals > 0){
            for(int i = 0; i < pMesh->nNormals; i++) n[i] = ObjectToWorld(n[i]);
        }
    }
    
    type = ShapeType::MESH;
}

__bidevice__ void Mesh::GetUVs(Point2f st[3], int triNum) const{
    if(uv){
        int i0 = indices[3 * triNum + 0].z;
        int i1 = indices[3 * triNum + 1].z;
        int i2 = indices[3 * triNum + 2].z;
        if((i0 < 0 || i0 > nUvs) || (i1 < 0 || i1 > nUvs) || (i2 < 0 || i2 > nUvs)){
            st[0] = Point2f(0, 0);
            st[1] = Point2f(1, 0);
            st[2] = Point2f(1, 1);
        }else{
            st[0] = Point2f(uv[i0].x, 1.0 - uv[i0].y); 
            st[1] = Point2f(uv[i1].x, 1.0 - uv[i1].y);
            st[2] = Point2f(uv[i2].x, 1.0 - uv[i2].y); 
        }
    }else{
        st[0] = Point2f(0, 0);
        st[1] = Point2f(1, 0);
        st[2] = Point2f(1, 1);
    }
}

__bidevice__ bool Mesh::IntersectTriangleLow(const Ray &ray, SurfaceInteraction * isect,
                                             int triNum, Float *tHit) const
{
    int i0 = indices[3 * triNum + 0].x;
    int i1 = indices[3 * triNum + 1].x;
    int i2 = indices[3 * triNum + 2].x;
    Point3f p0 = p[i0];
    Point3f p1 = p[i1];
    Point3f p2 = p[i2];
    vec3f v0v1 = p1 - p0;
    vec3f v0v2 = p2 - p0;
    vec3f N = Normalize(Cross(v0v1, v0v2));
    if(Dot(N, ray.d) > 0) N = -N;
    
    Float u = 0, v = 0;
    vec3f pvec = Cross(ray.d, v0v2);
    Float det = Dot(v0v1, pvec);
    if(IsZero(det)) return false;
    
    Float invDet = 1 / det;
    vec3f tvec = ray.o - p0;
    u = Dot(tvec, pvec) * invDet;
    if(u < 0 || u > 1) return false;
    
    vec3f qvec = Cross(tvec, v0v1);
    v = Dot(ray.d, qvec) * invDet;
    if(v < 0 || v + u > 1) return false;
    
    Float t = Dot(v0v2, qvec) * invDet;
    if(t < 0 || t > ray.tMax) return false;
    
    vec3f dpdu, dpdv;
    CoordinateSystem(N, &dpdu, &dpdv);
    
    Point3f pHit = ray(t);
    vec3f pError = gamma(9) * Abs((vec3f)pHit);
    
    *isect = SurfaceInteraction(pHit, pError, Point2f(u,v), -ray.d, dpdu, dpdv,
                                Normal3f(0, 0, 0), Normal3f(0, 0, 0), ray.time,
                                this, triNum);
    isect->n = Normal3f(Normalize(N));
    
    *tHit = t;
    return true;
}

__bidevice__ bool Mesh::IntersectTriangle(const Ray &ray, SurfaceInteraction * isect,
                                          int triNum, Float *tHit) const
{
    int i0 = indices[3 * triNum + 0].x;
    int i1 = indices[3 * triNum + 1].x;
    int i2 = indices[3 * triNum + 2].x;
    Point3f p0 = p[i0];
    Point3f p1 = p[i1];
    Point3f p2 = p[i2];
#if 0
    printf(v3fA(p0) " " v3fA(p1) " " v3fA(p2) "\n",
           v3aA(p0), v3aA(p1), v3aA(p2));
#endif
    
    Point3f p0t = p0 - vec3f(ray.o);
    Point3f p1t = p1 - vec3f(ray.o);
    Point3f p2t = p2 - vec3f(ray.o);
    
    int kz = MaxDimension(Abs(ray.d));
    int kx = kz + 1;
    if(kx == 3) kx = 0;
    int ky = kx + 1;
    if(ky == 3) ky = 0;
    vec3f d = Permute(ray.d, kx, ky, kz);
    
    p0t = Permute(p0t, kx, ky, kz);
    p1t = Permute(p1t, kx, ky, kz);
    p2t = Permute(p2t, kx, ky, kz);
    
    Float Sx = -d.x / d.z;
    Float Sy = -d.y / d.z;
    Float Sz = 1.f / d.z;
    p0t.x += Sx * p0t.z;
    p0t.y += Sy * p0t.z;
    p1t.x += Sx * p1t.z;
    p1t.y += Sy * p1t.z;
    p2t.x += Sx * p2t.z;
    p2t.y += Sy * p2t.z;
    
    Float e0 = p1t.x * p2t.y - p1t.y * p2t.x;
    Float e1 = p2t.x * p0t.y - p2t.y * p0t.x;
    Float e2 = p0t.x * p1t.y - p0t.y * p1t.x;
    
    if((IsZero(e0) || IsZero(e1) || IsZero(e2))){
        double p2txp1ty = (double)p2t.x * (double)p1t.y;
        double p2typ1tx = (double)p2t.y * (double)p1t.x;
        e0 = (float)(p2typ1tx - p2txp1ty);
        double p0txp2ty = (double)p0t.x * (double)p2t.y;
        double p0typ2tx = (double)p0t.y * (double)p2t.x;
        e1 = (float)(p0typ2tx - p0txp2ty);
        double p1txp0ty = (double)p1t.x * (double)p0t.y;
        double p1typ0tx = (double)p1t.y * (double)p0t.x;
        e2 = (float)(p1typ0tx - p1txp0ty);
    }
    
    if((e0 < 0 || e1 < 0 || e2 < 0) && (e0 > 0 || e1 > 0 || e2 > 0))
        return false;
    Float det = e0 + e1 + e2;
    if(IsZero(det)) return false;
    
    p0t.z *= Sz;
    p1t.z *= Sz;
    p2t.z *= Sz;
    Float tScaled = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z;
    if(det < 0 && (tScaled >= 0 || tScaled < ray.tMax * det))
        return false;
    else if(det > 0 && (tScaled <= 0 || tScaled > ray.tMax * det))
        return false;
    
    Float invDet = 1 / det;
    Float b0 = e0 * invDet;
    Float b1 = e1 * invDet;
    Float b2 = e2 * invDet;
    Float t = tScaled * invDet;
    
    Float maxZt = MaxComponent(Abs(vec3f(p0t.z, p1t.z, p2t.z)));
    Float deltaZ = gamma(3) * maxZt;
    
    Float maxXt = MaxComponent(Abs(vec3f(p0t.x, p1t.x, p2t.x)));
    Float maxYt = MaxComponent(Abs(vec3f(p0t.y, p1t.y, p2t.y)));
    Float deltaX = gamma(5) * (maxXt + maxZt);
    Float deltaY = gamma(5) * (maxYt + maxZt);
    
    Float deltaE = 2 * (gamma(2) * maxXt * maxYt + deltaY * maxXt + deltaX * maxYt);
    
    Float maxE = MaxComponent(Abs(vec3f(e0, e1, e2)));
    Float deltaT = 3 * (gamma(3) * maxE * maxZt + deltaE * maxZt + deltaZ * maxE) * Absf(invDet);
    if (t <= deltaT) return false;
    
    vec3f dpdu, dpdv;
    Point2f st[3];
    GetUVs(st, triNum);
    
    vec2f dst02 = st[0] - st[2], dst12 = st[1] - st[2];
    vec3f dp02 = p0 - p2, dp12 = p1 - p2;
    Float determinant = dst02[0] * dst12[1] - dst02[1] * dst12[0];
    
    bool degenerateUV = Absf(determinant) < 1e-8;
    
    // NOTE: Some meshes might have denegerate UVs and we can't trace correctly.
    //       However meshes that don't need that much precision on uv can get by
    //       simply ignoring uv on this triangle. This avoid the error that happen
    //       when we decide simply to not render this triangle and for meshes that
    //       don't rely on uv maps, i.e.: have constant materials. See small budda
    //       for reference.
    // TODO: Maybe instead of defaulting to no uv mapping we can map further away.
    //       This however needs some structured storage of triangles which I don't have now.
    //       Maybe a edge structure so that we can sample neighbor triangles solves this.
    if(degenerateUV){ // Fallback to no uv mapping and try again
        st[0] = Point2f(0, 0);
        st[1] = Point2f(1, 0);
        st[2] = Point2f(1, 1);
        dst02 = st[0] - st[2], dst12 = st[1] - st[2];
        dp02 = p0 - p2, dp12 = p1 - p2;
        determinant = dst02[0] * dst12[1] - dst02[1] * dst12[0];
        degenerateUV = Absf(determinant) < 1e-8;
    }
    
    if(!degenerateUV){
        Float invdet = 1 / determinant;
        dpdu = (dst12[1] * dp02 - dst02[1] * dp12) * invdet;
        dpdv = (-dst12[0] * dp02 + dst02[0] * dp12) * invdet;
    }
    
    // NOTE: Don't give up on this triangle, this can cause serious bugs on large
    //       meshes rendered with small scale matrix. Perhaps the mesh handling
    //       API for PBRTv4 needs to better handle scales on triangles. I caught 
    //       a lot errors of this type.
#if 0
    if(degenerateUV || IsZero(Cross(dpdu, dpdv).LengthSquared())){
        // Handle zero determinant for triangle partial derivative matrix
        vec3f ng = Cross(p2 - p0, p1 - p0);
        if(IsZero(ng.LengthSquared()))
            // The triangle is actually degenerate; the intersection is bogus.
            return false;
        
        CoordinateSystem(Normalize(ng), &dpdu, &dpdv);
    }
#endif
    
    Float xAbsSum = (Absf(b0 * p0.x) + Absf(b1 * p1.x) + Absf(b2 * p2.x));
    Float yAbsSum = (Absf(b0 * p0.y) + Absf(b1 * p1.y) + Absf(b2 * p2.y));
    Float zAbsSum = (Absf(b0 * p0.z) + Absf(b1 * p1.z) + Absf(b2 * p2.z));
    vec3f pError = gamma(7) * vec3f(xAbsSum, yAbsSum, zAbsSum);
    
    Point3f pHit = b0 * p0 + b1 * p1 + b2 * p2;
    Point2f stHit = b0 * st[0] + b1 * st[1] + b2 * st[2];
    
    *isect = SurfaceInteraction(pHit, pError, stHit, -ray.d, dpdu, dpdv,
                                Normal3f(0, 0, 0), Normal3f(0, 0, 0), ray.time,
                                this, triNum);
    
    isect->n = Normal3f(Normalize(Cross(dp02, dp12)));
    
    *tHit = t;
    return true;
}

__bidevice__ bool Mesh::IntersectMeshNode(Node *node, const Ray &r, 
                                          SurfaceInteraction * isect, Float *tHit) const
{
    Assert(node->n > 0 && node->is_leaf && node->handles);
    bool hit_anything = false;
    for(int i = 0; i < node->n; i++){
        int nTri = node->handles[i].handle;
        if(IntersectTriangle(r, isect, nTri, tHit)){
            hit_anything = true;
            r.tMax = *tHit;
        }
    }
    
    return hit_anything;
}

//TODO: This is a duplicated of the primitive.cpp call, refactor?
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

/* The following build is required since Mesh vtable is not available to the CPU */
struct MeshData{
    int nTriangles, nVertices;
};


__global__ void MeshGetBounds(PrimitiveHandle *handles, Mesh *mesh){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < mesh->nTriangles){
        PrimitiveHandle *pHandle = &handles[tid];
        int i0 = mesh->indices[3 * tid + 0].x;
        int i1 = mesh->indices[3 * tid + 1].x;
        int i2 = mesh->indices[3 * tid + 2].x;
        Point3f p0 = mesh->p[i0];
        Point3f p1 = mesh->p[i1];
        Point3f p2 = mesh->p[i2];
        
        Bounds3f bound(p0, p1);
        bound = Union(bound, p2);
        
        pHandle->bound = bound;
        pHandle->handle = tid;
    }
}

__global__ void GetMeshData(MeshData *data, Mesh *mesh){
    if(threadIdx.x == 0 && blockIdx.x == 0){
        data->nTriangles = mesh->nTriangles;
        data->nVertices = mesh->nVertices;
    }
}

__global__ void MeshSetHandles(Mesh *mesh, PrimitiveHandle *handles, Node *bvhNode){
    AssertA(mesh && handles, "Invalid mesh/handle pointers");
    if(threadIdx.x == 0 && blockIdx.x == 0){
        mesh->handles = handles;
        mesh->bvh = bvhNode;
    }
}

__host__ void WrapMesh(Mesh *mesh){
    PrimitiveHandle *handles;
    MeshData *data = cudaAllocateVx(MeshData, 1);
    GetMeshData<<<1, 1>>>(data, mesh);
    cudaDeviceAssert();
    
    handles = cudaAllocateVx(PrimitiveHandle, data->nTriangles);
    
    size_t pThreads = 64;
    size_t pBlocks = (data->nTriangles + pThreads - 1)/pThreads;
    printf("Computing triangles Bounds...");
    MeshGetBounds<<<pBlocks, pThreads>>>(handles, mesh);
    cudaDeviceAssert();
    
    printf("OK\nPacking...");
    int max_depth = 12;
    int totalNodes = 0;
    Node *bvh = CreateBVH(handles, data->nTriangles, 0, max_depth, &totalNodes);
    Point3f pMin = bvh->bound.pMin;
    Point3f pMax = bvh->bound.pMax;
    printf("OK [ Build BVH with %d nodes, bounds: " v3fA(pMin) ", " v3fA(pMax) " ] \n",
           totalNodes, v3aA(pMin), v3aA(pMax));
    
    MeshSetHandles<<<1, 1>>>(mesh, handles, bvh);
    cudaDeviceAssert();
    
    cudaFree(data);
}