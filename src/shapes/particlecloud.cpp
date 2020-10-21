#include <shape.h>

__bidevice__ ParticleCloud::ParticleCloud(vec3f *pos, int n, Float scale) : 
Shape(Transform())
{
    positions = pos;
    nParticles = n;
    type = ShapeType::PARTICLECLOUD;
    radius = scale;
}

__bidevice__ Bounds3f ParticleCloud::GetBounds() const{
    return bvh->bound;
}

__bidevice__ Float ParticleCloud::ParticleArea(int nPart) const{
    (void)nPart;
    return 4 * Pi * radius * radius;
}

__bidevice__ Float ParticleCloud::Area() const{
    printf("Warning: Slow method called with ParticleCloud::Area()\n");
    Float area = 0;
    for(int i = 0; i < nParticles; i++){
        area += ParticleArea(i);
    }
    
    return area;
}

__bidevice__ Interaction ParticleCloud::SampleIndex(const Point2f &u, Float *pdf, 
                                                    int index) const
{
    Point3f pObj = Point3f(0) + radius * SampleSphere(u);
    Interaction it;
    Transform toWorld = Translate(positions[index]);
    it.n = Normalize(toWorld(Normal3f(pObj.x, pObj.y, pObj.z)));
    if(reverseOrientation) it.n *= -1;
    
    pObj *= radius / Distance(pObj, Point3f(0, 0, 0));
    vec3f pObjError = gamma(5) * Abs(ToVec3(pObj));
    it.p = toWorld(pObj, pObjError, &it.pError);
    *pdf = 1 / Area();
    return it;
}

__bidevice__ Interaction ParticleCloud::Sample(const Interaction &ref, const Point2f &u,
                                               Float *pdf) const
{
    int partNum = ref.faceIndex;
    Transform toWorld = Translate(positions[partNum]);
    Point3f pCenter = (toWorld)(Point3f(0, 0, 0));
    Point3f pOrigin = OffsetRayOrigin(ref.p, ref.pError, ref.n, pCenter - ref.p);
    if(DistanceSquared(pOrigin, pCenter) <= radius * radius){
        Interaction intr = SampleIndex(u, pdf, partNum);
        vec3f wi = intr.p - ref.p;
        if(IsZero(wi.LengthSquared())){
            *pdf = 0;
        }else{
            wi = Normalize(wi);
            *pdf *= DistanceSquared(ref.p, intr.p) / AbsDot(intr.n, -wi);
        }
        if (std::isinf(*pdf)) *pdf = 0.f;
        return intr;
    }
    
    Float dc = Distance(ref.p, pCenter);
    Float invDc = 1 / dc;
    vec3f wc = (pCenter - ref.p) * invDc;
    vec3f wcX, wcY;
    CoordinateSystem(wc, &wcX, &wcY);
    
    Float sinThetaMax = radius * invDc;
    Float sinThetaMax2 = sinThetaMax * sinThetaMax;
    Float invSinThetaMax = 1 / sinThetaMax;
    Float cosThetaMax = std::sqrt(Max((Float)0.f, 1 - sinThetaMax2));
    Float cosTheta  = (cosThetaMax - 1) * u[0] + 1;
    Float sinTheta2 = 1 - cosTheta * cosTheta;
    
    if(sinThetaMax2 < 0.00068523f /* sin^2(1.5 deg) */){
        /* Fall back to a Taylor series expansion for small angles, where
           the standard approach suffers from severe cancellation errors */
        sinTheta2 = sinThetaMax2 * u[0];
        cosTheta = std::sqrt(1 - sinTheta2);
    }
    
    Float cosAlpha = sinTheta2 * invSinThetaMax +
        cosTheta * std::sqrt(Max((Float)0.f, 1.f - sinTheta2 * 
                                 invSinThetaMax * invSinThetaMax));
    
    Float sinAlpha = std::sqrt(Max((Float)0.f, 1.f - cosAlpha*cosAlpha));
    Float phi = u[1] * 2 * Pi;
    vec3f nWorld = SphericalDirection(sinAlpha, cosAlpha, phi, -wcX, -wcY, -wc);
    Point3f pWorld = pCenter + radius * Point3f(nWorld.x, nWorld.y, nWorld.z);
    Interaction it;
    it.p = pWorld;
    it.pError = gamma(5) * Abs((vec3f)pWorld);
    it.n = Normal3f(nWorld);
    if(reverseOrientation) it.n *= -1;
    *pdf = 1 / (2 * Pi * (1 - cosThetaMax));
    return it;
}

__bidevice__ bool ParticleCloud::IntersectSphere(const Ray &r, SurfaceInteraction * isect,
                                                 int partNum, Float *tHit) const
{
    Float phi;
    Point3f pHit;
    vec3f oErr, dErr;
    
    Float phiMax = 2 * Pi; // 2 pi
    
    Transform toWorld = Translate(positions[partNum]);
    Transform toObject = Inverse(toWorld);
    Ray ray = (toObject)(r, &oErr, &dErr);
    
    Float ox = ray.o.x, oy = ray.o.y, oz = ray.o.z;
    Float dx = ray.d.x, dy = ray.d.y, dz = ray.d.z;
    Float a = dx * dx + dy * dy + dz * dz;
    Float b = 2 * (dx * ox + dy * oy + dz * oz);
    Float c = ox * ox + oy * oy + oz * oz - Float(radius) * Float(radius);
    
    Float t0, t1;
    if(!Quadratic(a, b, c, &t0, &t1)) return false;
    
    if(t0 > ray.tMax || t1 <= 0) return false;
    Float tShapeHit = t0;
    if(tShapeHit <= 0 || IsUnsafeHit(tShapeHit)){
        tShapeHit = t1;
        if(tShapeHit > ray.tMax) return false;
    }
    
    pHit = ray((Float)tShapeHit);
    
    pHit *= radius / Distance(pHit, Point3f(0, 0, 0));
    if (pHit.x == 0 && pHit.y == 0) pHit.x = 1e-5f * radius;
    phi = std::atan2(pHit.y, pHit.x);
    if (phi < 0) phi += 2 * Pi;
    
    Float u = phi / (2 * Pi);
    Float theta = std::acos(Clamp(pHit.z / radius, -1, 1));
    Float v = 1.0 - theta / Pi;
    
    Float zRadius = std::sqrt(pHit.x * pHit.x + pHit.y * pHit.y);
    Float invZRadius = 1 / zRadius;
    Float cosPhi = pHit.x * invZRadius;
    Float sinPhi = pHit.y * invZRadius;
    vec3f dpdu(-phiMax * pHit.y, phiMax * pHit.x, 0);
    vec3f dpdv =
        (- Pi) * vec3f(pHit.z * cosPhi, pHit.z * sinPhi, -radius * std::sin(theta));
    
    vec3f d2Pduu = -phiMax * phiMax * vec3f(pHit.x, pHit.y, 0);
    vec3f d2Pduv = (- Pi) * pHit.z * phiMax * vec3f(-sinPhi, cosPhi, 0.);
    vec3f d2Pdvv = (-Pi) * (Pi) * vec3f(pHit.x, pHit.y, pHit.z);
    
    Float E = Dot(dpdu, dpdu);
    Float F = Dot(dpdu, dpdv);
    Float G = Dot(dpdv, dpdv);
    vec3f N = Normalize(Cross(dpdu, dpdv));
    Float e = Dot(N, d2Pduu);
    Float f = Dot(N, d2Pduv);
    Float g = Dot(N, d2Pdvv);
    
    Float invEGF2 = 1 / (E * G - F * F);
    Normal3f dndu = Normal3f((f * F - e * G) * invEGF2 * dpdu +
                             (e * F - f * E) * invEGF2 * dpdv);
    Normal3f dndv = Normal3f((g * F - f * G) * invEGF2 * dpdu +
                             (f * F - g * E) * invEGF2 * dpdv);
    
    vec3f pError = gamma(5) * Abs((vec3f)pHit);
    
    *isect = (toWorld)(SurfaceInteraction(pHit, pError, Point2f(u, v),
                                          -ray.d, dpdu, dpdv, dndu, dndv,
                                          ray.time, this, partNum));
    
    *tHit = (Float)tShapeHit;
    return true;
}

__bidevice__ bool ParticleCloud::IntersectParticleCloudNode(Node *node, const Ray &r, 
                                                            SurfaceInteraction * isect, 
                                                            Float *tHit) const
{
    Assert(node->n > 0 && node->is_leaf && node->handles);
    bool hit_anything = false;
    for(int i = 0; i < node->n; i++){
        int nPart = node->handles[i].handle;
        if(IntersectSphere(r, isect, nPart, tHit)){
            hit_anything = true;
            r.tMax = *tHit;
        }
    }
    
    return hit_anything;
}

__bidevice__ bool ParticleCloud::Intersect(const Ray &r, Float *tHit,
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
        return IntersectParticleCloudNode(node, r, isect, tHit);
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
                if(IntersectParticleCloudNode(childL, r, &tmp, tHit)){
                    hit_anything = true;
                }
            }
            
            if(hitr && childR->is_leaf){
                hit_tests += childR->n;
                if(IntersectParticleCloudNode(childR, r, &tmp, tHit)){
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




/* The following build is required since Mesh vtable is not available to the CPU */
struct PartData{
    int nParticles;
};

__global__ void ParticleCloudGetBounds(PrimitiveHandle *handles, ParticleCloud *pCloud){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < pCloud->nParticles){
        PrimitiveHandle *pHandle = &handles[tid];
        Transform toWorld = Translate(pCloud->positions[tid]);
        Bounds3f bound(Point3f(-pCloud->radius), Point3f(pCloud->radius));
        pHandle->bound = toWorld(bound);
        pHandle->handle = tid;
    }
}

__global__ void GetParticleCloudData(PartData *data, ParticleCloud *pCloud){
    if(threadIdx.x == 0 && blockIdx.x == 0){
        data->nParticles = pCloud->nParticles;
    }
}

__global__ void ParticleCloudSetHandles(ParticleCloud *pCloud, PrimitiveHandle *handles, 
                                        Node *bvhNode)
{
    if(threadIdx.x == 0 && blockIdx.x == 0){
        pCloud->handles = handles;
        pCloud->bvh = bvhNode;
    }
}

__host__ void WrapParticleCloud(ParticleCloud *pCloud){
    PrimitiveHandle *handles;
    PartData *data = cudaAllocateVx(PartData, 1);
    GetParticleCloudData<<<1,1>>>(data, pCloud);
    cudaDeviceAssert();
    
    handles = cudaAllocateVx(PrimitiveHandle, data->nParticles);
    size_t pThreads = 64;
    size_t pBlocks = (data->nParticles + pThreads - 1)/pThreads;
    printf("Computing cloud Bounds...");
    ParticleCloudGetBounds<<<pBlocks, pThreads>>>(handles, pCloud);
    cudaDeviceAssert();
    
    int max_depth = BVH_MAX_DEPTH;
    int totalNodes = 0;
    int maxNodes = 0;
    
    Node *bvh = CreateBVH(handles, data->nParticles, 0, max_depth, &totalNodes, &maxNodes);
    Point3f pMin = bvh->bound.pMin;
    Point3f pMax = bvh->bound.pMax;
    printf("[ Build BVH with %d nodes, max: %d bounds: " v3fA(pMin) ", " v3fA(pMax) " ] \n",
           totalNodes, maxNodes, v3aA(pMin), v3aA(pMax));
    
    ParticleCloudSetHandles<<<1, 1>>>(pCloud, handles, bvh);
    cudaDeviceAssert();
    
    cudaFree(data);
}