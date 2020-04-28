#if !defined(LBVH_H)
#define LBVH_H
#include <iostream>
#include <cutil.h>
#include <types.h>
#include <aabb.h>
#include <vector>
#include <algorithm>

#if 0
struct BVHPrimitiveInfo{
    size_t primitiveNumber;
    AABB bounds;
    glm::vec3 centroid;
};

struct LBVH{
    int id;
    int n;
    int is_leaf;
    struct LBVH *children[2];
};

inline __host__
LBVH *recursiveBuild(BVHPrimitiveInfo *info, int start, int end, 
                    unsigned int *total, std::vector<AABB *> *ordered,
                    AABB *aabbs)
{
    AABB aabb;
    LBVH *node = new LBVH;
    (*total)++;
    aabb_init(&aabb);
    for(int i = start; i < end; i++){
        surrounding_box(&aabb, &aabb, &info[i].bounds);
    }
    
    int nPris = end - start;
    if(nPris == 1){
        int firstPrimOffset = ordered->size();
        for(int i = start; i < end; i++){
            int primNum = info[i].primitiveNumber;
            ordered->push_back(&aabbs[primNum]);
        }
        
        node->n = nPris;
        node->is_leaf = 1;
        node->id = -1;
        return node;
    }else{
        AABB centroidBounds;
        aabb_init(&centroidBounds);
        for(int i = start; i < end; i++){
            centroidBounds = surrounding_box(centroidBounds, info[i].centroid);
        }
        
        int dim = aabb_maxextent(centroidBounds);
        int mid = (start + end) / 2;
        if(IsZero(centroidBounds._max[dim] - centroidBounds._min[dim])){
            int firstPrimOffset = ordered->size();
            for(int i = start; i < end; ++i){
                int primNum = info[i].primitiveNumber;
                ordered->push_back(&aabbs[primNum]);
            }
            node->n = end - start;
            node->is_leaf = 1;
            node->id = -1;
            return node;
        }else{
            std::nth_element(&info[start], &info[mid], &info[end-1]+1,
                             [dim](const BVHPrimitiveInfo &a, 
                                   const BVHPrimitiveInfo &b){
                             return a.centroid[dim] < b.centroid[dim];
                             });
            
            node->id = -1;
            node->is_leaf = 0;
            node->n = end - start;
            node->children[0] = recursiveBuild(info, start, mid, total, ordered, aabbs);
            node->children[1] = recursiveBuild(info, mid, end, total, ordered, aabbs);
            return node;
        }
    }
}

inline __host__ 
void build_lbvh(AABB *aabbs, int size){
    AABB bounds;
    aabb_init(&bounds);
    unsigned int totalNodes = 0;
    
    size_t memory = sizeof(BVHPrimitiveInfo) * size;
    BVHPrimitiveInfo *bInfo = (BVHPrimitiveInfo *)cudaAllocate(memory);

    std::vector<AABB *>ordered;
    ordered.resize(size);
    
    for(int i = 0; i < size; i++){
        bInfo[i].bounds = aabbs[i];
        bInfo[i].primitiveNumber = i;
        bInfo[i].centroid = 0.5f * aabbs[i]._min + 0.5f * aabbs[i]._max;
        bounds = surrounding_box(bounds, bInfo[i].centroid);
    }
    
    LBVH *node = recursiveBuild(bInfo, 0, size, &totalNodes, &ordered, aabbs);
    std::cout << "Done " << totalNodes << std::endl;
}
#endif

#endif