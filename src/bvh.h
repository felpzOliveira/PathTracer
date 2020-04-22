#if !defined(BVH_H)
#define BVH_H

#include <types.h>
#include <aabb.h>
#include <cutil.h>

inline __host__ BVHNode * bvh_node_new(int n){
    size_t memory = sizeof(BVHNode);
    BVHNode *node = (BVHNode *)cudaAllocate(memory);
    node->n_handles = n;
    node->is_leaf = 0;
    node->handles = nullptr;
    node->left = nullptr;
    node->right = nullptr;
    return node;
}


inline __host__ void bvh_node_mark_itens(BVHNode *node, int n){
    if(node){
        if(n > 0){
            node->n_handles = n;
        }
    }
}

inline __host__ void bvh_node_set_itens(BVHNode *node, int n){
    if(node){
        if(n > 0){
            node->handles = (Object *)cudaAllocate(sizeof(Object)*n);
            node->n_handles = n;
        }
    }
}

inline __host__ int get_bvh_node_count(BVHNode *root){
    if(root){
        if(root->is_leaf){
            return 1;
        }
        
        return 1 + get_bvh_node_count(root->left) + 
            get_bvh_node_count(root->right);
    }
    
    return 0;
}

template<typename Q>
int box_z_compare (const void * a, const void * b, void * arg) {
    AABB box_left, box_right;
    Object *ah = (Object *)a;
    Object *bh = (Object *)b;
    Q *locator = (Q *)arg;
    get_aabb(locator, *ah, &box_left);
    get_aabb(locator, *bh, &box_right);
    
    if(box_left._min.z - box_right._min.z < 0.0f){
        return -1;
    }else{
        return 1;
    }
}

template<typename Q>
int box_y_compare (const void * a, const void * b, void * arg) {
    AABB box_left, box_right;
    Object *ah = (Object *)a;
    Object *bh = (Object *)b;
    Q *locator = (Q *)arg;
    get_aabb(locator, *ah, &box_left);
    get_aabb(locator, *bh, &box_right);
    
    if(box_left._min.y - box_right._min.y < 0.0f){
        return -1;
    }else{
        return 1;
    }
}

template<typename Q>
int box_x_compare (const void * a, const void * b, void * arg) {
    AABB box_left, box_right;
    Object *ah = (Object *)a;
    Object *bh = (Object *)b;
    Q *locator = (Q *)arg;
    get_aabb(locator, *ah, &box_left);
    get_aabb(locator, *bh, &box_right);
    
    if(box_left._min.x - box_right._min.x < 0.0f){
        return -1;
    }else{
        return 1;
    }
}

template<typename Q>
inline __host__ BVHNode *build_bvh(Q *locator, Object *handles, 
                                   int n, int depth, int max_depth)
{
    int axis = int(3*random_float());
    BVHNode *node = bvh_node_new(0);
    bvh_node_mark_itens(node, n);
    if (axis == 0)
        qsort_r(handles, n, sizeof(Object), box_x_compare<Q>, locator);
    else if (axis == 1)
        qsort_r(handles, n, sizeof(Object), box_y_compare<Q>, locator);
    else
        qsort_r(handles, n, sizeof(Object), box_z_compare<Q>, locator);
    
    if(n == 1){
        bvh_node_set_itens(node, 1);
        memcpy(node->handles, handles, sizeof(Object)*n);
        get_aabb(locator, handles[0], &node->box);
        node->is_leaf = 1;
        return node;
        
    }else if(n == 2){
        node->left = bvh_node_new(0);
        node->right = bvh_node_new(0);
        bvh_node_set_itens(node->left, 1);
        bvh_node_set_itens(node->right, 1);
        memcpy(node->left->handles, &handles[0], sizeof(Object));
        memcpy(node->right->handles, &handles[1], sizeof(Object));
        node->left->is_leaf = 1;
        node->right->is_leaf = 1;
        get_aabb(locator, handles[0], &node->left->box);
        get_aabb(locator, handles[1], &node->right->box);
        
    }else if(depth > max_depth){
        bvh_node_set_itens(node, n);
        memcpy(node->handles, handles, sizeof(Object)*n);
        get_aabb(locator, handles[0], &node->box);
        for(int i = 1; i < n; i += 1){
            AABB aabb;
            get_aabb(locator, handles[i], &aabb);
            surrounding_box(&node->box, &aabb, &node->box);
        }
        node->is_leaf = 1;
        return node;
    }else{
        node->left = build_bvh<Q>(locator, handles, n/2, depth+1, max_depth);
        node->right = build_bvh<Q>(locator, &handles[n/2], n-n/2, depth+1, max_depth);
    }
    
    surrounding_box(&node->box, &node->left->box, &node->right->box);
    return node;
}

#endif