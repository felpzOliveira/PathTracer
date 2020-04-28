#if !defined(AABB_H)
#define AABB_H
#include <types.h>
#include <cutil.h>

inline __bidevice__
void aabb_init(AABB *aabb){
    if(aabb){
        aabb->_min = glm::vec3(0.f);
        aabb->_max = glm::vec3(0.f);
        aabb->centroid = glm::vec3(0.f);
    }
}

inline __bidevice__
void aabb_init(AABB *aabb, glm::vec3 a){
    if(aabb){
        aabb->_min = a;
        aabb->_max = a;
        aabb->centroid = a;
    }
}

inline __bidevice__
void aabb_init(AABB *aabb, glm::vec3 a, glm::vec3 b){
    if(aabb){
        aabb->_min = a;
        aabb->_max = b;
        aabb->centroid = (a+b)/2.f;
    }
}

inline __bidevice__
void aabb_init(AABB *aabb, glm::vec3 a, glm::vec3 b, 
               ObjectType type)
{
    if(aabb){
        aabb->_min = a;
        aabb->_max = b;
        aabb->centroid = (a+b)/2.f;
    }
}

inline __bidevice__
float aabb_max_length(AABB aabb){
    float x = aabb._max.x - aabb._min.x;
    float y = aabb._max.y - aabb._min.y;
    float z = aabb._max.z - aabb._min.z;
    if(x > y && x > z) return x;
    if(y > z) return y;
    return z;
}

inline __bidevice__
int aabb_max_extent(AABB aabb){
    float x = aabb._max.x - aabb._min.x;
    float y = aabb._max.y - aabb._min.y;
    float z = aabb._max.z - aabb._min.z;
    if(x > y && x > z) return 0;
    if(y > z) return 1;
    return 2;
}

inline __bidevice__
glm::vec3 aabb_offset(AABB aabb, glm::vec3 p){
    glm::vec3 o = p - aabb._min;
    if(aabb._max.x > aabb._min.x) o.x /= aabb._max.x - aabb._min.x;
    if(aabb._max.y > aabb._min.y) o.y /= aabb._max.y - aabb._min.y;
    if(aabb._max.z > aabb._min.z) o.z /= aabb._max.z - aabb._min.z;
    return o;
}

inline __bidevice__ 
AABB surrounding_box(AABB b, glm::vec3 p){
    glm::vec3 pmin(b._min.x > p.x ? p.x : b._min.x,
                   b._min.y > p.y ? p.y : b._min.y,
                   b._min.z > p.z ? p.z : b._min.z);
    
    glm::vec3 pmax(b._max.x > p.x ? b._max.x : p.x,
                   b._max.y > p.y ? b._max.y : p.y,
                   b._max.z > p.z ? b._max.z : p.z);
    AABB aabb;
    aabb_init(&aabb, pmin, pmax);
    return aabb;
}

inline __bidevice__
void surrounding_box(AABB *output, 
                     AABB *box0, 
                     AABB *box1)
{
    glm::vec3 small( ffmin(box0->_min.x, box1->_min.x),
                    ffmin(box0->_min.y, box1->_min.y),
                    ffmin(box0->_min.z, box1->_min.z));
    glm::vec3 big  ( ffmax(box0->_max.x, box1->_max.x),
                    ffmax(box0->_max.y, box1->_max.y),
                    ffmax(box0->_max.z, box1->_max.z));
    
    aabb_init(output, small, big, OBJECT_AABB);
}

inline __bidevice__
bool sphere_bounding_box(Sphere *sphere, AABB *aabb){
    aabb_init(aabb, sphere->center - glm::vec3(glm::abs(sphere->radius)),
              sphere->center + glm::vec3(glm::abs(sphere->radius)), 
              OBJECT_SPHERE);
    return true;
}

inline __bidevice__
bool xy_rect_bounding_box(Rectangle *rect, AABB *aabb){
    aabb_init(aabb, glm::vec3(rect->x0, rect->y0, rect->k-0.001f),
              glm::vec3(rect->x1, rect->y1, rect->k+0.001f),
              OBJECT_XY_RECTANGLE);
    return true;
}

inline __bidevice__
bool xz_rect_bounding_box(Rectangle *rect, AABB *aabb){
    aabb_init(aabb, glm::vec3(rect->x0, rect->k-0.001f, rect->z0),
              glm::vec3(rect->x1, rect->k+0.001f, rect->z1),
              OBJECT_XZ_RECTANGLE);
    return true;
}

inline __bidevice__
bool yz_rect_bounding_box(Rectangle *rect, AABB *aabb){
    aabb_init(aabb, glm::vec3(rect->k-0.001f, rect->y0, rect->z0),
              glm::vec3(rect->k+0.001f, rect->y1, rect->z1),
              OBJECT_YZ_RECTANGLE);
    return true;
}

inline __bidevice__
bool triangle_bounding_box(Triangle *tri, AABB *aabb){
    glm::vec3 v[3];
    v[0] = tri->v0; v[1] = tri->v1; v[2] = tri->v2;
    glm::vec3 min(FLT_MAX), max(-FLT_MAX);
    
    for(int k = 0; k < 3; k += 1){
        glm::vec3 p = v[k];
        for(int i = 0; i < 3; i += 1){
            if(min[i] > p[i]){
                min[i] = p[i];
            }
            
            if(max[i] < p[i]){
                max[i] = p[i];
            }
        }
    }
    
    //assure box is not 2D
    for(int i = 0; i < 3; i += 1){
        min[i] -= 0.001f;
        max[i] += 0.001f;
    }
    aabb_init(aabb, min, max, OBJECT_TRIANGLE);
    return true;
}


//NOTE: I don't want to handle random rotations, so I'll compute the maximum diagonal
//      and expand the minimal bounding box by diag this should be able to
//      get the maximum transformation for 45 degrees on the bigger side and get
//      any other configuration. The resulting bounding box is actually much bigger
//      than the original box but it is a simple hacky way so we don't need to care
//      too much about transformations.
inline __bidevice__
bool box_bounding_box(Box *box, AABB *aabb){
    glm::vec3 v0 = toWorld(box->p0, box->transforms);
    glm::vec3 v1 = toWorld(box->p1, box->transforms);
    
    float lx = v1.x - v0.x;
    float ly = v1.y - v0.y;
    float lz = v1.z - v0.z;
    
    float m = lx > ly ? lx : ly;
    m = m > lz ? m : lz;
    m *= 1.0f/glm::sqrt(2.0f);
    v0 = v0 - glm::vec3(m);
    v1 = v1 + glm::vec3(m);
    aabb_init(aabb, v0, v1, OBJECT_BOX);
    
    return true;
}

//NOTE: Update when adding new primitives
inline __host__ 
void _get_aabb(Scene *scene, Object obj, AABB *aabb){
    if(obj.object_type == OBJECT_SPHERE){
        Sphere *sphere = &scene->spheres[obj.handle];
        sphere_bounding_box(sphere, aabb);
    }else if(obj.object_type == OBJECT_XY_RECTANGLE){
        Rectangle *rect = &scene->rectangles[obj.handle];
        xy_rect_bounding_box(rect, aabb);
    }else if(obj.object_type == OBJECT_XZ_RECTANGLE){
        Rectangle *rect = &scene->rectangles[obj.handle];
        xz_rect_bounding_box(rect, aabb);
    }else if(obj.object_type == OBJECT_YZ_RECTANGLE){
        Rectangle *rect = &scene->rectangles[obj.handle];
        yz_rect_bounding_box(rect, aabb);
    }else if(obj.object_type == OBJECT_BOX){
        Box *box = &scene->boxes[obj.handle];
        box_bounding_box(box, aabb);
    }else if(obj.object_type == OBJECT_TRIANGLE){
        Triangle *tri = &scene->triangles[obj.handle];
        triangle_bounding_box(tri, aabb);
    }else if(obj.object_type == OBJECT_MESH){
        Mesh *mesh = scene->meshes[obj.handle];
        aabb_init(aabb, mesh->aabb._min, mesh->aabb._max, OBJECT_MESH);
    }
}

//NOTE: For meshes this must be a triangle
inline __host__ 
void get_aabb(Mesh *mesh, Object obj, AABB *aabb){
    if(obj.object_type != OBJECT_TRIANGLE){
        std::cout << "Mesh invocation with type != OBJECT_TRIANGLE"<< std::endl;
        exit(0);
    }
    
    Triangle *tri = &mesh->triangles[obj.handle];
    triangle_bounding_box(tri, aabb);
}

inline __host__ 
void get_aabb(Scene *scene, Object obj, AABB *aabb){
    if(obj.object_type != OBJECT_MEDIUM){
        _get_aabb(scene, obj, aabb);
    }else{
        Medium *med = &scene->mediums[obj.handle];
        _get_aabb(scene, med->geometry, aabb);
    }
}

#endif
