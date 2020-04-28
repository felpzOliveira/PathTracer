#if !defined(BOX_H)
#define BOX_H

#include <types.h>

inline __host__ void xy_rect_set(Rectangle *rect, float x0, float x1, 
                                 float y0, float y1, float k,
                                 object_handle handle, 
                                 material_handle mat_handle,
                                 int flip_normals)
{
    if(rect){
        rect->z0 = k;
        rect->z1 = k;
        rect->x0 = x0;
        rect->x1 = x1;
        rect->y0 = y0;
        rect->y1 = y1;
        rect->k = k;
        rect->rect_type = OBJECT_XY_RECTANGLE;
        rect->handle = handle;
        rect->mat_handle = mat_handle;
        rect->flip_normals = flip_normals;
    }
}

inline __host__ void xz_rect_set(Rectangle *rect, float x0, float x1,
                                 float z0, float z1, float k,
                                 object_handle handle,
                                 material_handle mat_handle,
                                 int flip_normals)
{
    if(rect){
        rect->z0 = z0;
        rect->z1 = z1;
        rect->x0 = x0;
        rect->x1 = x1;
        rect->y0 = k;
        rect->y1 = k;
        rect->k = k;
        rect->rect_type = OBJECT_XZ_RECTANGLE;
        rect->handle = handle;
        rect->mat_handle = mat_handle;
        rect->flip_normals = flip_normals;
    }
}

inline __host__ void yz_rect_set(Rectangle *rect, float y0, float y1,
                                 float z0, float z1, float k,
                                 object_handle handle,
                                 material_handle mat_handle,
                                 int flip_normals)
{
    if(rect){
        rect->z0 = z0;
        rect->z1 = z1;
        rect->x0 = k;
        rect->x1 = k;
        rect->y0 = y0;
        rect->y1 = y1;
        rect->k = k;
        rect->rect_type = OBJECT_YZ_RECTANGLE;
        rect->handle = handle;
        rect->mat_handle = mat_handle;
        rect->flip_normals = flip_normals;
    }
}

inline __host__ void box_set(Box *box, glm::vec3 p0, glm::vec3 p1, 
                             object_handle handle, material_handle mat)
{
    if(box){
        box->p0 = p0;
        box->p1 = p1;
        box->handle = handle;
        box->mat_handle = mat;
        
        xy_rect_set(&box->rects[0], p0.x, p1.x, p0.y, 
                    p1.y, p1.z, 0, mat, 0);
        
        xy_rect_set(&box->rects[1], p0.x, p1.x, p0.y, 
                    p1.y, p0.z, 0, mat, 1);
        
        xz_rect_set(&box->rects[2], p0.x, p1.x, p0.z, 
                    p1.z, p1.y, 0, mat, 0);
        
        xz_rect_set(&box->rects[3], p0.x, p1.x, p0.z, 
                    p1.z, p0.y, 0, mat, 1);
        
        yz_rect_set(&box->rects[4], p0.y, p1.y, p0.z, 
                    p1.z, p1.x, 0, mat, 0);
        
        yz_rect_set(&box->rects[5], p0.y, p1.y, p0.z, 
                    p1.z, p0.x, 0, mat, 1);
    }
}

#endif