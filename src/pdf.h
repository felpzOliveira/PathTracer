#if !defined(PDF_H)
#define PDF_H

#include <types.h>
#include <onb.h>
#include <geometry.h>

/*
* NOTE: Not being used
* NOTE: I started implementing this but I got tired, so I'll use 
* as brute force path tracer instead of a fast sampler.
*/

/////////////////////////////////////////////////////////////////////////////////////
//                                   C O S I N E                                   //
/////////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ void cosine_pdf_init(Pdf *pdf){
    if(pdf){
        pdf->type = PDF_COSINE;
    }
}

inline __host__ __device__ void cosine_pdf_init(Pdf *pdf, glm::vec3 normal){
    cosine_pdf_init(pdf);
    onb_from_w(&pdf->uvw, normal);
}

inline __device__ glm::vec3 cosine_pdf_generate(Pdf *pdf, curandState *state){
    glm::vec3 dir = onb_local(&pdf->uvw, random_cosine_direction(state));
    return dir;
}

inline __device__ glm::vec3 cosine_pdf_generate(Pdf *pdf, glm::vec3 normal, 
                                                curandState *state)
{
    onb_from_w(&pdf->uvw, normal);
    glm::vec3 dir = onb_local(&pdf->uvw, random_cosine_direction(state));
    return glm::normalize(dir);
}

inline __device__ float cosine_pdf_value(Pdf *pdf, glm::vec3 direction,
                                         hit_record *record)
{
    float cosine = glm::dot(onb_w(&pdf->uvw), direction);
    float pdfv = cosine / M_PI;
    return pdfv;
}

/////////////////////////////////////////////////////////////////////////////////////
//                                   O B J E C T S                                 //
/////////////////////////////////////////////////////////////////////////////////////

//TODO
inline __host__ __device__ void object_pdf_init(Pdf *pdf, Scene *scene,
                                                Object object)
{
    if(pdf){
        pdf->type = PDF_OBJECT;
        pdf->object = object;
    }
}

inline __device__ float xz_rect_pdf_value(Pdf *pdf, Scene *scene, hit_record *record,
                                          glm::vec3 o, glm::vec3 v)
{
    Rectangle *rect = &scene->rectangles[pdf->object.object_handle];
    hit_record rec;
    Ray r;
    r.origin = o;
    r.direction = v;
    if(hit_xz_rect(rect, r, 0.001f, FLT_MAX, &rec)){
        float lx = rect->x1 - rect->x0;
        float lz = rect->z1 - rect->z0;
        float area = lx * lz;
        float v2 = v.x * v.x + v.y * v.y + v.z * v.z;
        float distance2 = rec.t * rec.t * v2;
        float cosine = glm::abs(glm::dot(v, rec.normal) / glm::length(v));
        return distance2 / (cosine * area);
    }
    
    return 0.0f;
}

inline __device__ glm::vec3 xz_rect_pdf_generate(Pdf *pdf, Scene *scene,
                                                 glm::vec3 o, curandState *state)
{
    Rectangle *rect = &scene->rectangles[pdf->object.object_handle];
    float x = rect->x0 + random_float(state) * (rect->x1 - rect->x0);
    float z = rect->z0 + random_float(state) * (rect->z1 - rect->z0);
    float y = rect->k;
    return glm::vec3(x, y, z) - o;
}

inline __device__ float sphere_pdf_value(Pdf *pdf, Scene *scene, hit_record *record,
                                         glm::vec3 o, glm::vec3 v)
{
    Sphere *sphere = &scene->spheres[pdf->object.object_handle];
    hit_record rec;
    Ray r;
    r.origin = o;
    r.direction = v;
    float radius = sphere->radius;
    if(hit_sphere(sphere, r, 0.001f, FLT_MAX, &rec)){
        glm::vec3 p = sphere->center - o;
        float d2 = p.x*p.x + p.y*p.y + p.z*p.z;
        float cos_theta_max = sqrt(1 - radius*radius/d2);
        float solid_angle = 2*M_PI*(1-cos_theta_max);
        return  1 / solid_angle;
    }
    
    return 0;
}

inline __device__ glm::vec3 sphere_pdf_generate(Pdf *pdf, Scene *scene, 
                                                glm::vec3 o, curandState *state)
{
    Sphere *sphere = &scene->spheres[pdf->object.object_handle];
    glm::vec3 dir = sphere->center - o;
    float distance2 = dir.x*dir.x + dir.y*dir.y + dir.z*dir.z;
    Onb uvw;
    onb_from_w(&uvw, dir);
    return onb_local(&uvw, random_to_sphere(sphere->radius, distance2, state));
}

inline __device__ float object_pdf_value(Pdf *pdf, Scene *scene, hit_record *record,
                                         glm::vec3 o, glm::vec3 v)
{
    switch(pdf->object.object_type){
        case OBJECT_XZ_RECTANGLE: return xz_rect_pdf_value(pdf, scene, record, o, v);
        case OBJECT_SPHERE: return sphere_pdf_value(pdf, scene, record, o, v);
        default: return 0.0f;
    }
}

inline __device__ glm::vec3 object_pdf_genereate(Pdf *pdf, Scene *scene,
                                                 glm::vec3 o, curandState *state)
{
    switch(pdf->object.object_type){
        case OBJECT_XZ_RECTANGLE: return xz_rect_pdf_generate(pdf, scene, o, state);
        case OBJECT_SPHERE: return sphere_pdf_generate(pdf, scene, o, state);
        default: return glm::vec3(1.0f, 0.0f, 0.0f);
    }
}

//TODO
inline __device__ glm::vec3 pdf_generate(Pdf *pdf, Scene *scene, hit_record *record,
                                         curandState *state)
{
    switch(pdf->type){
        case PDF_COSINE: return cosine_pdf_generate(pdf, record->normal, state);
        case PDF_OBJECT: return object_pdf_genereate(pdf, scene, record->p, state);
        default: return glm::vec3(0.0f);
    }
}

inline __device__ float pdf_value(Pdf *pdf, Scene *scene, hit_record *record,
                                  glm::vec3 scatterDir)
{
    switch(pdf->type){
        case PDF_COSINE: return cosine_pdf_value(pdf, scatterDir, record);
        case PDF_OBJECT: return object_pdf_value(pdf, scene, record,
                                                 record->p, scatterDir);
        default: return 0.0f;
    }
}

inline __device__ glm::vec3 pdf_generate_mixture(Pdf *pdf0, Pdf *pdf1, 
                                                 Scene *scene, hit_record *record,
                                                 curandState *state)
{
    float rng = random_float(state);
    if(rng < 0.5f){
        return pdf_generate(pdf0, scene, record, state);
    }else{
        return pdf_generate(pdf1, scene, record, state);
    }
}

inline __device__ float pdf_value_mixture(Pdf *pdf0, Pdf *pdf1, Scene *scene,
                                          hit_record *record, glm::vec3 dir)
{
    float f0 = pdf_value(pdf0, scene, record, dir);
    float f1 = pdf_value(pdf1, scene, record, dir);
    return 0.5f * f0 + 0.5f * f1;
}

#endif