#if !defined(ONB_H)
#define ONB_H

#include <types.h>

inline __host__ __device__ void onb_from_w(Onb *onb, glm::vec3 n){
    onb->axis[2] = glm::normalize(n);
    glm::vec3 a;
    if(glm::abs(onb->axis[2].x) > 0.9f){
        a = glm::vec3(0.0f, 1.0f, 0.0f);
    }else{
        a = glm::vec3(1.0f, 0.0f, 0.0f);
    }
    onb->axis[1] = glm::normalize(glm::cross(onb->axis[2], a));
    onb->axis[0] = glm::normalize(glm::cross(onb->axis[2], onb->axis[1]));
}

inline __host__ __device__ glm::vec3 onb_local(Onb *onb, glm::vec3 a){
    glm::vec3 u = onb->axis[0];
    glm::vec3 v = onb->axis[1];
    glm::vec3 w = onb->axis[2];
    
    return a.x * u + a.y * v + a.z * w;
}

inline __host__ __device__ glm::vec3 onb_u(Onb *onb){ return onb->axis[0]; }
inline __host__ __device__ glm::vec3 onb_v(Onb *onb){ return onb->axis[1]; }
inline __host__ __device__ glm::vec3 onb_w(Onb *onb){ return onb->axis[2]; }

#endif