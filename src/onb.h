#if !defined(ONB_H)
#define ONB_H
#include <cutil.h>

#include <types/type_onb.h>

inline __bidevice__ glm::vec3 onb_u(Onb *onb){ return onb->axis[0]; }
inline __bidevice__ glm::vec3 onb_v(Onb *onb){ return onb->axis[1]; }
inline __bidevice__ glm::vec3 onb_w(Onb *onb){ return onb->axis[2]; }


inline __bidevice__ void onb_from_w(Onb *onb, glm::vec3 n){
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

inline __bidevice__ glm::vec3 onb_local(Onb *onb, glm::vec3 a){
    glm::vec3 u = onb->axis[0];
    glm::vec3 v = onb->axis[1];
    glm::vec3 w = onb->axis[2];
    
    return a.x * u + a.y * v + a.z * w;
}


inline __bidevice__ glm::vec3 onb_local_to_world(Onb *uvw, glm::vec3 v){
    glm::vec3 ns = onb_w(uvw);
    glm::vec3 ss = onb_u(uvw);
    glm::vec3 ts = onb_v(uvw);
    return glm::vec3(ss.x * v.x + ts.x * v.y + ns.x * v.z,
                     ss.y * v.x + ts.y * v.y + ns.y * v.z,
                     ss.z * v.x + ts.z * v.y + ns.z * v.z);
}

inline __bidevice__ glm::vec3 onb_world_to_local(Onb *uvw, glm::vec3 v){
    glm::vec3 ns = onb_w(uvw);
    glm::vec3 ss = onb_u(uvw);
    glm::vec3 ts = onb_v(uvw);
    return glm::vec3(glm::dot(v, ss), 
                     glm::dot(v, ts),
                     glm::dot(v, ns));
}


#endif