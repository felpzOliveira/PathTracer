#include <interaction.h>

__bidevice__ 
SurfaceInteraction::SurfaceInteraction(const Point3f &p, const vec3f &pError,
                                       const Point2f &uv, const vec3f &wo,
                                       const vec3f &dpdu, const vec3f &dpdv,
                                       const Normal3f &dndu, const Normal3f &dndv, 
                                       Float time, const Shape *sh, int faceIndex)
: Interaction(p, Normal3f(Normalize(Cross(dpdu, dpdv))), pError, wo, time),
uv(uv), dpdu(dpdu), dpdv(dpdv), dndu(dndu), dndv(dndv), 
shape(shape), faceIndex(faceIndex){}